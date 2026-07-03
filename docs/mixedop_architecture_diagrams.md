# QNAS: Discrete (PMF) vs MixedOp (PDF) — Architecture & Flow Diagrams

Comparison of the original discrete-sampling QNAS design and the new `mixedop_mode`
(DARTS-style MixedOperation with alpha-based weight reuse).

## 1. Network architecture — nodes & edges

### Old: discrete chain (one operation sampled per node)

```mermaid
graph LR
    I[Input] --> N1["Node 1<br/>op: conv_3_1_64"]
    N1 --> N2["Node 2<br/>op: max_pool_2_2"]
    N2 --> N3["Node 3<br/>op: conv_3_1_128"]
    N3 --> FC[FC / Output]
```

Each node has exactly **one incoming edge type** — the operation chosen by
`np.random.choice` from the PMF (`src/population.py: generate_classical`).

### New: MixedOp node (all operations in parallel, softmax(alpha)-weighted edges)

```mermaid
graph LR
    X0["Node i output<br/>(in_channels)"] -->|alpha_1| O1[conv_1_1_32]
    X0 -->|alpha_2| O2[conv_3_1_64]
    X0 -->|alpha_3| O3[conv_3_1_128]
    X0 -->|alpha_4| O4[max_pool_2_2]
    X0 -->|alpha_5| O5[avg_pool_2_2]
    X0 -->|alpha_6| O6[no_op]
    O1 --> P1["1x1 proj → canonical_ch"]
    O2 --> P2["1x1 proj → canonical_ch"]
    O3 --> P3["identity (already canonical_ch)"]
    O4 --> P4["adaptive_pool + 1x1 proj"]
    O5 --> P5["adaptive_pool + 1x1 proj"]
    O6 --> P6["1x1 proj → canonical_ch"]
    P1 --> SUM(("Σ softmax(alpha)·op_i"))
    P2 --> SUM
    P3 --> SUM
    P4 --> SUM
    P5 --> SUM
    P6 --> SUM
    SUM --> X1["Node i+1 input"]
```

Every node fans out to **all candidate ops simultaneously** (edges weighted by
`softmax(alpha)`), aligned in channels/spatial size, then summed — this is `MixedOp` in
[`src/cnn/mixed_model.py`](../src/cnn/mixed_model.py).

### Memory optimization: capped canonical channel width (PC-DARTS-style)

By default, `canonical_out_channels` at each node is `max()` over every candidate op's
output width. With a `fn_list` that includes a wide conv (e.g. `conv_3_1_256`), **every**
op — including cheap ones like `no_op` and `max_pool_2_2` — gets projected up to 256
channels before the weighted sum, and that width then becomes the next node's
`in_channels`, compounding across all `max_num_nodes` layers.

```mermaid
graph LR
    subgraph "Uncapped (canonical = max = 256)"
    direction LR
    A1[no_op] -->|"1x1 proj → 256ch"| S1(("Σ"))
    A2[conv_1_1_32] -->|"1x1 proj → 256ch"| S1
    A3[conv_3_1_256] -->|"identity"| S1
    S1 --> N1["node i+1<br/>in_channels=256"]
    end
    subgraph "Capped (mixedop_max_channels=64)"
    direction LR
    B1[no_op] -->|"1x1 proj → 64ch"| S2(("Σ"))
    B2[conv_1_1_32] -->|"1x1 proj → 64ch"| S2
    B3[conv_3_1_256] -->|"1x1 proj → 64ch"| S2
    S2 --> N2["node i+1<br/>in_channels=64"]
    end
```

Setting `mixedop_max_channels` in the config's `train:` section (see
`config_mixedop.yml`) caps `canonical_out_channels` at that value instead of following
the widest op. This is the same idea PC-DARTS uses to bound search-time memory: every
op's output — even the naturally-wide one — gets projected down to the cap, so channel
width no longer grows unboundedly with the widest entry in `fn_list`. Because the capped
width also becomes the next node's `in_channels`, the saving compounds with depth
(conv FLOPs/activation memory scale with `in_channels × out_channels`), which is why a
modest cap (e.g. 64 vs. 256) cuts total parameters by ~75% in practice.

Trade-off: the wide op's own internal computation is unaffected (it still runs at its
configured filter count, e.g. 256, before being projected down), so it doesn't reduce
that op's own peak activation — only what propagates onward. It does mean a wide op's
full representation is squeezed through the same bottleneck as every other candidate,
which can bias the learned alpha logits against wide/expensive ops during search. This
only affects the search-time proxy: `collapse_best_network()` only keeps the winning
*operation names*, and the retrain phase rebuilds the discrete network via
`NetworkGraph.create_functions()` at each op's full native width — uncapped.
Implementation: `MixedOp`/`MixedNetworkGraph.create_mixed_ops()` in
[`src/cnn/mixed_model.py`](../src/cnn/mixed_model.py); read from the config in
[`src/qnas_config.py`](../src/qnas_config.py) (config-file only, no CLI flag).

### End of evolution: collapse back to a discrete chain

```mermaid
graph LR
    A["best_alpha[node]<br/>(trained, per node)"] --> B["argmax per node"]
    B --> C["Single-op chain<br/>(same shape as the old discrete network)"]
```

This is the `collapse_best_network()` step in [`src/qnas.py`](../src/qnas.py), run once
at the end of `evolve()` when `mixedop_mode` is enabled.

## 2. Evolutionary flow

### Old flow

```mermaid
flowchart TD
    Q1["Quantum probabilities (PMF)<br/>[num_ind, nodes, funcs]"] --> S["Sample discrete ops<br/>np.random.choice per node"]
    S --> D["Decode → net_list (fn names)"]
    D --> T["Train NetworkGraph<br/>(single op per layer)"]
    T --> F["Fitness: accuracy/loss/params"]
    F --> U["update_quantum()<br/>shift PMF toward best individuals"]
    U --> Q1
```

### New flow

```mermaid
flowchart TD
    Q2["Quantum alpha_logits (PDF)<br/>[num_ind, nodes, funcs]"] --> G["generate_classical_mixedop()<br/>alpha_logits + N(0, sigma)"]
    G --> WB{"WeightBank: cosine-close<br/>alpha match found?"}
    WB -- yes --> LW["Load closest weights (strict=False)<br/>fine-tune fewer epochs"]
    WB -- no --> FS["Train from scratch"]
    LW --> M["MixedNetworkGraph<br/>(all ops parallel, softmax(alpha) weighted)"]
    FS --> M
    M --> TR["Train weights AND alpha<br/>(gradient descent)"]
    TR --> SA["Save trained_alpha.npy<br/>+ register weights in bank"]
    SA --> F2["Fitness: accuracy/loss/params"]
    F2 --> UQ["update_quantum_from_alpha()<br/>shift alpha_logits toward best trained alpha"]
    UQ --> Q2
    F2 -. last generation .-> COL["collapse_best_network()<br/>argmax per node"]
    COL --> FIN["Final discrete network<br/>(one op per node)"]
```
