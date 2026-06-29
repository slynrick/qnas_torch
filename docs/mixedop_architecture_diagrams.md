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
