# EXPERIMENTS.md — índice das configs de `configs/config_files_cifar/`

Configs de CIFAR-10 com progressive growth, na ordem em que as ideias foram testadas. Cada linha diz o que
a config muda em relação à antecessora, o que se esperava e o que aconteceu. **Mantenha este arquivo em dia
a cada config nova ou resultado novo** (ver `CLAUDE.md`). O cabeçalho de cada YAML traz os mesmos detalhes.

**Convenções:**
- **Run original:** run de `experiment_cifar10_progressive/` (setembro de 2026). Para os runs de pqnas, o
  retrain sofreu early stopping e saiu inválido.
- **pqNN:** reexecução de outubro de 2026 em `experiment_cifar_pqnas/NN_*`.
- **Retrain:** melhor validação entre 5 retrains no protocolo completo (augmentation + MultiStepLR,
  300 épocas). **Teste:** média ± dp no conjunto de teste do CIFAR-10.
- Análise completa: [reports/analise_experiment_cifar_pqnas.md](reports/analise_experiment_cifar_pqnas.md).
- Plano das configs 18–26: [docs/QNAS_SEARCH_ENGINE_IMPROVEMENT_PLAN_V2.md](docs/QNAS_SEARCH_ENGINE_IMPROVEMENT_PLAN_V2.md).
- **Referências:** Daniela 80,30 / 92,96 / 58 GPU-days / 1,60M; Diego (controle) 79,27 / 92,43 /
  1,70 GPU-days / 1,81M.

## Configs 00–15: busca progressiva

| Config | O que muda | Run original | Reexecução pq (busca / retrain / teste / params) | Status |
|---|---|---|---|---|
| 00_smoketest | versão mínima de 01 para testar o pipeline | — | — | utilitário |
| 01_deterministic_13-8-4 | 13→8→4 ops, carryover | exp6: 77,10 / 90,70 | pq01: 76,09 / 90,18 / 89,25 ± 0,36 / 1,61M | Success |
| 02_..._ancestor-decay | 01 + `ancestor_decay` | exp4_ancestor_decay: 77,00 / 84,48† | pq02: 78,17 / 93,46 / 92,54 ± 0,10 / 1,93M | Success (era Failed†) |
| 03_..._reset_ancestor-decay | 13→8→4, reset + AD | exp_reset_ad_1/_2: 79,08 / 93,32; 78,39 / 94,42 | pq03: 79,27 / 91,72 / 90,76 ± 0,24 / 1,08M | Success |
| 04_deterministic_13-10-8_reset | 13→10→8, reset | exp7: 77,20 / 90,30† | pq04: 77,49 / 92,80 / 91,67 ± 0,14 / 1,04M | Success (era Failed†) |
| 05_..._reset_ancestor-decay | 04 + AD | exp7_ancestor_decay: 77,89 / 88,96† | pq05: 77,89 / 94,16 / 93,12 ± 0,07 / 1,79M | Success (era Failed†) |
| 06_dynamic_v1 | modo dinâmico v1 | exp_dynamic_1: 72,70 / 82,48† | pq06: 70,20 / 83,80 / 83,47 ± 0,20 / 0,43M | Failed (menu colapsa) |
| 07_dynamic_v2 | dinâmico v2 (poda conservadora) | exp_dynamic_2: 75,60 / 82,46† | pq07: 76,00 / 89,96 / 89,30 ± 0,15 / 0,72M | Failed (cresce pouco) |
| 08_dynamic_v2_ancestor-decay | 07 + AD | 75,00 / 82,88† | pq08: 75,70 / 89,08 / 88,32 ± 0,14 / 1,02M | Failed |
| 11_deterministic_13-8-4_reset | reconstrução da config de exp4/exp5 | exp4: 78,70 / 92,90; exp5: 76,40 / 93,60 | pq11: 79,88 / 93,20 / 92,56 ± 0,11 / 2,23M | Success (a mais estável, 3 seeds) |
| 12_..._reset_global-prune | 11 + `global_op_pruning` | — | pq12: 79,39 / 91,56 / 90,68 ± 0,20 / 0,74M | Success (rede 3× menor) |
| 13_deterministic_13-10-8_carryover | 04 com carryover | — | **pq13: 77,49 / 94,66 / 93,70 ± 0,10 / 2,66M** | **Success — melhor run** |
| 14_dynamic_v2_reset | 07 + reset | — | pq14: 78,59 / 90,22 / 89,72 ± 0,21 / 1,00M | Failed |
| 15_tese3-exp1_reconstruction_GUESS | reconstrução do fabio_tese3/exp1 (100 gerações, sem crossover) | exp1: 73,80 / 89,40 | pq15: 76,29 / 92,22 / 91,26 ± 0,28 / 1,01M | Invalid (fora do protocolo) |
| 16_config_progressive_rank_update | base exp4/exp5 + Melhoria 1 do plano v1 | — | não rodada | — |
| 17_..._rank_update_ancestor_decay | base exp_reset_ad + Melhoria 1 | — | não rodada | — |

† Retrain antigo interrompido por early stopping, sem augmentation. O config não tinha falhado: na
reexecução com o protocolo completo, ele passa das referências.

**O que aprendemos com 00–15:**
- A fitness de busca prevê mal o retrain (ρ = 0,57). FLOPs e params preveem melhor (ρ = 0,83 / 0,76).
- O modelo quântico quase não aprende: entropia final ≈ 0,93–0,999 e KL ≤ 0,02.
- O ruído entre seeds chega a 2,7 pt de retrain.

## Configs 18–26: plano v2 (busca mais rápida e melhor)

Branch `pqnas-search-v2`. Cada config parte de um dos melhores runs e acrescenta uma combinação das
direções do plano v2:
- **D1:** triagem por surrogate com seleção de Boltzmann (`QNAS.surrogate`);
- **D2:** proxy de fitness (`train.fitness_aggregation`, `train.limit_val_value`) e fidelidade em duas
  rodadas (`QNAS.fidelity`);
- **D3:** PMF que aprende (`quantum_lr` e afins, `prune_criterion: lift`) e temperatura de amostragem
  (`QNAS.sampling_temperature`);
- **D4:** nichos (`replace_method: niche`, `QNAS.niche`).

Para conferir o que difere da base: `make diff-runs A=<base> B=<config>`.

| Config | Base | Direções | Hipótese | Resultado |
|---|---|---|---|---|
| 18_v2_pq13_d1 | pq13 | D1, mesmo custo (20 treinados de 100) | retrain ≥ pq13 + 0,5 pt com o mesmo custo; melhor indivíduo encontrado mais cedo | pendente |
| 19_v2_pq13_d1-half | pq13 | D1, metade do custo (10 de 50) | GPU-days ≤ 50% de pq13, teste no máximo 0,3 pt abaixo | pendente |
| 20_v2_pq13_d3 | pq13 | D3 | KL no fim do estágio > 0,2; sem perda de retrain | pendente |
| 21_v2_pq13_d1-d3 | pq13 | D1 + D3 | D3 concentra a amostragem e D1 filtra; ganho acima de 18 | pendente |
| 22_v2_pq13_d1-d3-d2proxy | pq13 | D1 + D3 + D2 (média das últimas 5 épocas, validação de 5k) | proxy menos ruidoso → ρ busca–retrain maior | pendente |
| 23_v2_pq13_d1-d3-d2 | pq13 | 22 + fidelidade (2× por 10 épocas, a melhor metade continua) | mais candidatos vistos por GPU-hora | pendente |
| 24_v2_pq13_all | pq13 | 23 + D4 (nichos, `max_age` 30, reinício após 40 gerações) | evita colapso causado por D1/D3 | pendente |
| 25_v2_pq05_all | pq05 | todas | confirmar em outra base (reset + AD) | pendente |
| 26_v2_pq11_d1 | pq11 | D1, mesmo custo | efeito de D1 na família mais estável (13→8→4 reset) | pendente |

**Protocolo:**
- 3 seeds por braço.
- Retrain default do `run_pipeline.sh`: augmentation + multistep, `-R 5`.
- GPU-days medidos pelo tempo de GPU logado (`GPU time so far` no `log_QNAS.txt`) ou com um job por
  máquina.
- Antes de 22–25, rodar o estudo de calibração do proxy (fase F1 do plano). Os valores de D2 nessas
  configs são provisórios.
