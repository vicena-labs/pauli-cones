# Fair rerun, Apple M1 Pro (6 performance + 2 efficiency cores), macOS

Warm medians of 5 fresh parameter vectors per run, one fresh process per run, 120 s limit. Pauli Cones ran at 1 thread; every competitor ran at 1 and 8 threads (pauli-prop is single-threaded). A competitor's figure is its faster configuration. PC graph rows count only runs where the graph did not fall back to native.

**Caveats.** The laptop was not quiet: other applications used about 1.5-2 of the 8 cores throughout, so 8-thread figures are unreliable and 1-thread figures carry run-to-run variation of up to several times. During the 8-thread PauliPropagation.jl pass, Julia start-up alone took 14-97 s, so its 8-thread timeouts are artifacts, not limits of the package. Publishable numbers need a dedicated quiet machine.

## Exact values

| Workload | PC native 1t | PC graph 1t | Graph prepare | Fastest competitor (config) | Its value | PC best vs it | Refused or unsupported |
|---|---|---|---|---|---|---|---|
| A_vqe_3x4_L2 | 0.0889 ms | 0.0311 ms | 6.54 ms | PauliPropagation.jl surrogate, 1t | 0.0532 ms | 1.71× |  |
| A_vqe_4x5_L2 | 0.165 ms | 0.0344 ms | 6.5 ms | PauliPropagation.jl surrogate, 1t | 0.109 ms | 3.17× |  |
| A_vqe_4x5_L4 | 31.5 ms | 0.344 ms | 34.1 ms | Aer, 8t | 157 ms | 455× | PauliPropagation.jl (timeout); PauliPropagation.jl surrogate (timeout); pauli-prop (over cap) |
| A_vqe_4x6_L6 | over cap | over cap | — | Aer, 1t | 6.48 s | PC loses | Lightning (timeout); PC graph (over cap); PC native (over cap); PauliPropagation.jl (timeout); PauliPropagation.jl surrogate (timeout); Qulacs (timeout); pauli-prop (over cap); quimb (timeout) |
| B_noisy_2x5_L4 | 0.247 ms | 0.0385 ms | 5.37 ms | pauli-prop, 1t | 8.67 ms | 225× | PauliPropagation.jl (unsupported) |
| B_noisy_6x6_L3 | 0.362 ms | 0.052 ms | 7.24 ms | pauli-prop, 1t | 6.35 ms | 122× | PauliPropagation.jl (unsupported) |
| B_noisy_6x6_L5 | 41 ms | 0.517 ms | 55.6 ms | none finished | — | — | PauliPropagation.jl (unsupported); pauli-prop (over cap) |
| B_noisy_20x20_L3 | 6.43 ms | 0.177 ms | 51.6 ms | pauli-prop, 1t | 788 ms | 4.45e+03× | PauliPropagation.jl (unsupported) |
| B_noisy_32x32_L3 | 13.7 ms | 0.312 ms | 133 ms | none finished | — | — | PauliPropagation.jl (unsupported); pauli-prop (timeout) |
| C_centre_10x10_L4 | 0.0936 ms | 0.0623 ms | 14.6 ms | pauli-prop, 1t | 24.4 ms | 391× | PauliPropagation.jl (timeout); PauliPropagation.jl surrogate (unsupported) |
| C_centre_10x10_L6 | over cap | over cap | — | quimb, 1t | 187 ms | PC loses | PC graph (over cap); PC native (over cap); PauliPropagation.jl (timeout); PauliPropagation.jl surrogate (unsupported); pauli-prop (over cap) |
| C_centre_10x10_L8 | over cap | over cap | — | quimb, 1t | 380 ms | PC loses | PC graph (over cap); PC native (over cap); PauliPropagation.jl (timeout); PauliPropagation.jl surrogate (unsupported); pauli-prop (timeout) |
| C_centre_32x32_L6 | over cap | over cap | — | quimb, 1t | 2.17 s | PC loses | PC graph (over cap); PC native (over cap); PauliPropagation.jl (timeout); PauliPropagation.jl surrogate (unsupported); pauli-prop (timeout) |
| D_noisy1q_2x5_L4 | 0.27 ms | 0.0362 ms | 5.89 ms | PauliPropagation.jl, 1t | 2.95 ms | 81.2× | Aer (failed); Qulacs (failed) |
| D_noisy1q_6x6_L3 | 0.411 ms | 0.0418 ms | 6.87 ms | PauliPropagation.jl, 1t | 2.22 ms | 53.1× |  |
| D_noisy1q_6x6_L5 | 44.7 ms | 0.477 ms | 58.2 ms | none finished | — | — | PauliPropagation.jl (timeout); pauli-prop (over cap) |
| D_noisy1q_20x20_L3 | 7.14 ms | 0.133 ms | 61.1 ms | PauliPropagation.jl, 1t | 502 ms | 3.79e+03× |  |
| D_noisy1q_32x32_L3 | 15.7 ms | 0.353 ms | 148 ms | PauliPropagation.jl, 1t | 4.22 s | 1.2e+04× | pauli-prop (timeout) |
| D_noisy1q_6x6_L6 | over cap | over cap | — | none finished | — | — | PC graph (over cap); PC native (over cap); PauliPropagation.jl (timeout); pauli-prop (over cap) |
| D_noisy1q_10x10_L6 | over cap | over cap | — | none finished | — | — | PC graph (over cap); PC native (over cap); PauliPropagation.jl (timeout); pauli-prop (timeout) |
| K_eagle_mz_S5_t0pi8 | 66.8 ms | fell back | — | none finished | — | — | PauliPropagation.jl (timeout); PauliPropagation.jl surrogate (unsupported); pauli-prop (timeout); quimb (timeout) |
| K_eagle_mz_S5_t1pi8 | 5.58 s | fell back | — | none finished | — | — | PauliPropagation.jl (timeout); PauliPropagation.jl surrogate (unsupported); pauli-prop (timeout); quimb (timeout) |
| K_eagle_mz_S5_t2pi8 | 5.69 s | fell back | — | none finished | — | — | PauliPropagation.jl (timeout); PauliPropagation.jl surrogate (unsupported); pauli-prop (timeout); quimb (timeout) |
| K_eagle_mz_S5_t3pi8 | 5.56 s | fell back | — | none finished | — | — | PauliPropagation.jl (timeout); PauliPropagation.jl surrogate (unsupported); pauli-prop (timeout); quimb (timeout) |
| K_eagle_mz_S5_t4pi8 | 822 ms | fell back | — | none finished | — | — | PauliPropagation.jl (timeout); PauliPropagation.jl surrogate (unsupported); pauli-prop (timeout); quimb (timeout) |
| K_eagle_z62_S20_t1pi8 | over cap | over cap | — | none finished | — | — | PC graph (over cap); PC native (over cap); PauliPropagation.jl (timeout); PauliPropagation.jl surrogate (unsupported); pauli-prop (timeout); quimb (timeout) |

## Gradients (all shared parameters)

| Workload | PC 1t | Fastest competitor (config) | Its gradient | PC vs it |
|---|---|---|---|---|
| A_vqe_3x4_L2 | 0.0304 ms | Qulacs, 1t | 2.5 ms | 82.1× |
| A_vqe_4x5_L2 | 0.0347 ms | PauliPropagation.jl, 1t | 31.3 ms | 900× |
| A_vqe_4x5_L4 | 1.06 ms | Lightning, 1t | 2.32 s | 2.18e+03× |
| B_noisy_2x5_L4 | 0.0405 ms | none | — | — |
| B_noisy_6x6_L3 | 0.0415 ms | none | — | — |
| B_noisy_6x6_L5 | 1.53 ms | none | — | — |
| B_noisy_20x20_L3 | 0.178 ms | none | — | — |
| B_noisy_32x32_L3 | 0.443 ms | none | — | — |
| C_centre_10x10_L4 | 0.0532 ms | none | — | — |
| D_noisy1q_2x5_L4 | 0.0409 ms | none | — | — |
| D_noisy1q_6x6_L3 | 0.0407 ms | none | — | — |
| D_noisy1q_6x6_L5 | 1.3 ms | none | — | — |
| D_noisy1q_20x20_L3 | 0.176 ms | none | — | — |
| D_noisy1q_32x32_L3 | 0.497 ms | none | — | — |
| K_eagle_mz_S5_t0pi8 | 135 ms | none | — | — |
| K_eagle_mz_S5_t1pi8 | 6.31 s | none | — | — |
| K_eagle_mz_S5_t2pi8 | 6.63 s | none | — | — |
| K_eagle_mz_S5_t3pi8 | 6.37 s | none | — | — |
| K_eagle_mz_S5_t4pi8 | 1.73 s | none | — | — |

## Pauli Cones threads (1 → 8)

| Workload | Native 1t | Native 8t | Speed-up |
|---|---|---|---|
| A_vqe_3x4_L2 | 0.0889 ms | 0.148 ms | 0.6× |
| A_vqe_4x5_L2 | 0.165 ms | 0.217 ms | 0.76× |
| A_vqe_4x5_L4 | 31.5 ms | 40 ms | 0.79× |
| B_noisy_2x5_L4 | 0.247 ms | 2.18 ms | 0.11× |
| B_noisy_6x6_L3 | 0.362 ms | 0.736 ms | 0.49× |
| B_noisy_6x6_L5 | 41 ms | 77.6 ms | 0.53× |
| B_noisy_20x20_L3 | 6.43 ms | 2.36 ms | 2.7× |
| B_noisy_32x32_L3 | 13.7 ms | 4.06 ms | 3.4× |
| C_centre_10x10_L4 | 0.0936 ms | 0.143 ms | 0.66× |
| D_noisy1q_2x5_L4 | 0.27 ms | 0.409 ms | 0.66× |
| D_noisy1q_6x6_L3 | 0.411 ms | 0.52 ms | 0.79× |
| D_noisy1q_6x6_L5 | 44.7 ms | 57.9 ms | 0.77× |
| D_noisy1q_20x20_L3 | 7.14 ms | 2.15 ms | 3.3× |
| D_noisy1q_32x32_L3 | 15.7 ms | 4.09 ms | 3.8× |
| K_eagle_mz_S5_t0pi8 | 66.8 ms | 70 ms | 0.95× |
| K_eagle_mz_S5_t1pi8 | 5.58 s | 5.6 s | 1× |
| K_eagle_mz_S5_t2pi8 | 5.69 s | timeout | — |
| K_eagle_mz_S5_t3pi8 | 5.56 s | timeout | — |
| K_eagle_mz_S5_t4pi8 | 822 ms | timeout | — |

## Agreement between exact tools at the first vector

Largest difference between any two exact results on the same input: 4e-13 (over 23 workloads).

## Truncated, coefficient threshold only (max_weight = Inf)

Thresholds are absolute for PauliPropagation.jl and pauli-prop and converted to Pauli Cones' relative form by dividing by the observable l1 norm, so all tools drop the same coefficients. Errors are absolute, in units of the observable's l1 norm, against the exact references above where one exists. A value timing counts even when the same run later failed computing gradients.

| Workload | Threshold | PC native 1t / 8t (error) | PC graph 1t / 8t (error) | PauliPropagation.jl 1t / 8t (error) | pauli-prop 1t / 8t (error) |
|---|---|---|---|---|---|
| A_vqe_4x6_L6 | 0.0001 | over cap / over cap | fell back | timeout / timeout | timeout / — |
| A_vqe_4x6_L6 | 1e-05 | over cap / over cap | over cap / over cap | timeout / timeout | timeout / — |
| C_centre_10x10_L6 | 0.0001 | 124 ms / 190 ms (4.3e-04) | 28.1 ms / 46.7 ms (3.7e-04) | timeout / timeout | timeout / — |
| C_centre_10x10_L6 | 1e-05 | over cap / over cap | over cap / over cap | timeout / timeout | timeout / — |
| C_centre_10x10_L8 | 0.0001 | over cap / over cap | over cap / over cap | timeout / timeout | timeout / — |
| C_centre_10x10_L8 | 1e-05 | over cap / over cap | over cap / over cap | timeout / timeout | timeout / — |
| C_centre_32x32_L6 | 0.0001 | 203 ms / 177 ms (4.3e-04) | 15.4 ms / 14.1 ms (4.4e-04) | timeout / timeout | timeout / — |
| C_centre_32x32_L6 | 1e-05 | over cap / over cap | over cap / over cap | timeout / timeout | timeout / — |
| D_noisy1q_6x6_L6 | 0.0001 | 154 ms / 288 ms (no ref) | 15.6 ms / 16 ms (no ref) | 248 ms / 397 ms (no ref) | 1.75 s / — (no ref) |
| D_noisy1q_6x6_L6 | 1e-05 | timeout / 1.71 s (no ref) | timeout / fell back | 6.03 s / 4.75 s (no ref) | 11.2 s / — (no ref) |
| D_noisy1q_10x10_L6 | 0.0001 | 204 ms / 137 ms (no ref) | 10.8 ms / 8.46 ms (no ref) | 630 ms / 761 ms (no ref) | 2.26 s / — (no ref) |
| D_noisy1q_10x10_L6 | 1e-05 | timeout / timeout | timeout / timeout | 16.1 s / 15.8 s (no ref) | timeout / — |
| K_eagle_mz_S5_t1pi8 | 0.0001 | 66 ms / 97.3 ms (5.4e-03) | 1.33 ms / 1.5 ms (5.0e-04) | 190 ms / 270 ms (7.0e-03) | 384 ms / — (9.8e-03) |
| K_eagle_mz_S5_t1pi8 | 1e-05 | 459 ms / 315 ms (9.0e-04) | 15 ms / 9.44 ms (1.5e-04) | 1.02 s / 936 ms (8.3e-04) | 1.46 s / — (1.1e-03) |
| K_eagle_mz_S5_t2pi8 | 0.0001 | 557 ms / 808 ms (2.2e-02) | 9.23 ms / 23.1 ms (1.0e-02) | 1.4 s / 2.19 s (3.7e-02) | 2.31 s / — (2.9e-02) |
| K_eagle_mz_S5_t2pi8 | 1e-05 | timeout / 4.24 s (6.3e-04) | timeout / fell back | timeout / timeout | timeout / — |
| K_eagle_mz_S5_t3pi8 | 0.0001 | 478 ms / 1.13 s (4.5e-03) | 11.9 ms / 15.9 ms (1.1e-03) | 1.47 s / 1.97 s (1.6e-02) | 5.97 s / — (1.8e-02) |
| K_eagle_mz_S5_t3pi8 | 1e-05 | timeout / 2.75 s (1.0e-03) | timeout / fell back | timeout / timeout | timeout / — |
| K_eagle_z62_S20_t1pi8 | 0.0001 | 2.68 s / 3.74 s (no ref) | fell back | timeout / timeout | 7.81 s / — (no ref) |
| K_eagle_z62_S20_t1pi8 | 1e-05 | over cap / over cap | over cap / over cap | timeout / timeout | timeout / — |
