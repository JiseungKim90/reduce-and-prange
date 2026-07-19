# Reduce and Prange — estimator and validation code

Code accompanying *"Reduce and Prange: Revisiting Prange's ISD for Solving SD/RSD over Large
Fields"*. It produces the numerical tables and the figures, and it contains the Monte Carlo
checks of the success-probability model.

## Cost model

RP partitions the information set into blocks `N_1, ..., N_{k+1}` with `sum_j N_j = n`. With
`P(j)` the conditional probability that block `j` is noise-free and `Q(j) = prod_{i<=j} P(i)`,
the bounded-pass body cost is

```
B(N) = sum_{j=1..k} C(j)/Q(j)  +  S/Q(k)
```

The pass-success factor is the recursion

```
G_{L} = 1 - (1 - P(L))^{r_L}
G_{j} = 1 - (1 - P(j) * G_{j+1})^{r_j}        for j = L-1, ..., 1
gamma = G_1
```

not a single `1 - 1/e` and not the product `(1 - 1/e)^L`, because the inner searches reuse a
fixed clean prefix. The reported cost is `log2(B(N) / gamma)`.

The estimator uses `r_j = 1/P(j)` (continuous fixed budget); the validation scripts use
`r_j = ceil(1/P(j))`. Linear-algebra exponent `omega = 2.8`. All costs count arithmetic
operations over `F_q`, not bit operations.

## Layout

```
check_tables_match.py         compares the two parameter tables
estimator/
  rp_estimator_parallel.py    threshold-partition search + G_1 recursion (reference)
  rp_estimator_parallel.cpp   same algorithm in C++; source of the reported values
validation/
  make_figures.py             regenerates the figures
  rp_success_mc.py            coordinate-level Monte Carlo for one bounded pass
  sweep_success.py            sweep: MC vs G_1 vs 1-1/e vs (1-1/e)^L
  grid_success.py             recursion-only grid over rate, noise, levels, shapes
  rp_full_e2e.cpp             end-to-end RP over F_p, verifying the full residual weight
logs/                         C++ estimator output and the end-to-end run
figures/                      the figures, and figure_data.csv with every swept point
```

Only the C++ logs are stored, since the reported values come from that implementation. The
Python version is meant to be run, not archived.

Baselines in the `old` and `newImp` columns are taken from Liu, Wang, Yang and Yu
(ePrint 2022/712): the exact-noise tables for the SD rows, the regular-noise tables for the
RSD rows.

## Build and run

```sh
g++ -O3 -std=c++17 -pthread estimator/rp_estimator_parallel.cpp -o rp_estimator_parallel
./rp_estimator_parallel                      # all 47 rows
./rp_estimator_parallel --table SD-low       # one table
./rp_estimator_parallel --max-n 1000000      # skip the largest rows
./rp_estimator_parallel --skip-quick

python3 estimator/rp_estimator_parallel.py --workers 12 --out results
# writes results/estimator_results.csv and results/estimator_summary.md

python3 validation/rp_success_mc.py
python3 validation/rp_success_mc.py --m 40 --n 20 --t 6 --Ns 8,6,6 --trials 200000
python3 validation/sweep_success.py
python3 validation/grid_success.py
python3 validation/make_figures.py --out figures

g++ -O3 -std=c++17 -pthread validation/rp_full_e2e.cpp -o rp_full_e2e
./rp_full_e2e --threads 12
```

Requirements: a C++17 compiler, Python 3.8+ (standard library only), and matplotlib for
`make_figures.py`.

Each estimator holds its own copy of the parameter table and nothing keeps the two in step, so
`check_tables_match.py` compares them and names any row that differs.

## Output columns

| column | meaning |
|---|---|
| `old` | the RP value in the originally submitted tables |
| `new` | `log2(B/G_1)` |
| `ceil` | `new` rounded up |
| `G1` | the recursive pass-success factor |
| `+bit` | `log2((1-1/e)/G_1)`, the bits by which a single `1-1/e` is optimistic |
| `newImp` | `baseline - new`; negative means RP does not beat the baseline |

## Validation

The two implementations produce the same values; this was checked against the stored C++ logs
and can be repeated by running the Python version.

`rp_success_mc.py` and `sweep_success.py` sample coordinate sets without replacement and run
the nested bounded search directly. Dirty prefixes consume their branch budget but never
succeed, which matches the bounded-pass success event.

`rp_full_e2e.cpp` generates an instance over `F_p` with `p = 2147483647`, runs the nested
fixed-budget search, solves the final linear system and checks the full residual weight
`|b - As| = t`. In `logs/e2e_validation.txt` the end-to-end success rate agrees with `G_1`
within the Monte Carlo confidence interval on every case, and the `false` column, counting
verified dirty accepts, is zero throughout.

`sweep_success.py` reports `d_single = log2((1-1/e)/G_1)` and `d_product = log2(G_1/(1-1/e)^L)`.
A single `1-1/e` is optimistic and the independent product is far too pessimistic.

`grid_success.py` scans a wider grid with both `r_j = ceil(1/P(j))` and `r_j = 1/P(j)`. This is
a sensitivity check and is not used to alter the reported values.

## Figures

The figures plot the bit gain `log2(T_Prange) - log2(T_RP)` against each of `m`, `n` and `t`,
with Prange evaluated as the `k=0` case of the same estimator. That baseline is plain Prange,
whereas the comparison tables use the best estimate reported by Liu et al., so the two sets of
numbers are not directly comparable.

`figures/figure_data.csv` also records `log2(T_RP)/log2(T_Prange)` and an effective exponent.
Neither is plotted: the ratio is not monotone for reasons coming from its denominator, and the
effective exponent falls below 2, so it cannot be read as a linear-algebra exponent.

## Environment for the reported runs

| | |
|---|---|
| OS | Ubuntu 20.04.6 LTS |
| CPU | Intel Core i9-10940X, 3.30 GHz, 28 threads |
| Compiler | g++ 13.1.0, `-O3 -std=c++17 -pthread` |
| Python | 3.8.10 |

The Monte Carlo scripts take `--seed` and are deterministic given it, so the stored logs are
reproducible.
