# Reduce and Prange — estimator and success-probability validation

Artifact for *"Reduce and Prange: Revisiting Prange's ISD for Solving SD/RSD over
Large Fields"* (Jiseung Kim, Changmin Lee).

This repository contains the code that produces the numerical tables in the paper
and the Monte Carlo experiments that validate the success-probability model.

## Cost model

The Reduce and Prange (RP) algorithm partitions the information set into blocks
`N_1, ..., N_{k+1}` with `sum_j N_j = n`. Writing `P(j)` for the conditional
probability that block `j` is noise-free and `Q(j) = prod_{i<=j} P(i)`, the
bounded-pass body cost is

```
B(N) = sum_{j=1..k} C(j)/Q(j)  +  S/Q(k)
```

The pass-success factor is **not** a single `1 - 1/e`, and it is **not** the
independent product `(1 - 1/e)^L`. Because the inner searches reuse a fixed clean
prefix, the stage success probabilities do not multiply as independent factors.
The correct factor is the recursion

```
G_{L}   = 1 - (1 - P(L))^{r_L}
G_{j}   = 1 - (1 - P(j) * G_{j+1})^{r_j}        for j = L-1, ..., 1
gamma   = G_1
```

and the reported cost is `log2( B(N) / gamma )`. The estimator uses the
*continuous* fixed-budget convention `r_j = 1/P(j)`; the validation scripts use
literal integer budgets `r_j = ceil(1/P(j))`.

Linear-algebra exponent: `omega = 2.8` throughout. All costs count **arithmetic
operations over `F_q`**, not bit operations.

## Layout

```
check_tables_match.py         checks that the two parameter tables agree
estimator/
  rp_estimator_parallel.py    threshold-partition search + G_1 recursion (reference)
  rp_estimator_parallel.cpp   same algorithm in C++ (used for the reported runs)
validation/
  rp_success_mc.py            coordinate-level Monte Carlo for one bounded pass
  sweep_success.py            curated sweep: MC vs G_1 vs 1-1/e vs (1-1/e)^L
  grid_success.py             recursion-only grid over rate/noise/levels/shapes
  rp_full_e2e.cpp             end-to-end RP over F_p: build instance, search,
                              solve, verify full residual weight
logs/
  cpp_estimator_*.txt         C++ estimator output for the reported runs
  e2e_validation.txt          end-to-end validation output
```

The reported table values come from the C++ estimator, so only its logs are stored here. The
Python implementation is a reference for the same algorithm and is meant to be run rather than
archived.

Baseline values in the `old` and `newImp` columns are taken from the corresponding tables of
Liu, Wang, Yang and Yu (ePrint 2022/712): the exact-noise tables for the SD rows and the
regular-noise tables for the RSD rows.

## Build and run

Estimator (C++, used for the reported numbers):

```sh
g++ -O3 -std=c++17 -pthread estimator/rp_estimator_parallel.cpp -o rp_estimator_parallel
./rp_estimator_parallel                      # all 47 rows
./rp_estimator_parallel --table SD-low       # one table
./rp_estimator_parallel --max-n 1000000      # skip the largest rows
./rp_estimator_parallel --skip-quick
```

Estimator (Python reference implementation):

```sh
python3 estimator/rp_estimator_parallel.py --workers 12 --out results
# writes results/estimator_results.csv and results/estimator_summary.md
```

Validation:

```sh
python3 validation/rp_success_mc.py
python3 validation/rp_success_mc.py --m 40 --n 20 --t 6 --Ns 8,6,6 --trials 200000
python3 validation/sweep_success.py
python3 validation/grid_success.py

g++ -O3 -std=c++17 -pthread validation/rp_full_e2e.cpp -o rp_full_e2e
./rp_full_e2e --threads 12
```

Requirements: a C++17 compiler, and Python 3.8+ (standard library only — no
third-party packages).

Each estimator carries its own copy of the parameter table: the problem parameters, and the
baseline values quoted from Liu et al. Nothing in the build forces the two copies to stay in
step, so a check is provided:

```sh
python3 check_tables_match.py     # exits nonzero and names the row if they have drifted
```

## Output columns

`rp_estimator_parallel` prints one row per parameter set:

| column | meaning |
|---|---|
| `old` | the RP value reported in the originally submitted tables |
| `new` | `log2(B/G_1)`, the corrected estimate |
| `ceil` | `new` rounded up |
| `G1` | the recursive pass-success factor `gamma` |
| `+bit` | `log2((1-1/e)/G_1)` — bits by which a single `1-1/e` is optimistic |
| `newImp` | `baseline - new`; negative means RP does not beat the baseline |

## What the validation runs establish

1. **The two implementations agree.** They follow the same specification and produce the same
   values, which we have checked against the stored C++ logs. Running the Python reference
   writes `results/estimator_results.csv`, so the check can be repeated. (Row ordering in the
   C++ output follows asynchronous completion and is not significant.)

2. **The recursion matches an honest coordinate-level Monte Carlo.**
   `validation/rp_success_mc.py` and `sweep_success.py` sample coordinate sets
   without replacement and run the nested bounded search directly. Dirty prefixes
   consume their branch budget but can never succeed, matching the bounded-pass
   success event.

3. **The recursion matches a full end-to-end attack.** `validation/rp_full_e2e.cpp`
   generates a random instance over `F_p` with `p = 2147483647`, runs the nested
   fixed-budget search, solves the final linear system, and verifies the **full**
   residual weight `|b - As| = t`. See `logs/e2e_validation.txt`: across all cases
   the end-to-end success rate agrees with `G_1` to within the Monte Carlo
   confidence interval, and the number of **verified dirty accepts is zero**
   (column `false`), consistent with the correctness lemma in the paper.

4. **Both common heuristics are wrong, in opposite directions.** `sweep_success.py`
   reports `d_single = log2((1-1/e)/G_1)` and `d_product = log2(G_1/(1-1/e)^L)`.
   A single `1-1/e` is optimistic; the independent product `(1-1/e)^L` is far too
   pessimistic.

5. **Integer vs continuous budgets.** `grid_success.py` scans a broader grid and
   reports both `r_j = ceil(1/P(j))` and `r_j = 1/P(j)`. This is a
   model-sensitivity check; it is not used to alter the reported table values.

## Environment for the reported runs

| | |
|---|---|
| OS | Ubuntu 20.04.6 LTS |
| CPU | Intel Core i9-10940X @ 3.30 GHz, 28 threads |
| Compiler | g++ 13.1.0, `-O3 -std=c++17 -pthread` |
| Python | 3.8.10 |

Monte Carlo scripts take a `--seed` argument and are deterministic given the seed
(defaults are fixed in the sources), so all logs in `logs/` are reproducible.
