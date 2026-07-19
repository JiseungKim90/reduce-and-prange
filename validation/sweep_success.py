#!/usr/bin/env python3
"""Sweep fixed-budget RP success probabilities over diverse small parameters.

This script is a companion to rp_success_mc.py. It runs a curated set of
small/medium cases where honest coordinate-level Monte Carlo is still feasible,
and prints a compact table comparing:

  - G1: recursive bounded-pass success probability;
  - MC: honest coordinate-level Monte Carlo estimate;
  - single: the heuristic 1-1/e;
  - product: the independent product heuristic (1-1/e)^L;
  - d_single: log2((1-1/e)/G1), the optimism in bits if single is used;
  - d_product: log2(G1/(1-1/e)^L), the pessimism in bits of product.

All stage budgets use r_j = ceil(1/P_j).
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from math import comb
from typing import Sequence


@dataclass(frozen=True)
class Case:
    name: str
    m: int
    n: int
    t: int
    ns: tuple[int, ...]
    trials: int = 60_000


CASES = [
    Case("L2-balanced", 24, 12, 4, (6, 6)),
    Case("L3-balanced", 24, 12, 4, (4, 4, 4)),
    Case("L4-balanced", 24, 12, 4, (3, 3, 3, 3), 50_000),
    Case("L6-balanced", 24, 12, 4, (2, 2, 2, 2, 2, 2), 35_000),
    Case("medium-balanced", 40, 20, 6, (8, 6, 6)),
    Case("front-heavy", 40, 20, 6, (12, 4, 4)),
    Case("back-heavy", 40, 20, 6, (4, 4, 12)),
    Case("many-small", 50, 25, 5, (5, 5, 5, 5, 5), 35_000),
    Case("low-noise", 60, 30, 3, (10, 10, 10)),
    Case("higher-noise", 60, 30, 12, (10, 10, 10)),
    Case("singleton-tail", 64, 32, 4, (16, 8, 4, 1, 1, 1, 1), 25_000),
    Case("deep-small", 72, 36, 6, (12, 8, 4, 3, 3, 2, 2, 2), 20_000),
]


def stage_probs(m: int, t: int, ns: Sequence[int]) -> list[float]:
    probs = []
    mu = 0
    for n_i in ns:
        remaining = m - mu
        numerator = comb(remaining - t, n_i) if remaining - t >= n_i else 0
        denominator = comb(remaining, n_i)
        if numerator == 0:
            raise ValueError(f"zero stage probability for N_i={n_i} after mu={mu}")
        probs.append(numerator / denominator)
        mu += n_i
    return probs


def recursive_success(probs: Sequence[float], reps: Sequence[int]) -> float:
    g = 1.0 - (1.0 - probs[-1]) ** reps[-1]
    for p, r in zip(reversed(probs[:-1]), reversed(reps[:-1])):
        g = 1.0 - (1.0 - p * g) ** r
    return g


def continuous_success(probs: Sequence[float]) -> float:
    """Same recursion with idealized real budgets r_j=1/P_j."""
    g = 1.0 - (1.0 - probs[-1]) ** (1.0 / probs[-1])
    for p in reversed(probs[:-1]):
        g = 1.0 - (1.0 - p * g) ** (1.0 / p)
    return g


def plain_success(m: int, n: int, t: int) -> float:
    p = comb(m - t, n) / comb(m, n)
    return 1.0 - (1.0 - p) ** math.ceil(1.0 / p)


def honest_mc(
    m: int,
    t: int,
    ns: Sequence[int],
    reps: Sequence[int],
    trials: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    zeros = set(range(m - t))
    full = set(range(m))
    levels = len(ns)

    def attempt(level: int, available: set[int]) -> bool:
        n_i = ns[level]
        available_tuple = tuple(available)
        for _ in range(reps[level]):
            chosen = set(rng.sample(available_tuple, n_i))
            if not chosen <= zeros:
                continue
            if level == levels - 1:
                return True
            if attempt(level + 1, available - chosen):
                return True
        return False

    successes = sum(1 for _ in range(trials) if attempt(0, full))
    p_hat = successes / trials
    ci95 = 1.96 * math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / trials)
    return p_hat, ci95


def run_case(case: Case, seed: int) -> dict[str, object]:
    if sum(case.ns) != case.n:
        raise ValueError(f"{case.name}: sum(Ns)={sum(case.ns)} != n={case.n}")
    probs = stage_probs(case.m, case.t, case.ns)
    reps = [math.ceil(1.0 / p) for p in probs]
    g1 = recursive_success(probs, reps)
    g_cont = continuous_success(probs)
    mc, ci95 = honest_mc(case.m, case.t, case.ns, reps, case.trials, seed)
    single = 1.0 - 1.0 / math.e
    product = single ** len(case.ns)
    return {
        "name": case.name,
        "m": case.m,
        "n": case.n,
        "t": case.t,
        "L": len(case.ns),
        "Ns": ",".join(str(x) for x in case.ns),
        "minP": min(probs),
        "maxP": max(probs),
        "maxr": max(reps),
        "G1": g1,
        "Gcont": g_cont,
        "MC": mc,
        "CI": ci95,
        "plain": plain_success(case.m, case.n, case.t),
        "single": single,
        "product": product,
        "d_single": math.log2(single / g1),
        "d_product": math.log2(g1 / product),
    }


def print_table(rows: Sequence[dict[str, object]]) -> None:
    print(
        "case               m   n   t  L  Ns                 "
        "P[min,max]       maxr   MC +/-CI       G1      Gcont   plain   "
        "d_single  d_product"
    )
    print("-" * 142)
    for row in rows:
        print(
            f"{row['name']:<16} "
            f"{row['m']:>3} {row['n']:>3} {row['t']:>3} {row['L']:>2} "
            f"{row['Ns']:<18} "
            f"[{row['minP']:.3f},{row['maxP']:.3f}] "
            f"{row['maxr']:>5} "
            f"{row['MC']:.4f} +/-{row['CI']:.4f} "
            f"{row['G1']:.4f} "
            f"{row['Gcont']:.4f} "
            f"{row['plain']:.4f} "
            f"{row['d_single']:+.3f} "
            f"{row['d_product']:+.3f}"
        )


def print_summary(rows: Sequence[dict[str, object]]) -> None:
    max_mc_diff = max(abs(float(r["MC"]) - float(r["G1"])) for r in rows)
    max_single = max(float(r["d_single"]) for r in rows)
    min_single = min(float(r["d_single"]) for r in rows)
    max_product = max(float(r["d_product"]) for r in rows)
    min_product = min(float(r["d_product"]) for r in rows)
    print()
    print("Summary")
    print(f"  max |MC-G1|              : {max_mc_diff:.5f}")
    print(f"  d_single range (bits)    : [{min_single:+.3f}, {max_single:+.3f}]")
    print(f"  d_product range (bits)   : [{min_product:+.3f}, {max_product:+.3f}]")
    print("  Interpretation           : single 1-1/e is usually optimistic for nested RP;")
    print("                             product (1-1/e)^L is far too pessimistic.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260627)
    args = parser.parse_args()
    rows = [run_case(case, args.seed + i) for i, case in enumerate(CASES)]
    print_table(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
