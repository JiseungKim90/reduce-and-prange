#!/usr/bin/env python3
"""Monte Carlo check for fixed-budget RP pass success probability.

The script compares three quantities on small parameters:

  1. Honest Monte Carlo: actually sample coordinate sets without replacement.
  2. The recursive bounded-pass probability G_1.
  3. The common heuristics 1-1/e and (1-1/e)^L.

Example:
  python rp_success_mc.py
  python rp_success_mc.py --m 40 --n 20 --t 6 --Ns 8,6,6 --trials 200000
"""

from __future__ import annotations

import argparse
import math
import random
from math import comb
from typing import Iterable, Sequence


DEFAULT_CASES = [
    (24, 12, 4, [4, 4, 4]),
    (30, 15, 5, [5, 5, 5]),
    (36, 18, 5, [6, 6, 6]),
    (40, 20, 6, [8, 6, 6]),
    (48, 24, 6, [10, 8, 6]),
]


def parse_ns(value: str) -> list[int]:
    try:
        ns = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--Ns must be comma-separated integers") from exc
    if not ns or any(n <= 0 for n in ns):
        raise argparse.ArgumentTypeError("--Ns must contain positive integers")
    return ns


def stage_probs(m: int, t: int, ns: Sequence[int]) -> list[float]:
    """Conditional clean-block probabilities for the RP stages."""
    probs: list[float] = []
    mu = 0
    for n_i in ns:
        remaining = m - mu
        if n_i < 0 or n_i > remaining:
            raise ValueError(f"invalid block size N_i={n_i} after mu={mu}")
        denominator = comb(remaining, n_i)
        numerator = comb(remaining - t, n_i) if remaining - t >= n_i else 0
        if numerator == 0:
            raise ValueError(
                f"stage has zero success probability: m={m}, t={t}, mu={mu}, N_i={n_i}"
            )
        probs.append(numerator / denominator)
        mu += n_i
    return probs


def repetition_counts(probs: Sequence[float]) -> list[int]:
    """Fixed-budget counts r_j = ceil(1/P_j)."""
    return [math.ceil(1.0 / p) for p in probs]


def recursive_success(probs: Sequence[float], reps: Sequence[int]) -> float:
    """Compute G_1 from G_L = 1-(1-P_L)^r_L and G_j recursion."""
    if len(probs) != len(reps):
        raise ValueError("probs and reps must have the same length")
    g = 1.0 - (1.0 - probs[-1]) ** reps[-1]
    for p, r in zip(reversed(probs[:-1]), reversed(reps[:-1])):
        g = 1.0 - (1.0 - p * g) ** r
    return g


def plain_prange_success(m: int, n: int, t: int) -> tuple[float, int, float]:
    """Fixed-budget success probability for plain Prange on n coordinates."""
    p = comb(m - t, n) / comb(m, n)
    r = math.ceil(1.0 / p)
    return p, r, 1.0 - (1.0 - p) ** r


def honest_monte_carlo(
    m: int,
    t: int,
    ns: Sequence[int],
    reps: Sequence[int],
    trials: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Coordinate-level Monte Carlo for one bounded RP pass.

    The zero coordinates are fixed as {0, ..., m-t-1}. Each trial samples
    fresh disjoint stage sets. Dirty prefixes still consume their branch budget
    but can never succeed, matching the bounded pass success event.
    """
    zeros = set(range(m - t))
    all_coords = set(range(m))
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

    successes = 0
    for _ in range(trials):
        if attempt(0, all_coords):
            successes += 1

    estimate = successes / trials
    ci95 = 1.96 * math.sqrt(max(estimate * (1.0 - estimate), 0.0) / trials)
    return estimate, ci95


def format_float_list(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def run_case(m: int, n: int, t: int, ns: Sequence[int], trials: int, seed: int) -> None:
    if sum(ns) != n:
        raise ValueError(f"sum(Ns)={sum(ns)} must equal n={n}")

    rng = random.Random(seed)
    probs = stage_probs(m, t, ns)
    reps = repetition_counts(probs)
    g1 = recursive_success(probs, reps)
    mc, ci95 = honest_monte_carlo(m, t, ns, reps, trials, rng)

    plain_p, plain_r, plain_g = plain_prange_success(m, n, t)
    single = 1.0 - 1.0 / math.e
    product = single ** len(ns)

    print(f"m={m}, n={n}, t={t}, Ns={list(ns)}, L={len(ns)}")
    print(f"  P_j              = {format_float_list(probs)}")
    print(f"  r_j=ceil(1/P_j) = {reps}")
    print(f"  MC success       = {mc:.5f} +/- {ci95:.5f}  (95% CI, {trials} passes)")
    print(f"  G1 recursion     = {g1:.5f}")
    print(f"  single 1-1/e     = {single:.5f}")
    print(f"  product heuristic= {product:.5f}")
    print(f"  plain Prange     = {plain_g:.5f}  (P={plain_p:.6g}, r={plain_r})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, help="code length / number of samples")
    parser.add_argument("--n", type=int, help="secret dimension / total selected zeros")
    parser.add_argument("--t", type=int, help="error weight")
    parser.add_argument("--Ns", type=parse_ns, help="comma-separated RP partition, e.g. 8,6,6")
    parser.add_argument("--trials", type=int, default=150_000, help="Monte Carlo passes per case")
    parser.add_argument("--seed", type=int, default=20260627, help="random seed")
    args = parser.parse_args()

    custom = [args.m is not None, args.n is not None, args.t is not None, args.Ns is not None]
    if any(custom) and not all(custom):
        parser.error("--m, --n, --t, and --Ns must be supplied together")

    if all(custom):
        run_case(args.m, args.n, args.t, args.Ns, args.trials, args.seed)
        return

    for index, (m, n, t, ns) in enumerate(DEFAULT_CASES):
        run_case(m, n, t, ns, args.trials, args.seed + index)


if __name__ == "__main__":
    main()
