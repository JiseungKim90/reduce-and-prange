#!/usr/bin/env python3
"""Recursion-only grid for nested RP fixed-budget success probability.

The Monte Carlo sweep validates the recursion on small cases. This script uses
the recursion to scan a broader grid of rates, noise weights, levels, and
partition shapes. It reports the bit gap between the single-factor heuristic
1-1/e and the recursive success probability G_1.
"""

from __future__ import annotations

import math
from collections import defaultdict
from math import comb
from statistics import mean
from typing import Sequence


SINGLE = 1.0 - 1.0 / math.e


def one_minus_pow_one_minus(x: float, r: float) -> float:
    """Return 1-(1-x)^r stably for x in [0,1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return -math.expm1(r * math.log1p(-x))


def stage_probs(m: int, t: int, ns: Sequence[int]) -> list[float]:
    probs = []
    mu = 0
    for n_i in ns:
        remaining = m - mu
        if remaining - t < n_i:
            return []
        probs.append(comb(remaining - t, n_i) / comb(remaining, n_i))
        mu += n_i
    return probs


def recursive_success(probs: Sequence[float], mode: str) -> float:
    if mode == "ceil":
        reps = [math.ceil(1.0 / p) for p in probs]
    elif mode == "cont":
        reps = [1.0 / p for p in probs]
    else:
        raise ValueError(mode)
    g = one_minus_pow_one_minus(probs[-1], reps[-1])
    for p, r in zip(reversed(probs[:-1]), reversed(reps[:-1])):
        g = one_minus_pow_one_minus(p * g, r)
    return g


def balanced_partition(n: int, level: int) -> tuple[int, ...]:
    q, r = divmod(n, level)
    return tuple(q + 1 if i < r else q for i in range(level))


def front_heavy_partition(n: int, level: int) -> tuple[int, ...]:
    if level == 1:
        return (n,)
    first = max(1, n // 2)
    tail = balanced_partition(n - first, level - 1)
    return (first,) + tail


def back_heavy_partition(n: int, level: int) -> tuple[int, ...]:
    return tuple(reversed(front_heavy_partition(n, level)))


def singleton_tail_partition(n: int, level: int) -> tuple[int, ...] | None:
    if level < 3 or n <= level:
        return None
    tail_count = level // 2
    head_level = level - tail_count
    head_sum = n - tail_count
    if head_sum < head_level:
        return None
    return balanced_partition(head_sum, head_level) + (1,) * tail_count


def partitions(n: int, level: int) -> dict[str, tuple[int, ...]]:
    result = {
        "balanced": balanced_partition(n, level),
        "front": front_heavy_partition(n, level),
        "back": back_heavy_partition(n, level),
    }
    singleton = singleton_tail_partition(n, level)
    if singleton is not None:
        result["tail1"] = singleton
    return result


def main() -> None:
    rows = []
    for m in (48, 72, 96, 144, 192):
        for rate in (0.35, 0.50, 0.65):
            n = round(m * rate)
            for noise in (0.05, 0.10, 0.20):
                t = max(1, round(m * noise))
                if n >= m - t:
                    continue
                for level in (2, 3, 4, 6, 8, 12):
                    if level > n:
                        continue
                    for shape, ns in partitions(n, level).items():
                        probs = stage_probs(m, t, ns)
                        if not probs:
                            continue
                        g_ceil = recursive_success(probs, "ceil")
                        g_cont = recursive_success(probs, "cont")
                        rows.append(
                            {
                                "m": m,
                                "n": n,
                                "t": t,
                                "rate": rate,
                                "noise": noise,
                                "level": level,
                                "shape": shape,
                                "g_ceil": g_ceil,
                                "g_cont": g_cont,
                                "d_ceil": math.log2(SINGLE / g_ceil),
                                "d_cont": math.log2(SINGLE / g_cont),
                            }
                        )

    by_level = defaultdict(list)
    by_shape = defaultdict(list)
    for row in rows:
        by_level[row["level"]].append(row)
        by_shape[row["shape"]].append(row)

    print(f"scanned rows: {len(rows)}")
    print()
    print("By level: bit gap log2((1-1/e)/G1)")
    print("level  rows   ceil[min,avg,max]        cont[min,avg,max]")
    for level in sorted(by_level):
        vals = by_level[level]
        ceil_vals = [r["d_ceil"] for r in vals]
        cont_vals = [r["d_cont"] for r in vals]
        print(
            f"{level:>5} {len(vals):>5} "
            f"[{min(ceil_vals):+6.3f},{mean(ceil_vals):+6.3f},{max(ceil_vals):+6.3f}] "
            f"[{min(cont_vals):+6.3f},{mean(cont_vals):+6.3f},{max(cont_vals):+6.3f}]"
        )

    print()
    print("By partition shape: bit gap log2((1-1/e)/G1)")
    print("shape      rows   ceil[min,avg,max]        cont[min,avg,max]")
    for shape in sorted(by_shape):
        vals = by_shape[shape]
        ceil_vals = [r["d_ceil"] for r in vals]
        cont_vals = [r["d_cont"] for r in vals]
        print(
            f"{shape:<9} {len(vals):>5} "
            f"[{min(ceil_vals):+6.3f},{mean(ceil_vals):+6.3f},{max(ceil_vals):+6.3f}] "
            f"[{min(cont_vals):+6.3f},{mean(cont_vals):+6.3f},{max(cont_vals):+6.3f}]"
        )

    print()
    print("Largest ceil optimism cases")
    for row in sorted(rows, key=lambda r: r["d_ceil"], reverse=True)[:10]:
        print(
            f"  d={row['d_ceil']:+.3f} cont={row['d_cont']:+.3f} "
            f"m={row['m']} n={row['n']} t={row['t']} L={row['level']} {row['shape']} "
            f"Gceil={row['g_ceil']:.4f} Gcont={row['g_cont']:.4f}"
        )

    print()
    print("Most conservative ceil cases")
    for row in sorted(rows, key=lambda r: r["d_ceil"])[:10]:
        print(
            f"  d={row['d_ceil']:+.3f} cont={row['d_cont']:+.3f} "
            f"m={row['m']} n={row['n']} t={row['t']} L={row['level']} {row['shape']} "
            f"Gceil={row['g_ceil']:.4f} Gcont={row['g_cont']:.4f}"
        )


if __name__ == "__main__":
    main()
