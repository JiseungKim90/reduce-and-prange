#!/usr/bin/env python3
"""Regenerate the figures under the current cost model.

The plotted quantity is the bit gain

    Delta = log2(T_Prange) - log2(T_RP),

i.e. how many bits of estimated cost RP saves over plain Prange. Both sides are
evaluated with the SAME estimator and the same fixed-budget convention:

  T_Prange : the k=0 case, i.e. a single block of size n. Its pass-success
             factor is 1-(1-P)^(1/P) with P = binom(m-t,n)/binom(m,n).
  T_RP     : the threshold-partition optimum, with the recursive factor G_1.

Both therefore use omega = 2.8 and count F_q operations, exactly as the tables do.
Note that the baseline here is plain Prange, whereas the comparison tables use the
best estimate reported by Liu et al.; the two baselines are not the same.

The CSV also records the ratio log2(T_RP)/log2(T_Prange) and the effective
exponent (log2(T) - log2(1/P))/log2(n) for reference, but neither is plotted:
the ratio is not monotone for reasons that come from its denominator, and the
effective exponent falls below 2, so it cannot be read as a linear-algebra
exponent.

Requires matplotlib. Writes the PNGs and a CSV of the raw sweep next to them.

  python3 validation/make_figures.py --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "estimator"))
import rp_estimator_parallel as E  # noqa: E402


def prange_log_cost(m: int, n: int, t: int) -> float:
    """Prange as the k=0 case of the same model: one block of size n."""
    cq = lambda x: E.sd_logq(m, t, x)
    sp = lambda mu, x: E.sd_stage_logp(m, t, mu, x)
    corrected, _paper, _g, _d = E.evaluate_partition(m, n, t, [n], cq, sp)
    return corrected


def ratio(args: tuple[int, int, int]):
    m, n, t = args
    r = E.eval_row(E.Row("F", "sd", m, n, t, 0, 0))
    p = prange_log_cost(m, n, t)
    log_inv_P = -E.sd_logq(m, t, n) / math.log(2)
    return (m, n, t, p, r.log_cost,
            p - r.log_cost,                              # bit gain (plotted)
            r.log_cost / p,                              # ratio (recorded only)
            (r.log_cost - log_inv_P) / math.log2(n),     # eff. exponent, RP
            (p - log_inv_P) / math.log2(n),              # eff. exponent, Prange
            r.levels)


# (filename stem, swept symbol, base (m,n,t), values of the swept symbol)
SWEEPS = [
    ("plot(2^10,652,106,m+)", "m", (2**10, 652, 106), list(range(1000, 3001, 100))),
    ("plot(2^12,1589,172,m+)", "m", (2**12, 1589, 172), list(range(4200, 7201, 150))),
    ("plot(2^10,652,106,n+)", "n", (2**10, 652, 106), list(range(650, 751, 10))),
    ("plot(2^12,1589,172,n+)", "n", (2**12, 1589, 172), list(range(1590, 1831, 20))),
    ("plot(2^10,652,106,t+)", "t", (2**10, 652, 106), list(range(115, 206, 5))),
    ("plot(2^12,1589,172,t+)", "t", (2**12, 1589, 172), list(range(180, 341, 10))),
]


def points(symbol: str, base: tuple[int, int, int], values: list[int]):
    m, n, t = base
    if symbol == "m":
        return [(v, n, t) for v in values]
    if symbol == "n":
        return [(m, v, t) for v in values]
    return [(m, n, v) for v in values]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = []
    for stem, symbol, base, values in SWEEPS:
        pts = points(symbol, base, values)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            res = list(pool.map(ratio, pts))
        res.sort(key=lambda r: {"m": r[0], "n": r[1], "t": r[2]}[symbol])
        xs = [{"m": r[0], "n": r[1], "t": r[2]}[symbol] for r in res]
        cs = [r[5] for r in res]   # bit gain

        fig, ax = plt.subplots(figsize=(9.0, 3.6))
        ax.plot(xs, cs, marker="o", markersize=3.5, linewidth=1.4)
        ax.set_ylabel("bits saved")
        ax.grid(alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(args.out / f"{stem}.png", dpi=200)
        plt.close(fig)

        hi = max(range(len(cs)), key=lambda i: cs[i])
        print(f"{stem:28s} {symbol}={xs[0]}..{xs[-1]}  "
              f"gain={min(cs):.2f}..{max(cs):.2f} bits  max at {symbol}={xs[hi]}")
        for r in res:
            all_rows.append((stem, symbol) + r)

    csv_path = args.out / "figure_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["figure", "swept", "m", "n", "t",
                    "log2_T_Prange", "log2_T_RP", "bit_gain",
                    "ratio_C", "eff_exponent_RP", "eff_exponent_Prange", "levels"])
        w.writerows(all_rows)
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
