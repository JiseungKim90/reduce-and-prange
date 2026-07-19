#!/usr/bin/env python3
"""
Parallel RP/regular-RP estimator for the revised success-probability accounting.

Model used here:
  1. Generate the RP partition with the paper's threshold family.
  2. Evaluate the fixed-pass body cost
        B = sum_j body_j / Q_j
     in log-space.
  3. Replace the single 1-1/e normalization by the pass-success recursion
        G_L = 1-(1-p_L)^(1/p_L),
        G_j = 1-(1-p_j G_{j+1})^(1/p_j),
     using the continuous repetition count r_j=1/p_j.

The final reported cost is log2(B/G_1).  This is the "continuous fixed-budget"
reading of the formulas in the manuscript; it is also the convention for which
the cost and success budgets are dimensionally consistent.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ONE_MINUS_INV_E = 1.0 - 1.0 / math.e
OMEGA = 2.8


def logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def lbinom(n: int, k: int) -> float:
    if k < 0 or k > n or n < 0:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def log_body(m_eff: int, n_stage: int, leaf: bool) -> float:
    if n_stage <= 0:
        return -math.inf
    if leaf:
        # Last Prange/verifier body in the paper's tables.
        val = n_stage ** OMEGA + m_eff * n_stage
    else:
        val = n_stage ** OMEGA + n_stage * n_stage * (m_eff - n_stage) + n_stage * (m_eff - n_stage) ** 2
    return math.log(val)


def sd_logq(m: int, t: int, n: int) -> float:
    return lbinom(m - t, n) - lbinom(m, n)


def sd_stage_logp(m: int, t: int, mu_prev: int, n_stage: int) -> float:
    return lbinom(m - mu_prev - t, n_stage) - lbinom(m - mu_prev, n_stage)


def reg_logq(m_adj: int, b: int, k_blocks: int, n: int) -> float:
    if n < 0:
        return -math.inf
    full = n // b
    rem = n - full * b
    if full > k_blocks:
        return -math.inf
    if full == k_blocks and rem:
        return -math.inf
    # Product_{i=0}^{full-1} (1-1/(k-i))^b telescopes to ((k-full)/k)^b.
    total = b * (math.log(k_blocks - full) - math.log(k_blocks)) if full else 0.0
    if rem:
        x = 1.0 - 1.0 / (k_blocks - full)
        if x <= 0.0:
            return -math.inf
        total += rem * math.log(x)
    return total


def reg_stage_logp(m_adj: int, b: int, k_blocks: int, mu_prev: int, n_stage: int) -> float:
    return reg_logq(m_adj, b, k_blocks, mu_prev + n_stage) - reg_logq(m_adj, b, k_blocks, mu_prev)


@dataclass(frozen=True)
class Row:
    table: str
    kind: str
    m: int
    n: int
    t: int
    baseline: int
    paper_rp: int
    quick: bool = False


@dataclass
class EvalResult:
    row: Row
    levels: int
    best_delta: float
    log_cost: float
    paper_model_log_cost: float
    gamma: float
    delta_bits: float
    old_improvement: float
    new_improvement: float
    partition_prefix: str


def pass_success(stage_logps: list[float]) -> float:
    g = 1.0
    for lp in reversed(stage_logps):
        p = math.exp(lp)
        pg = p * g
        if pg <= 0.0:
            g = 0.0
        elif p < 1e-8:
            # (1/p)*log(1-p*g) = -g + O(p)
            g = -math.expm1(-g)
        else:
            exponent = math.log1p(-pg) / p
            g = -math.expm1(exponent)
    return max(0.0, min(1.0, g))


def build_partition(
    m_eff: int,
    n: int,
    t_eff: int,
    threshold_log: float,
    cumulative_logq: Callable[[int], float],
    stage_logp: Callable[[int, int], float],
) -> list[int]:
    parts: list[int] = []
    mu = 0
    # A hard guard is useful for degenerate thresholds but should never bind for
    # the table parameters.
    while mu < n and len(parts) < n + 1:
        remain = n - mu

        def term_log(x: int) -> float:
            return log_body(m_eff - mu, x, leaf=False) - cumulative_logq(mu + x)

        if term_log(1) >= threshold_log:
            break

        lo, hi = 1, remain
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if term_log(mid) < threshold_log:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        parts.append(best)
        mu += best
    if mu < n:
        parts.append(n - mu)
    return parts


def evaluate_partition(
    m_eff: int,
    n: int,
    t_eff: int,
    parts: list[int],
    cumulative_logq: Callable[[int], float],
    stage_logp: Callable[[int, int], float],
) -> tuple[float, float, float, float]:
    mu = 0
    log_b = -math.inf
    stage_logps: list[float] = []
    for idx, part in enumerate(parts):
        leaf = idx == len(parts) - 1
        mu_next = mu + part
        log_q = cumulative_logq(mu_next)
        log_b = logaddexp(log_b, log_body(m_eff - mu, part, leaf=leaf) - log_q)
        stage_logps.append(stage_logp(mu, part))
        mu = mu_next
    gamma = pass_success(stage_logps)
    corrected = log_b - math.log(gamma)
    paper_model = log_b - math.log(ONE_MINUS_INV_E)
    return corrected / math.log(2), paper_model / math.log(2), gamma, math.log2(ONE_MINUS_INV_E / gamma)


def geometric_deltas(n: int) -> list[float]:
    # Original code effectively searches Delta in [2, n^omega] on a log scale.
    lo = math.log(2.0)
    hi = OMEGA * math.log(max(n, 2))
    grid = 260
    vals = [math.exp(lo + (hi - lo) * i / (grid - 1)) for i in range(grid)]
    # Add several hand-picked small values where low-dimensional rows often move.
    vals.extend([1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0])
    return sorted(set(vals))


def eval_row(row: Row) -> EvalResult:
    q_cache: dict[int, float] = {}

    if row.kind == "sd":
        m_eff = row.m
        t_eff = row.t

        def cumulative_logq(x: int) -> float:
            v = q_cache.get(x)
            if v is None:
                v = sd_logq(row.m, row.t, x)
                q_cache[x] = v
            return v

        stage_logp = lambda mu, x: sd_stage_logp(row.m, row.t, mu, x)
    elif row.kind == "rsd":
        b = row.t
        k_blocks = math.ceil(row.m / b)
        m_eff = b * k_blocks
        t_eff = b

        def cumulative_logq(x: int) -> float:
            v = q_cache.get(x)
            if v is None:
                v = reg_logq(m_eff, b, k_blocks, x)
                q_cache[x] = v
            return v

        stage_logp = lambda mu, x: cumulative_logq(mu + x) - cumulative_logq(mu)
    else:
        raise ValueError(f"unknown kind {row.kind}")

    log_t = -cumulative_logq(row.n)
    best: tuple[float, float, float, float, float, list[int]] | None = None
    for delta in geometric_deltas(row.n):
        parts = build_partition(
            m_eff=m_eff,
            n=row.n,
            t_eff=t_eff,
            threshold_log=math.log(delta) + log_t,
            cumulative_logq=cumulative_logq,
            stage_logp=stage_logp,
        )
        corrected, paper_model, gamma, delta_bits = evaluate_partition(
            m_eff=m_eff,
            n=row.n,
            t_eff=t_eff,
            parts=parts,
            cumulative_logq=cumulative_logq,
            stage_logp=stage_logp,
        )
        if best is None or corrected < best[0]:
            best = (corrected, paper_model, gamma, delta_bits, delta, parts)

    assert best is not None
    corrected, paper_model, gamma, delta_bits, delta, parts = best
    prefix = ",".join(str(x) for x in parts[:10])
    if len(parts) > 10:
        prefix += ",..."
    return EvalResult(
        row=row,
        levels=len(parts),
        best_delta=delta,
        log_cost=corrected,
        paper_model_log_cost=paper_model,
        gamma=gamma,
        delta_bits=delta_bits,
        old_improvement=row.baseline - row.paper_rp,
        new_improvement=row.baseline - corrected,
        partition_prefix=prefix,
    )


def table_rows() -> list[Row]:
    rows = [
        # Table I, SD low-noise.
        Row("SD-low", "sd", 2**10, 652, 57, 111, 105),
        Row("SD-low", "sd", 2**12, 1589, 98, 100, 94),
        Row("SD-low", "sd", 2**14, 3482, 198, 101, 97),
        Row("SD-low", "sd", 2**16, 7391, 389, 103, 99),
        Row("SD-low", "sd", 2**18, 15336, 760, 105, 108),
        Row("SD-low", "sd", 2**20, 32771, 1419, 107, 104),
        Row("SD-low", "sd", 2**22, 67440, 2735, 108, 107),
        Row("SD-low", "sd", 2**12, 3072, 44, 117, 111),
        Row("SD-low", "sd", 2**14, 12288, 39, 111, 107),
        Row("SD-low", "sd", 2**16, 49152, 34, 107, 104),
        Row("SD-low", "sd", 2**18, 196608, 32, 108, 106),
        Row("SD-low", "sd", 2**20, 786432, 31, 112, 110, quick=True),
        Row("SD-low", "sd", 2**22, 3145728, 30, 116, 114, quick=True),
        Row("SD-low", "sd", 2**24, 12582912, 29, 119, 118, quick=True),
        # Table II, SD recommended parameters.
        Row("SD-rec", "sd", 2**12, 1321, 172, 128, 121),
        Row("SD-rec", "sd", 2**14, 2895, 338, 128, 122),
        Row("SD-rec", "sd", 2**16, 6005, 667, 128, 123),
        Row("SD-rec", "sd", 2**18, 12160, 1312, 128, 124),
        Row("SD-rec", "sd", 2**20, 25346, 2467, 128, 124),
        Row("SD-rec", "sd", 2**22, 50854, 4788, 128, 125),
        # Table III, RSD.
        Row("RSD-hi", "rsd", 2**10, 652, 106, 178, 161),
        Row("RSD-hi", "rsd", 2**12, 1589, 172, 150, 141),
        Row("RSD-hi", "rsd", 2**14, 3482, 338, 149, 144),
        Row("RSD-hi", "rsd", 2**16, 7391, 667, 150, 142),
        Row("RSD-hi", "rsd", 2**18, 15336, 1312, 133, 145),
        Row("RSD-hi", "rsd", 2**20, 32771, 2467, 131, 148),
        Row("RSD-hi", "rsd", 2**22, 67440, 4788, 110, 150),
        Row("RSD-mid", "rsd", 2**10, 652, 57, 107, 105),
        Row("RSD-mid", "rsd", 2**12, 1589, 98, 99, 91),
        Row("RSD-mid", "rsd", 2**14, 3482, 198, 101, 94),
        Row("RSD-mid", "rsd", 2**16, 7391, 389, 103, 97),
        Row("RSD-mid", "rsd", 2**18, 15336, 760, 105, 100),
        Row("RSD-mid", "rsd", 2**20, 32771, 1419, 102, 102),
        Row("RSD-mid", "rsd", 2**22, 67440, 2735, 104, 105),
        Row("RSD-34", "rsd", 2**12, 3072, 44, 116, 107),
        Row("RSD-34", "rsd", 2**14, 12288, 39, 111, 105),
        Row("RSD-34", "rsd", 2**16, 49152, 34, 107, 101),
        Row("RSD-34", "rsd", 2**18, 196608, 32, 108, 104),
        Row("RSD-34", "rsd", 2**20, 786432, 31, 112, 110),
        Row("RSD-34", "rsd", 2**22, 3145728, 30, 116, 120, quick=True),
        Row("RSD-34", "rsd", 2**24, 12582912, 29, 119, 124, quick=True),
        # Table IV, RSD recommended parameters.
        Row("RSD-rec", "rsd", 2**12, 1377, 172, 128, 120),
        Row("RSD-rec", "rsd", 2**14, 2909, 338, 128, 118),
        Row("RSD-rec", "rsd", 2**16, 6091, 667, 128, 118),
        Row("RSD-rec", "rsd", 2**18, 14796, 1312, 128, 128),
        Row("RSD-rec", "rsd", 2**20, 30978, 2467, 128, 142),
        Row("RSD-rec", "rsd", 2**22, 75396, 4788, 128, 165),
    ]
    return rows


def write_outputs(results: list[EvalResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "estimator_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "table",
                "kind",
                "m",
                "log2m",
                "n",
                "t",
                "baseline",
                "paper_rp",
                "new_log2_cost",
                "rounded_up",
                "levels",
                "gamma",
                "delta_bits_vs_single",
                "paper_model_log2",
                "old_improvement",
                "new_improvement",
                "best_delta",
                "partition_prefix",
            ]
        )
        for r in results:
            row = r.row
            w.writerow(
                [
                    row.table,
                    row.kind,
                    row.m,
                    int(math.log2(row.m)),
                    row.n,
                    row.t,
                    row.baseline,
                    row.paper_rp,
                    f"{r.log_cost:.6f}",
                    math.ceil(r.log_cost - 1e-12),
                    r.levels,
                    f"{r.gamma:.8g}",
                    f"{r.delta_bits:.6f}",
                    f"{r.paper_model_log_cost:.6f}",
                    f"{r.old_improvement:.6f}",
                    f"{r.new_improvement:.6f}",
                    f"{r.best_delta:.8g}",
                    r.partition_prefix,
                ]
            )

    md_path = out_dir / "estimator_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# RP estimator rerun\n\n")
        f.write("Model: threshold partition, continuous fixed-budget success recursion, final cost `B/G1`.\n\n")
        f.write("| table | log2 m | n | t | old RP | new RP | ceil | gamma | +bits | old impr | new impr |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            row = r.row
            f.write(
                f"| {row.table} | {int(math.log2(row.m))} | {row.n} | {row.t} | "
                f"{row.paper_rp} | {r.log_cost:.2f} | {math.ceil(r.log_cost - 1e-12)} | "
                f"{r.gamma:.4f} | {r.delta_bits:.2f} | {r.old_improvement:.1f} | {r.new_improvement:.1f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 12)))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = table_rows()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(eval_row, rows))
    results.sort(key=lambda r: (r.row.table, int(math.log2(r.row.m)), r.row.n, r.row.t))
    write_outputs(results, args.out)

    print(
        f"{'table':>8} {'log2m':>5} {'n':>9} {'t':>6} {'old':>5} "
        f"{'new':>8} {'ceil':>5} {'G1':>8} {'+bit':>6} {'newImp':>7}"
    )
    for r in results:
        row = r.row
        print(
            f"{row.table:>8} {int(math.log2(row.m)):>5} {row.n:>9} {row.t:>6} "
            f"{row.paper_rp:>5} {r.log_cost:>8.2f} {math.ceil(r.log_cost - 1e-12):>5} "
            f"{r.gamma:>8.4f} {r.delta_bits:>6.2f} {r.new_improvement:>7.1f}"
        )
    print(f"\nwrote {args.out / 'estimator_results.csv'}")
    print(f"wrote {args.out / 'estimator_summary.md'}")


if __name__ == "__main__":
    main()
