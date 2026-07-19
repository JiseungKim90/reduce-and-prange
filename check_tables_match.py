#!/usr/bin/env python3
"""Check that the parameter tables in the two estimators agree.

The Python and C++ estimators each carry their own copy of the parameter table
(problem parameters, and the baseline values quoted from Liu et al.). Nothing
enforces that the two copies stay in step, so editing one and forgetting the
other would silently make the implementations disagree. This script parses both
tables and compares them row by row.

Usage:
  python3 check_tables_match.py          # exits 0 if the tables agree
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PY = Path(__file__).parent / "estimator" / "rp_estimator_parallel.py"
CPP = Path(__file__).parent / "estimator" / "rp_estimator_parallel.cpp"

# Row("SD-low", "sd", 2**20, 786432, 31, 112, 110, quick=True),
PY_ROW = re.compile(
    r'Row\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*2\*\*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,'
    r'\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*quick\s*=\s*(True|False)\s*)?\)'
)

# {"SD-low", "sd", 1LL << 20, 786432, 31, 112, 110, true},
CPP_ROW = re.compile(
    r'\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*1LL\s*<<\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,'
    r'\s*(\d+)\s*,\s*(\d+)\s*,\s*(true|false)\s*\}'
)


def parse(path: Path, pattern: re.Pattern, truth: str) -> list[tuple]:
    rows = []
    for m in pattern.finditer(path.read_text(encoding="utf-8")):
        table, kind, log2m, n, t, baseline, paper_rp, quick = m.groups()
        rows.append(
            (table, kind, int(log2m), int(n), int(t), int(baseline), int(paper_rp),
             (quick or "").lower() == truth)
        )
    return rows


def label(r: tuple) -> str:
    return f"{r[0]:<8} 2^{r[2]:<2} n={r[3]:<9} t={r[4]:<5}"


def main() -> int:
    for p in (PY, CPP):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    py = parse(PY, PY_ROW, "true")
    cpp = parse(CPP, CPP_ROW, "true")

    if not py or not cpp:
        print(f"parsed no rows (python: {len(py)}, c++: {len(cpp)}) -- "
              f"the table format probably changed", file=sys.stderr)
        return 2

    if len(py) != len(cpp):
        print(f"row count differs: python {len(py)}, c++ {len(cpp)}", file=sys.stderr)
        return 1

    ok = True
    for a, b in zip(sorted(py), sorted(cpp)):
        if a != b:
            ok = False
            print(f"MISMATCH\n  python: {label(a)} baseline={a[5]} old={a[6]} quick={a[7]}"
                  f"\n  c++   : {label(b)} baseline={b[5]} old={b[6]} quick={b[7]}",
                  file=sys.stderr)

    if not ok:
        return 1

    print(f"parameter tables agree ({len(py)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
