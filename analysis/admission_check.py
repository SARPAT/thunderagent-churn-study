"""
Does step-0 latency correlate with program size? If ThunderAgent admits NEW
programs smallest-first (same bias as eviction), small programs get in fast
and large programs wait longest for their very first request -- a POSITIVE
correlation (bigger = more startup wait).

Usage:
    python3 analysis/admission_check.py results/grpo101.json
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np


def main(client):
    data = json.loads(Path(client).read_text())
    per = defaultdict(list)
    for e in data["events"]:
        per[e["program_id"]].append(e)

    sizes, lat0 = [], []
    for pid, evs in per.items():
        evs.sort(key=lambda e: e["step"])
        sizes.append(evs[0]["start_tokens"])
        lat0.append(evs[0]["latency_s"])

    sizes, lat0 = np.array(sizes, float), np.array(lat0, float)
    c = np.corrcoef(sizes, lat0)[0, 1]
    print(f"  n={len(sizes)}")
    print(f"  corr(start_tokens, step0_latency) = {c:+.3f}")
    print(f"  (positive => bigger programs wait LONGER to be admitted --")
    print(f"   same smallest-first bias as eviction, at a different point)")

    order = np.argsort(sizes)
    print(f"\n  {'size':>8}{'step0_latency':>15}")
    for i in order[::max(len(order)//15, 1)]:
        print(f"  {sizes[i]:>8.0f}{lat0[i]:>15.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
