"""
Is step-0 latency generally elevated across ALL programs (a burst-queueing
artifact from launching 64 programs almost simultaneously), or specific to
just a couple? Settles whether the ~300s anomaly is a general startup effect
or something narrower worth chasing further.

Usage:
    python3 analysis/step0_check.py results/grpo101.json
    python3 analysis/step0_check.py results/grpo102.json
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import statistics as st


def main(client_json):
    data = json.loads(Path(client_json).read_text())
    per = defaultdict(list)
    for e in data["events"]:
        per[e["program_id"]].append(e)

    step0, later = [], []
    for pid, evs in per.items():
        evs.sort(key=lambda e: e["step"])
        for e in evs:
            (step0 if e["step"] == 0 else later).append(e["latency_s"])

    print(f"  n programs: {len(per)}")
    print(f"  step-0 latency:   mean={st.mean(step0):6.1f}s  "
          f"median={st.median(step0):6.1f}s  max={max(step0):6.1f}s")
    print(f"  step-1+ latency:  mean={st.mean(later):6.1f}s  "
          f"median={st.median(later):6.1f}s  max={max(later):6.1f}s")

    sorted0 = sorted(step0, reverse=True)
    print(f"  top 5 step-0 latencies: {[round(x,1) for x in sorted0[:5]]}")
    n_elevated = sum(1 for x in step0 if x > 20)
    print(f"  step-0 values > 20s: {n_elevated}/{len(step0)}")


if __name__ == "__main__":
    main(sys.argv[1])
