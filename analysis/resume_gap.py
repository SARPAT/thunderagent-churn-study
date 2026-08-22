"""
Where does the ~150s step-0 delay actually go? Compares ThunderAgent's own
'Resumed' timestamp (when IT decided to admit the program) against the
client's step-0 completion timestamp -- isolates admission-decision time
from downstream engine/queueing wait.

Usage:
    python3 analysis/resume_gap.py results/scheduler_grpo101.log results/grpo101.json
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict
import statistics as st

RESUME_RE = re.compile(r"^(\d+\.\d+)\s+.*Resumed program (\S+) to")


def main(log, client):
    resumed_at = {}
    for line in Path(log).read_text(errors="ignore").splitlines():
        m = RESUME_RE.match(line)
        if m:
            resumed_at[m.group(2)] = float(m.group(1))

    data = json.loads(Path(client).read_text())
    per = defaultdict(list)
    for e in data["events"]:
        per[e["program_id"]].append(e)

    gaps = []
    for pid, evs in per.items():
        evs.sort(key=lambda e: e["step"])
        step0 = evs[0]
        step0_complete = step0["wallclock"]
        if pid in resumed_at:
            r = resumed_at[pid]
            gaps.append({"pid": pid, "downstream_wait": step0_complete - r,
                        "latency": step0["latency_s"]})

    print(f"  matched {len(gaps)}/{len(per)} programs to a Resumed log line")
    if gaps:
        dw = [g["downstream_wait"] for g in gaps]
        lat = [g["latency"] for g in gaps]
        print(f"  downstream_wait (Resumed -> step0 complete): "
              f"mean={st.mean(dw):.1f}s median={st.median(dw):.1f}s")
        print(f"  step0 total latency (client-measured):        "
              f"mean={st.mean(lat):.1f}s median={st.median(lat):.1f}s")
        ratio = st.mean(dw) / st.mean(lat) if st.mean(lat) else 0
        print(f"  -> downstream_wait accounts for ~{ratio*100:.0f}% of step0 latency")
        print(f"     (near 100% => the wait is AFTER ThunderAgent admits it,")
        print(f"      i.e. vLLM engine backlog, not ThunderAgent's decision)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
