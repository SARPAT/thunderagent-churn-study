"""
Diagnose the ~7000-token, ~360s-blocking, zero-eviction anomaly seen in
seeds 101 and 102.

Two independent checks, using data already on disk (no new runs needed):

  CHECK A: did grpo_analysis.py's regex MISS any REASONING-phase pauses?
     The router logs ACTING pauses as "paused ACTING program X (tokens=Y)"
     but REASONING pauses as "marked REASONING program X for pause
     (tokens=Y)" -- a DIFFERENT string. grpo_analysis.py's PAUSE_RE only
     matches the first pattern. If REASONING pauses happened in the
     anomalous groups, our G1-G4 numbers for those groups are wrong.

  CHECK B: full raw history of whichever trajectory caused the stall.
     Pinpoints the exact step, then greps the ENTIRE log for every line
     mentioning that program_id. If nothing shows up, ThunderAgent never
     touched it -- meaning the delay is NOT eviction-caused, and lives
     in vLLM queueing, network, or client-side timing instead.

Usage:
    python3 analysis/diagnose_anomaly.py results/scheduler_grpo101.log results/grpo101.json 1
    python3 analysis/diagnose_anomaly.py results/scheduler_grpo102.log results/grpo102.json 2
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

PAUSE_RE = re.compile(r"^(\d+\.\d+)\s+.*paused (?:ACTING|REASONING) program (\S+) \(tokens=(\d+)\)")
MARK_RE  = re.compile(r"^(\d+\.\d+)\s+.*marked (?:ACTING|REASONING) program (\S+) for pause \(tokens=(\d+)\)")


def check_missed_marks(log_path):
    n_pause = n_mark = 0
    mark_pids = set()
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        if PAUSE_RE.match(line):
            n_pause += 1
        m = MARK_RE.match(line)
        if m:
            n_mark += 1
            mark_pids.add(m.group(2))
    print(f"  'paused ... program' lines (what grpo_analysis.py counted): {n_pause}")
    print(f"  'marked ... for pause' lines (MISSED by grpo_analysis.py) : {n_mark}")
    if n_mark:
        print(f"  Programs affected by the missed pattern: {sorted(mark_pids)}")
    return n_mark, mark_pids


def find_worst_trajectory(client_json, group_id=None):
    data = json.loads(Path(client_json).read_text())
    per = defaultdict(list)
    for e in data["events"]:
        per[e["program_id"]].append(e)
    worst_pid, worst_span, worst_evs = None, -1, None
    for pid, evs in per.items():
        if group_id is not None and evs[0]["group"] != group_id:
            continue
        evs.sort(key=lambda e: e["wallclock"])
        span = evs[-1]["wallclock"] + evs[-1]["latency_s"] - evs[0]["wallclock"]
        if span > worst_span:
            worst_span, worst_pid, worst_evs = span, pid, evs
    return worst_pid, worst_span, worst_evs


def full_history(log_path, pid):
    print(f"\n  Full raw log history for {pid}:")
    found = False
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        if pid in line:
            found = True
            print(f"    {line}")
    if not found:
        print(f"    (NOTHING -- ThunderAgent's log never mentions {pid} at all.")
        print(f"     The delay is NOT scheduler-caused -- look at vLLM queueing")
        print(f"     or network/client-side latency instead.)")


def main(log, client, group_id):
    print("=" * 70)
    print("  CHECK A: did we miss REASONING-phase pause-marks?")
    print("=" * 70)
    check_missed_marks(log)

    print("\n" + "=" * 70)
    print(f"  CHECK B: worst trajectory in group {group_id}")
    print("=" * 70)
    pid, span, evs = find_worst_trajectory(client, group_id)
    print(f"\n  Worst trajectory: {pid}  (total span {span:.1f}s)")
    print(f"  Per-step latencies:")
    for e in evs:
        print(f"    step {e['step']}  latency={e['latency_s']:.1f}s  ok={e['ok']}")
    full_history(log, pid)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
