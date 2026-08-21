"""
Parse ThunderAgent's scheduler log to answer one question:

  Does shortest-first eviction concentrate churn on the same programs?

We only read the log lines ThunderAgent already prints -- no patching of
their code. Two line types matter:

  "Scheduler paused ACTING program prog-XXX (tokens=NNNN)"
  "Resumed program prog-XXX to <backend> (status=..., tokens=NNNN)"
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAUSE_RE = re.compile(r"paused (?:ACTING|REASONING) program (\S+) \(tokens=(\d+)\)")
MARK_RE = re.compile(r"marked (?:ACTING|REASONING) program (\S+) for pause \(tokens=(\d+)\)")
RESUME_RE = re.compile(r"Resumed program (\S+) to .*tokens=(\d+)")


def parse(log_path):
    pauses = defaultdict(list)   # pid -> [tokens at each pause]
    resumes = defaultdict(list)
    order = []                   # chronological (event, pid, tokens)

    for line in Path(log_path).read_text(errors="ignore").splitlines():
        m = PAUSE_RE.search(line) or MARK_RE.search(line)
        if m:
            pid, tok = m.group(1), int(m.group(2))
            pauses[pid].append(tok)
            order.append(("pause", pid, tok))
            continue
        m = RESUME_RE.search(line)
        if m:
            pid, tok = m.group(1), int(m.group(2))
            resumes[pid].append(tok)
            order.append(("resume", pid, tok))
    return pauses, resumes, order


def churn_cycles(order):
    """Count resume->pause pairs: how often a program is evicted again
    shortly after being restored (the 'same rider at the front' pattern)."""
    last_resume = {}
    cycles = defaultdict(int)
    for ev, pid, tok in order:
        if ev == "resume":
            last_resume[pid] = tok
        elif ev == "pause" and pid in last_resume:
            cycles[pid] += 1
            del last_resume[pid]
    return cycles


def main(log_path, client_json=None):
    pauses, resumes, order = parse(log_path)

    if not pauses:
        print("No pause events found -- no memory pressure occurred.")
        return

    # Starting context size per program, from the client's own record
    starts = {}
    if client_json and Path(client_json).exists():
        data = json.loads(Path(client_json).read_text())
        for e in data["events"]:
            starts[e["program_id"]] = e["start_tokens"]

    all_pids = sorted(set(list(pauses) + list(resumes) + list(starts)))
    cycles = churn_cycles(order)

    print(f"Total pause events : {sum(len(v) for v in pauses.values())}")
    print(f"Total resume events: {sum(len(v) for v in resumes.values())}")
    print(f"Programs ever paused: {len(pauses)} / {len(all_pids)}\n")

    print(f"{'program':>10} {'start_tok':>10} {'pauses':>7} {'resumes':>8} {'re-evicts':>10}")
    rows = []
    for pid in all_pids:
        np_ = len(pauses.get(pid, []))
        nr = len(resumes.get(pid, []))
        st = starts.get(pid, 0)
        print(f"{pid:>10} {st:>10} {np_:>7} {nr:>8} {cycles.get(pid,0):>10}")
        rows.append((st, np_))

    counts = np.array([len(pauses.get(p, [])) for p in all_pids])
    print(f"\nEvictions: mean={counts.mean():.2f} max={counts.max()} "
          f"min={counts.min()} std={counts.std():.2f}")

    # THE key number: do short programs absorb the churn?
    with_start = [(s, n) for s, n in rows if s > 0]
    if len(with_start) > 2:
        s_arr = np.array([s for s, _ in with_start])
        n_arr = np.array([n for _, n in with_start])
        if n_arr.std() > 0:
            corr = np.corrcoef(s_arr, n_arr)[0, 1]
            print(f"corr(start_context, eviction_count) = {corr:+.3f}")
            print("  (negative => shortest-first concentrates churn on short programs)")
        else:
            print("All programs evicted equally -- no concentration.")


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else "results/scheduler_run2.log"
    cj = sys.argv[2] if len(sys.argv) > 2 else "results/run2.json"
    main(log, cj)
