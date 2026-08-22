"""
Exact analysis of ThunderAgent eviction decisions.

Unlike the earlier reconstruction (which ESTIMATED which programs were
eligible and how big they were, and failed), this reads the candidate list
the scheduler itself logged at each decision point:

    CHURN_CANDIDATES acting n=5 list=prog-003:2100,prog-011:4300,...
    Scheduler paused ACTING program prog-003 (tokens=2100)

So we know exactly who was eligible, exactly how big each was, and exactly
who was chosen. No estimation.

Three outputs:
  A. Verification  - is the victim always the smallest candidate?
  B. Concentration - how unevenly are evictions spread, given eligibility?
  C. Counterfactual - what WOULD a fatigue-aware policy have chosen instead,
                      replayed offline over the same decision points?

(C) predicts the effect of the patch before we build it, using real
decisions rather than a simulation.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TS = r"^(\d+\.\d+)\s+"
CAND_RE = re.compile(TS + r".*CHURN_CANDIDATES (\w+) n=(\d+) list=(\S*)")
PAUSE_RE = re.compile(TS + r".*paused (?:ACTING|REASONING) program (\S+) \(tokens=(\d+)\)")


def parse(log_path):
    """Pair each candidate list with the pause that immediately follows."""
    decisions, pending = [], None
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        m = CAND_RE.match(line)
        if m:
            t, phase, lst = float(m.group(1)), m.group(2), m.group(4)
            cands = []
            for item in lst.split(","):
                if ":" in item:
                    pid, tok = item.rsplit(":", 1)
                    try:
                        cands.append((pid, int(tok)))
                    except ValueError:
                        pass
            pending = {"t": t, "phase": phase, "candidates": cands}
            continue
        m = PAUSE_RE.match(line)
        if m and pending is not None:
            pending["victim"] = m.group(2)
            pending["victim_tokens"] = int(m.group(3))
            decisions.append(pending)
            pending = None
    return decisions


def load_starts(client_json):
    data = json.loads(Path(client_json).read_text())
    starts = {}
    for e in data["events"]:
        starts.setdefault(e["program_id"], e["start_tokens"])
    return starts, data.get("elapsed_s", 0)


def gini(c):
    a = np.sort(np.asarray(c, float))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="results/scheduler_run9.log")
    ap.add_argument("client", nargs="?", default="results/run9.json")
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()

    decisions = parse(args.log)
    starts, elapsed = load_starts(args.client)

    print(f"Decisions with a logged candidate list: {len(decisions)}")
    if not decisions:
        print("None found -- is the instrumented router running?")
        return

    sizes = [len(d["candidates"]) for d in decisions]
    print(f"Candidates per decision: mean={np.mean(sizes):.1f} "
          f"min={min(sizes)} max={max(sizes)}")

    correct = sum(1 for d in decisions
                  if d["candidates"] and d["victim"] == d["candidates"][0][0])
    multi = [d for d in decisions if len(d["candidates"]) > 1]
    correct_multi = sum(1 for d in multi if d["victim"] == d["candidates"][0][0])
    print(f"\nA. Policy verification")
    print(f"   Victim == smallest candidate: {correct}/{len(decisions)} "
          f"({correct/len(decisions)*100:.0f}%)")
    if multi:
        print(f"   Restricted to real choices (>1 candidate): "
              f"{correct_multi}/{len(multi)} ({correct_multi/len(multi)*100:.0f}%)")
    else:
        print(f"   NOTE: no decision had more than one candidate -- the policy")
        print(f"   was never actually exercised. Nothing to conclude.")

    evicted, eligible = defaultdict(int), defaultdict(int)
    for d in decisions:
        evicted[d["victim"]] += 1
        for pid, _ in d["candidates"]:
            eligible[pid] += 1

    pids = sorted(eligible)
    print(f"\nB. Concentration (conditional on being eligible)")
    print(f"   {'program':>10} {'start':>7} {'eligible':>9} {'evicted':>8} {'rate':>6}")
    for pid in sorted(pids, key=lambda p: -evicted[p])[:12]:
        el, ev_ = eligible[pid], evicted[pid]
        print(f"   {pid:>10} {starts.get(pid,0):>7} {el:>9} {ev_:>8} "
              f"{(ev_/el if el else 0):>6.2f}")

    st = np.array([starts.get(p, 0) for p in pids], float)
    ev = np.array([evicted[p] for p in pids], float)
    el = np.array([eligible[p] for p in pids], float)
    if len(st) > 2 and ev.std() > 0:
        print(f"\n   corr(start_tokens, times_evicted)  = {np.corrcoef(st, ev)[0,1]:+.3f}")
        cond = ev / np.maximum(el, 1)
        if cond.std() > 0:
            print(f"   corr(start_tokens, evict|eligible) = {np.corrcoef(st, cond)[0,1]:+.3f}")
            print(f"   (the second controls for eligibility -- if it stays")
            print(f"    negative, size drives selection, not exposure)")

    print(f"\nC. Counterfactual: fatigue-aware policy (beta={args.beta})")
    print(f"   score = C_REF/tokens - beta * times_already_evicted")
    C_REF = 4000.0
    cf_evicted = defaultdict(int)
    changed = 0
    for d in decisions:
        if not d["candidates"]:
            continue
        best, best_score = None, -1e18
        for pid, tok in d["candidates"]:
            sc = C_REF / max(tok, 1) - args.beta * cf_evicted[pid]
            if sc > best_score:
                best, best_score = pid, sc
        cf_evicted[best] += 1
        if best != d["victim"]:
            changed += 1

    actual_counts = [evicted[p] for p in pids]
    cf_counts = [cf_evicted[p] for p in pids]
    print(f"   Decisions changed: {changed}/{len(decisions)} "
          f"({changed/len(decisions)*100:.0f}%)")
    print(f"   max evictions on one program: actual={max(actual_counts)} "
          f"-> counterfactual={max(cf_counts) if cf_counts else 0}")
    print(f"   Gini of eviction counts     : actual={gini(actual_counts):.3f} "
          f"-> counterfactual={gini(cf_counts):.3f}")
    print(f"   (lower Gini = burden shared more evenly)")

    Path("results/exact_decisions.json").write_text(json.dumps({
        "n_decisions": len(decisions),
        "victim_is_smallest": correct / len(decisions),
        "n_multi_candidate": len(multi),
        "actual_max": max(actual_counts) if actual_counts else 0,
        "cf_max": max(cf_counts) if cf_counts else 0,
        "actual_gini": gini(actual_counts),
        "cf_gini": gini(cf_counts),
        "changed_frac": changed / len(decisions),
        "decisions": decisions,
    }, indent=2))
    print(f"\nSaved -> results/exact_decisions.json")


if __name__ == "__main__":
    main()
