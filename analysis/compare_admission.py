"""
Compare admission-ordering policies: smallest-first (upstream), FIFO, largest-first.

Only ONE variable differs between the three runs: the sort applied to
new_program_group. Eviction ran at CHURN_BETA=0 (byte-identical upstream) in
all three, and the workload seed was identical.

The question is not "which policy wins" but what the tradeoff curve looks like:
  - worst-case time-to-first-response (the head-of-line blocking)
  - aggregate throughput (what upstream optimises for)
  - group completion time (what RL rollouts actually need)

Usage:
    python3 analysis/compare_admission.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PAUSE_RE = re.compile(r"paused (?:ACTING|REASONING) program (\S+) \(tokens=(\d+)\)")
MARK_RE = re.compile(r"marked (?:ACTING|REASONING) program (\S+) for pause \(tokens=(\d+)\)")
DEC_RE = re.compile(r"CHURN_DECISION .* chosen=(\S+) tokens=")

ARMS = ["smallest", "fifo", "largest"]


def load(arm, results_dir="results"):
    cj = Path(results_dir) / f"admit_{arm}.json"
    log = Path(results_dir) / f"scheduler_admit_{arm}.log"
    if not cj.exists():
        return None

    evict = defaultdict(int)
    if log.exists():
        for line in log.read_text(errors="ignore").splitlines():
            m = DEC_RE.search(line)
            if m:
                evict[m.group(1)] += 1
                continue
            m = PAUSE_RE.search(line) or MARK_RE.search(line)
            if m:
                evict[m.group(1)] += 1

    data = json.loads(cj.read_text())
    per = defaultdict(list)
    for e in data["events"]:
        per[e["program_id"]].append(e)

    progs = {}
    for pid, evs in per.items():
        evs.sort(key=lambda e: e["wallclock"])
        s0 = evs[0]
        t_done = evs[-1]["wallclock"] + evs[-1]["latency_s"]
        progs[pid] = {
            "size": s0["start_tokens"],
            "group": s0.get("group"),
            "ttft": s0["latency_s"],
            "t_admitted": s0["wallclock"],
            "t_done": t_done,
            "post_admission": max(t_done - s0["wallclock"], 0.0),
            "evictions": evict.get(pid, 0),
        }
    ok = sum(1 for e in data["events"] if e["ok"])
    return {"arm": arm, "progs": progs, "elapsed": data["elapsed_s"],
            "ok": ok, "n_events": len(data["events"])}


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def f(c):
    return "n/a" if c is None else f"{c:+.3f}"


def group_times(progs):
    groups = defaultdict(list)
    for v in progs.values():
        if v["group"] is not None:
            groups[v["group"]].append(v)
    out = []
    for g, mem in sorted(groups.items()):
        first_req = min(m["t_admitted"] - m["ttft"] for m in mem)
        done = max(m["t_done"] for m in mem)
        out.append({"g": g, "size": int(np.mean([m["size"] for m in mem])),
                    "completion": done - first_req,
                    "evictions": sum(m["evictions"] for m in mem)})
    return out


def main():
    runs = {}
    for arm in ARMS:
        r = load(arm)
        if r:
            runs[arm] = r
            print(f"  loaded {arm}: {len(r['progs'])} programs, "
                  f"{r['ok']}/{r['n_events']} ok")
        else:
            print(f"  MISSING admit_{arm}.json")
    if len(runs) < 2:
        print("need at least two arms")
        return

    print("\n" + "=" * 78)
    print("  SANITY CHECK: does 'smallest' reproduce the known baseline?")
    print("=" * 78)
    if "smallest" in runs:
        p = runs["smallest"]["progs"]
        c = corr([v["size"] for v in p.values()], [v["ttft"] for v in p.values()])
        print(f"\n  corr(size, TTFT) for smallest-first = {f(c)}")
        print(f"  Previously measured on this seed     = +0.855")
        if c is not None and c > 0.7:
            print(f"  OK: baseline reproduces, the other arms are trustworthy.")
        else:
            print(f"  WARNING: baseline does NOT reproduce. Something drifted.")
            print(f"  Do not trust the comparison below until this is explained.")

    print("\n" + "=" * 78)
    print("  TIME TO FIRST RESPONSE (the head-of-line blocking)")
    print("=" * 78)
    print(f"\n  {'arm':<12}{'corr(size,TTFT)':>17}{'median':>9}{'p95':>9}"
          f"{'p99':>9}{'MAX':>9}")
    print("  " + "-" * 66)
    for arm in ARMS:
        if arm not in runs:
            continue
        p = runs[arm]["progs"]
        t = np.array([v["ttft"] for v in p.values()])
        c = corr([v["size"] for v in p.values()], [v["ttft"] for v in p.values()])
        print(f"  {arm:<12}{f(c):>17}{np.median(t):>9.1f}"
              f"{np.percentile(t,95):>9.1f}{np.percentile(t,99):>9.1f}{t.max():>9.1f}")

    print("\n" + "=" * 78)
    print("  WHO PAYS? TTFT by size quartile")
    print("=" * 78)
    for arm in ARMS:
        if arm not in runs:
            continue
        p = list(runs[arm]["progs"].values())
        p.sort(key=lambda v: v["size"])
        q = max(len(p) // 4, 1)
        smallest_q = np.mean([v["ttft"] for v in p[:q]])
        largest_q = np.mean([v["ttft"] for v in p[-q:]])
        print(f"\n  {arm:<12} smallest 25% wait {smallest_q:7.1f}s   "
              f"largest 25% wait {largest_q:7.1f}s   "
              f"ratio {largest_q/max(smallest_q,0.01):6.1f}x")

    print("\n" + "=" * 78)
    print("  AGGREGATE COST (what upstream optimises for)")
    print("=" * 78)
    print(f"\n  {'arm':<12}{'makespan':>11}{'throughput':>13}{'evictions':>11}"
          f"{'max evict':>11}")
    print("  " + "-" * 58)
    base = runs.get("smallest")
    for arm in ARMS:
        if arm not in runs:
            continue
        r = runs[arm]
        ev = [v["evictions"] for v in r["progs"].values()]
        thr = r["ok"] / r["elapsed"] * 60
        delta = ""
        if base and arm != "smallest":
            d = (r["elapsed"] - base["elapsed"]) / base["elapsed"] * 100
            delta = f"  ({d:+.1f}%)"
        print(f"  {arm:<12}{r['elapsed']:>10.0f}s{thr:>13.1f}"
              f"{sum(ev):>11}{max(ev) if ev else 0:>11}{delta}")

    print("\n" + "=" * 78)
    print("  GROUP COMPLETION (what RL rollouts actually need)")
    print("=" * 78)
    print(f"\n  {'arm':<12}{'mean':>10}{'median':>10}{'MAX':>10}{'spread':>10}")
    print("  " + "-" * 52)
    for arm in ARMS:
        if arm not in runs:
            continue
        gt = group_times(runs[arm]["progs"])
        if not gt:
            continue
        c = np.array([g["completion"] for g in gt])
        print(f"  {arm:<12}{c.mean():>10.1f}{np.median(c):>10.1f}"
              f"{c.max():>10.1f}{c.max()-c.min():>10.1f}")

    print("\n  Per-group detail (completion seconds by arm):")
    allg = {}
    for arm in ARMS:
        if arm not in runs:
            continue
        for g in group_times(runs[arm]["progs"]):
            allg.setdefault(g["g"], {})["size"] = g["size"]
            allg[g["g"]][arm] = g["completion"]
    hdr = f"\n  {'grp':>4}{'size':>8}" + "".join(f"{a:>12}" for a in ARMS if a in runs)
    print(hdr)
    print("  " + "-" * (12 + 12 * len(runs)))
    for g in sorted(allg, key=lambda k: allg[k]["size"]):
        row = f"  {g:>4}{allg[g]['size']:>8}"
        for a in ARMS:
            if a in runs:
                row += f"{allg[g].get(a, float('nan')):>12.1f}"
        print(row)

    print("\n" + "=" * 78)
    print("  THE TRADEOFF, IN ONE LINE PER ARM")
    print("=" * 78)
    for arm in ARMS:
        if arm not in runs:
            continue
        r = runs[arm]
        t = np.array([v["ttft"] for v in r["progs"].values()])
        gt = group_times(r["progs"])
        gc = np.array([g["completion"] for g in gt]) if gt else np.array([0])
        print(f"\n  {arm:<10} worst TTFT {t.max():6.1f}s | "
              f"makespan {r['elapsed']:5.0f}s | worst group {gc.max():6.1f}s")


if __name__ == "__main__":
    main()
