"""
Group-level analysis of ThunderAgent's group-blind eviction.

In GRPO, advantages need ALL G rollouts of a group. So the meaningful unit is
group completion time. ThunderAgent schedules per-trajectory and has no group
concept. This asks what that costs.

  G1. Blocking waste     - how long does a group wait on its last straggler?
  G2. Eviction timing    - do evictions hit nearly-complete groups?
  G3. Whole-group target - does shortest-first hit entire small groups
                            (since GRPO siblings share a prompt => similar size)?
  G4. Natural ordering   - do small groups still finish fastest, or does
                            eviction invert that?

Usage:
    python3 analysis/grpo_analysis.py results/scheduler_grpo.log results/grpo_run.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TS = r"^(\d+\.\d+)\s+"
PAUSE_RE = re.compile(TS + r".*paused (?:ACTING|REASONING) program (\S+) \(tokens=(\d+)\)")
ID_RE = re.compile(r"^g(\d+)t(\d+)$")


def parse_pauses(log):
    out = []
    for line in Path(log).read_text(errors="ignore").splitlines():
        m = PAUSE_RE.match(line)
        if m:
            pid = m.group(2)
            g = ID_RE.match(pid)
            if g:
                out.append({"t": float(m.group(1)), "pid": pid,
                            "group": int(g.group(1)), "tokens": int(m.group(3))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="results/scheduler_grpo.log")
    ap.add_argument("client", nargs="?", default="results/grpo_run.json")
    args = ap.parse_args()

    pauses = parse_pauses(args.log)
    data = json.loads(Path(args.client).read_text())
    G = data["config"]["group_size"]

    per_traj = defaultdict(list)
    for e in data["events"]:
        per_traj[e["program_id"]].append(e)
    traj = {}
    for pid, evs in per_traj.items():
        evs.sort(key=lambda e: e["wallclock"])
        traj[pid] = {
            "group": evs[0]["group"],
            "start_tokens": evs[0]["start_tokens"],
            "t_start": evs[0]["wallclock"],
            "t_end": evs[-1]["wallclock"] + evs[-1]["latency_s"],
        }

    groups = defaultdict(list)
    for pid, v in traj.items():
        groups[v["group"]].append(v)

    ev_count = defaultdict(int)
    for p in pauses:
        ev_count[p["pid"]] += 1
    grp_ev = defaultdict(int)
    for p in pauses:
        grp_ev[p["group"]] += 1

    print(f"Groups: {len(groups)}   members/group: {G}   "
          f"pause events: {len(pauses)}\n")

    print("=" * 72)
    print("  G1. Blocking waste -- time a group waits on its last straggler")
    print("=" * 72)
    print(f"\n  {'grp':>4}{'size':>8}{'evict':>7}{'first_done':>12}"
          f"{'last_done':>11}{'blocked_s':>11}{'%of span':>10}")
    rows = []
    for g in sorted(groups):
        mem = groups[g]
        t0 = min(m["t_start"] for m in mem)
        ends = sorted(m["t_end"] for m in mem)
        span = ends[-1] - t0
        blocked = ends[-1] - ends[0]
        size = int(np.mean([m["start_tokens"] for m in mem]))
        rows.append({"g": g, "size": size, "ev": grp_ev.get(g, 0), "span": span,
                     "blocked": blocked, "completion": ends[-1] - t0})
        print(f"  {g:>4}{size:>8}{grp_ev.get(g,0):>7}{ends[0]-t0:>12.1f}"
              f"{ends[-1]-t0:>11.1f}{blocked:>11.1f}"
              f"{blocked/span*100 if span else 0:>9.0f}%")

    bl = np.array([r["blocked"] for r in rows])
    print(f"\n  Blocking window: mean={bl.mean():.1f}s  max={bl.max():.1f}s")
    print(f"  (in synchronous GRPO this is idle time before advantages compute)")

    print("\n" + "=" * 72)
    print("  G2. Did evictions hit nearly-complete groups?")
    print("=" * 72)
    prog_hist = defaultdict(int)
    for p in pauses:
        sibs = groups[p["group"]]
        done = sum(1 for m in sibs if m["t_end"] <= p["t"])
        prog_hist[done] += 1
    if prog_hist:
        tot = sum(prog_hist.values())
        print(f"\n  Siblings already finished when the eviction happened:")
        for k in range(G + 1):
            c = prog_hist.get(k, 0)
            if c:
                print(f"    {k}/{G} done: {'#'*c} ({c}, {c/tot*100:.0f}%)")
        mean_prog = sum(k * c for k, c in prog_hist.items()) / tot
        print(f"\n  Mean group progress at eviction: {mean_prog:.2f}/{G}")
        print(f"  Uniform would be ~{G/2:.1f}. Higher => evictions land on")
        print(f"  groups close to done (more costly to delay).")
    else:
        print("\n  No evictions matched to groups.")

    print("\n" + "=" * 72)
    print("  G3. Is eviction concentrated on whole (small) groups?")
    print("=" * 72)
    sizes = np.array([r["size"] for r in rows], float)
    evs = np.array([r["ev"] for r in rows], float)
    if evs.std() > 0 and sizes.std() > 0:
        print(f"\n  corr(group size, evictions on that group) = "
              f"{np.corrcoef(sizes, evs)[0,1]:+.3f}")
    hit = [r for r in rows if r["ev"] > 0]
    clean = [r for r in rows if r["ev"] == 0]
    if hit and clean:
        print(f"  Groups ever evicted   : {len(hit)}  "
              f"(sizes {min(r['size'] for r in hit):,}-{max(r['size'] for r in hit):,})")
        print(f"  Groups never evicted  : {len(clean)}  "
              f"(sizes {min(r['size'] for r in clean):,}-{max(r['size'] for r in clean):,})")
    n_hit = sum(1 for g in groups if grp_ev.get(g, 0) > 0)
    worst = max((grp_ev.get(g, 0) for g in groups), default=0)
    print(f"  Evictions landed on {n_hit}/{len(groups)} groups; "
          f"worst single group took {worst}")

    print("\n" + "=" * 72)
    print("  G4. Are SHORT groups penalised? (they should finish fastest)")
    print("=" * 72)
    comp = np.array([r["completion"] for r in rows], float)
    if sizes.std() > 0 and comp.std() > 0:
        c = np.corrcoef(sizes, comp)[0, 1]
        print(f"\n  corr(group size, group completion time) = {c:+.3f}")
        print(f"  Expected POSITIVE (bigger prompts take longer).")
        if c < 0.1:
            print(f"  -> Weak or inverted: small groups are NOT finishing first,")
            print(f"     consistent with group-blind eviction hurting them.")
        else:
            print(f"  -> Natural ordering largely preserved.")

    srt = sorted(rows, key=lambda r: r["size"])
    print(f"\n  {'group':>6}{'size':>9}{'evictions':>11}{'completion_s':>14}")
    for r in srt:
        print(f"  {r['g']:>6}{r['size']:>9}{r['ev']:>11}{r['completion']:>14.1f}")

    Path("results/grpo_analysis.json").write_text(json.dumps({
        "n_groups": len(groups), "group_size": G, "n_pauses": len(pauses),
        "mean_blocking_s": float(bl.mean()), "max_blocking_s": float(bl.max()),
        "progress_hist": dict(prog_hist), "rows": rows,
    }, indent=2))
    print("\nSaved -> results/grpo_analysis.json")


if __name__ == "__main__":
    main()
