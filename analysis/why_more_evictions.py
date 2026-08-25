"""
Why does FIFO admission produce MORE evictions than smallest-first?

Observed across 3 seeds:
    smallest: 38, 28, 40 evictions
    fifo:     48, 40, 46 evictions

Hypothesis: smallest-first admission fills the backend with SMALL programs, so
total resident KV stays low. FIFO admits large programs early, each consuming
far more KV, so capacity is hit sooner and more often.

  Q1. WHY more? Compare the resident set at each eviction moment: how many
      programs were resident, and how many tokens did they hold?
  Q2. Are they WORSE? More evictions is not automatically bad if they are
      spread across more programs rather than concentrated on a few.

Usage:
    python3 analysis/why_more_evictions.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TS = r"^(\d+\.\d+)\s+"
PAUSE_RE = re.compile(TS + r".*paused (?:ACTING|REASONING) program (\S+) \(tokens=(\d+)\)")
MARK_RE = re.compile(TS + r".*marked (?:ACTING|REASONING) program (\S+) for pause \(tokens=(\d+)\)")

ARMS = ["smallest", "fifo", "largest"]
SEEDS = [100, 101, 102]
PER_STEP_GROWTH = 400


def load(arm, seed, results_dir="results"):
    cj = Path(results_dir) / f"admit_{arm}_s{seed}.json"
    log = Path(results_dir) / f"scheduler_admit_{arm}_s{seed}.log"
    if not (cj.exists() and log.exists()):
        return None

    pauses = []
    for line in log.read_text(errors="ignore").splitlines():
        m = PAUSE_RE.match(line) or MARK_RE.match(line)
        if m:
            pauses.append({"t": float(m.group(1)), "pid": m.group(2),
                           "tokens": int(m.group(3))})

    data = json.loads(cj.read_text())
    per = defaultdict(list)
    for e in data["events"]:
        per[e["program_id"]].append(e)

    progs = {}
    for pid, evs in per.items():
        evs.sort(key=lambda e: e["wallclock"])
        progs[pid] = {
            "size": evs[0]["start_tokens"],
            "t_admitted": evs[0]["wallclock"],
            "t_done": evs[-1]["wallclock"] + evs[-1]["latency_s"],
            "step_times": [e["wallclock"] for e in evs],
        }
    return {"pauses": pauses, "progs": progs, "elapsed": data["elapsed_s"],
            "t_start": min(p["t_admitted"] for p in progs.values())}


def tokens_at(prog, t):
    done = sum(1 for wc in prog["step_times"] if wc <= t)
    return prog["size"] + done * PER_STEP_GROWTH


def resident_state(run, t):
    n, tok = 0, 0
    for p in run["progs"].values():
        if p["t_admitted"] <= t <= p["t_done"]:
            n += 1
            tok += tokens_at(p, t)
    return n, tok


def gini(c):
    a = np.sort(np.asarray(c, float))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


def main():
    print("=" * 78)
    print("  Q1. WHY does FIFO evict more? Resident set at each eviction moment")
    print("=" * 78)
    print()
    print(f"  {'arm':<11}{'seed':>6}{'evictions':>11}{'1st evict':>11}"
          f"{'resident n':>12}{'resident tokens':>17}")
    print("  " + "-" * 68)

    agg = defaultdict(lambda: {"n": [], "tok": [], "first": [], "ev": []})
    for arm in ARMS:
        for seed in SEEDS:
            run = load(arm, seed)
            if not run or not run["pauses"]:
                continue
            ns, toks = [], []
            for p in run["pauses"]:
                n, tok = resident_state(run, p["t"])
                ns.append(n)
                toks.append(tok)
            first = run["pauses"][0]["t"] - run["t_start"]
            nev = len(run["pauses"])
            mn, mt = float(np.mean(ns)), float(np.mean(toks))
            print(f"  {arm:<11}{seed:>6}{nev:>11}{first:>10.0f}s"
                  f"{mn:>12.1f}{mt:>17,.0f}")
            agg[arm]["n"].append(mn)
            agg[arm]["tok"].append(mt)
            agg[arm]["first"].append(first)
            agg[arm]["ev"].append(nev)

    print()
    print(f"  {'arm':<11}{'mean evict':>12}{'1st evict':>11}"
          f"{'resident n':>12}{'resident tokens':>17}")
    print("  " + "-" * 63)
    for arm in ARMS:
        a = agg[arm]
        if not a["ev"]:
            continue
        print(f"  {arm:<11}{np.mean(a['ev']):>12.1f}{np.mean(a['first']):>10.0f}s"
              f"{np.mean(a['n']):>12.1f}{np.mean(a['tok']):>17,.0f}")

    if agg["smallest"]["tok"] and agg["fifo"]["tok"]:
        s_tok, f_tok = np.mean(agg["smallest"]["tok"]), np.mean(agg["fifo"]["tok"])
        s_n, f_n = np.mean(agg["smallest"]["n"]), np.mean(agg["fifo"]["n"])
        print()
        print("  FIFO vs smallest, measured at eviction moments:")
        print(f"    resident programs : {f_n:.1f} vs {s_n:.1f} "
              f"({(f_n - s_n) / s_n * 100:+.0f}%)")
        print(f"    resident tokens   : {f_tok:,.0f} vs {s_tok:,.0f} "
              f"({(f_tok - s_tok) / s_tok * 100:+.0f}%)")
        if f_tok > s_tok * 1.05:
            print("    => CONFIRMED: FIFO holds more KV per resident program, so")
            print("       capacity is hit more often. The extra evictions follow")
            print("       directly from admitting large programs earlier.")
        elif f_n > s_n * 1.05:
            print("    => Resident COUNT is higher but tokens are not.")
            print("       Different mechanism than hypothesised.")
        else:
            print("    => NOT confirmed. Neither count nor tokens explain it.")

    print()
    print("=" * 78)
    print("  Q2. Are the extra evictions WORSE, or just spread wider?")
    print("=" * 78)
    print()
    print(f"  {'arm':<11}{'seed':>6}{'total':>8}{'progs hit':>12}"
          f"{'max/prog':>10}{'Gini':>8}")
    print("  " + "-" * 55)

    agg2 = defaultdict(lambda: {"tot": [], "hit": [], "max": [], "gini": []})
    for arm in ARMS:
        for seed in SEEDS:
            run = load(arm, seed)
            if not run or not run["pauses"]:
                continue
            c = defaultdict(int)
            for p in run["pauses"]:
                c[p["pid"]] += 1
            vals = [c.get(pid, 0) for pid in run["progs"]]
            print(f"  {arm:<11}{seed:>6}{sum(vals):>8}"
                  f"{sum(1 for v in vals if v > 0):>12}"
                  f"{max(vals):>10}{gini(vals):>8.3f}")
            agg2[arm]["tot"].append(sum(vals))
            agg2[arm]["hit"].append(sum(1 for v in vals if v > 0))
            agg2[arm]["max"].append(max(vals))
            agg2[arm]["gini"].append(gini(vals))

    print()
    print(f"  {'arm':<11}{'mean total':>12}{'progs hit':>12}"
          f"{'max/prog':>10}{'Gini':>8}")
    print("  " + "-" * 53)
    for arm in ARMS:
        a = agg2[arm]
        if not a["tot"]:
            continue
        print(f"  {arm:<11}{np.mean(a['tot']):>12.1f}{np.mean(a['hit']):>12.1f}"
              f"{np.mean(a['max']):>10.1f}{np.mean(a['gini']):>8.3f}")

    if agg2["smallest"]["max"] and agg2["fifo"]["max"]:
        print()
        print("  Max evictions on any single program, per seed:")
        print(f"    smallest: {agg2['smallest']['max']}")
        print(f"    fifo    : {agg2['fifo']['max']}")
        s_max, f_max = np.mean(agg2["smallest"]["max"]), np.mean(agg2["fifo"]["max"])
        s_g, f_g = np.mean(agg2["smallest"]["gini"]), np.mean(agg2["fifo"]["gini"])
        print()
        print(f"    mean max/program : {f_max:.1f} (fifo) vs {s_max:.1f} (smallest)")
        print(f"    mean Gini        : {f_g:.3f} (fifo) vs {s_g:.3f} (smallest)")
        if f_max <= s_max and f_g <= s_g:
            print("    => FIFO evicts more in total but spreads it wider: no single")
            print("       program is hit harder, and the distribution is flatter.")
        elif f_max > s_max:
            print("    => FIFO hits its worst-case program harder. Genuine")
            print("       regression; report it plainly.")
        else:
            print("    => Mixed. Report both numbers without a verdict.")


if __name__ == "__main__":
    main()
