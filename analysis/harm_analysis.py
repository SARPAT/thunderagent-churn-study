"""
Does being evicted actually COST the program anything?

Everything so far measured WHO gets evicted. This measures WHAT IT COSTS them.
Without this, the finding is a fairness statistic with no demonstrated harm --
and "you improved a Gini coefficient" is not a result anyone ships.

  Q1 (harm exists?)  In the baseline (beta=0), do programs that were evicted
                     more finish later than comparable programs?

  Q2 (fix helps?)    Does beta=1 tighten the completion-time tail (p95/p99)
                     versus beta=0 on the same workload?

Q1 has a confound: short programs are BOTH evicted more AND naturally faster
(less context to process), so raw correlation is misleading. We therefore
compare WITHIN size bands -- evicted vs non-evicted programs of similar size.

Usage:
    python3 analysis/harm_analysis.py --pairs s21 s22 s23
    python3 analysis/harm_analysis.py --pairs hp hp42 hp43
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DEC_RE = re.compile(
    r"CHURN_DECISION phase=(\w+) beta=([\d.]+) n=(\d+) chosen=(\S+) "
    r"tokens=(\d+) prior_evictions=(\d+) baseline=(\S+)"
)


def load(tag, results_dir="results"):
    log = Path(results_dir) / f"scheduler_{tag}.log"
    cj = Path(results_dir) / f"{tag}.json"
    if not (log.exists() and cj.exists()):
        return None

    counts = defaultdict(int)
    for line in log.read_text(errors="ignore").splitlines():
        m = DEC_RE.search(line)
        if m:
            counts[m.group(4)] += 1

    data = json.loads(cj.read_text())
    per = defaultdict(list)
    for e in data["events"]:
        per[e["program_id"]].append(e)

    prog = {}
    for pid, evs in per.items():
        evs.sort(key=lambda e: e["wallclock"])
        span = evs[-1]["wallclock"] + evs[-1]["latency_s"] - evs[0]["wallclock"]
        prog[pid] = {
            "start_tokens": evs[0]["start_tokens"],
            "completion_s": span,
            "sum_latency": sum(e["latency_s"] for e in evs),
            "max_latency": max(e["latency_s"] for e in evs),
            "evictions": counts.get(pid, 0),
        }
    return prog


def safe_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def q1_within_band(progs_list, n_bands=3):
    rows = []
    for progs in progs_list:
        for pid, v in progs.items():
            rows.append((v["start_tokens"], v["evictions"], v["completion_s"],
                         v["max_latency"]))
    if not rows:
        return
    rows.sort(key=lambda r: r[0])
    per_band = max(len(rows) // n_bands, 1)

    print(f"\n  {'size band':<22}{'n':>4}{'evicted':>9}{'clean':>8}"
          f"{'compl(ev)':>11}{'compl(clean)':>14}{'delta':>10}")
    print("  " + "-" * 76)
    deltas = []
    for b in range(n_bands):
        chunk = rows[b * per_band:(b + 1) * per_band] if b < n_bands - 1 \
            else rows[b * per_band:]
        if not chunk:
            continue
        ev = [r for r in chunk if r[1] > 0]
        cl = [r for r in chunk if r[1] == 0]
        lo, hi = chunk[0][0], chunk[-1][0]
        if ev and cl:
            m_ev = float(np.mean([r[2] for r in ev]))
            m_cl = float(np.mean([r[2] for r in cl]))
            d = m_ev - m_cl
            deltas.append(d)
            print(f"  {f'{lo:,}-{hi:,} tok':<22}{len(chunk):>4}{len(ev):>9}{len(cl):>8}"
                  f"{m_ev:>11.1f}{m_cl:>14.1f}{d:>+10.1f}")
        else:
            print(f"  {f'{lo:,}-{hi:,} tok':<22}{len(chunk):>4}{len(ev):>9}{len(cl):>8}"
                  f"{'--':>11}{'--':>14}{'--':>10}")
    if deltas:
        print(f"\n  Mean within-band delta: {np.mean(deltas):+.1f}s")
        print(f"  (positive => evicted programs finish LATER than similar-sized")
        print(f"   programs that escaped eviction; this is the harm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["s21", "s22", "s23"])
    args = ap.parse_args()

    b0_list, b1_list = [], []
    for tag in args.pairs:
        a, b = load(f"{tag}_b0"), load(f"{tag}_b1")
        if a and b:
            b0_list.append(a)
            b1_list.append(b)
        else:
            print(f"  (skipped {tag}: missing files)")

    if not b0_list:
        print("No complete pairs found.")
        return

    print("=" * 78)
    print("  Q1. Does eviction actually delay a program? (baseline, beta=0)")
    print("=" * 78)

    all_ev, all_ct, all_st = [], [], []
    for progs in b0_list:
        for v in progs.values():
            all_ev.append(v["evictions"])
            all_ct.append(v["completion_s"])
            all_st.append(v["start_tokens"])

    c_raw = safe_corr(all_ev, all_ct)
    c_size = safe_corr(all_st, all_ct)
    print(f"\n  Raw corr(evictions, completion_time) = "
          f"{c_raw if c_raw is None else round(c_raw, 3)}")
    print(f"  corr(start_tokens, completion_time)  = "
          f"{c_size if c_size is None else round(c_size, 3)}")
    print(f"\n  CONFOUND: short programs are both evicted more AND naturally")
    print(f"  faster, so the raw number is not trustworthy. Compare within")
    print(f"  size bands instead:")
    q1_within_band(b0_list)

    print("\n" + "=" * 78)
    print("  Q2. Does the fix tighten the completion-time tail?")
    print("=" * 78)

    def tail(progs_list):
        ct = [v["completion_s"] for p in progs_list for v in p.values()]
        ml = [v["max_latency"] for p in progs_list for v in p.values()]
        return np.array(ct), np.array(ml)

    ct0, ml0 = tail(b0_list)
    ct1, ml1 = tail(b1_list)

    print(f"\n  {'metric':<28}{'beta=0':>12}{'beta=1':>12}{'change':>12}")
    print("  " + "-" * 64)
    for name, a, b in [("completion p50", np.percentile(ct0, 50), np.percentile(ct1, 50)),
                       ("completion p95", np.percentile(ct0, 95), np.percentile(ct1, 95)),
                       ("completion p99", np.percentile(ct0, 99), np.percentile(ct1, 99)),
                       ("completion max", ct0.max(), ct1.max()),
                       ("worst single step p99", np.percentile(ml0, 99), np.percentile(ml1, 99)),
                       ("worst single step max", ml0.max(), ml1.max())]:
        ch = f"{(b-a)/a*100:+.1f}%" if a else "--"
        print(f"  {name:<28}{a:>12.1f}{b:>12.1f}{ch:>12}")

    print(f"\n  n programs: beta=0 {len(ct0)}, beta=1 {len(ct1)} "
          f"(across {len(b0_list)} seed pairs)")
    print(f"\n  If p95/p99 completion improves, the fix has a user-visible")
    print(f"  benefit. If not, it only improves a fairness statistic -- say so.")


if __name__ == "__main__":
    main()
