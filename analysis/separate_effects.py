"""
Separate admission wait from eviction cost.

Problem this solves:
    Earlier completion-time numbers (group blocking, the "+24s eviction cost")
    measured wall-clock from a program's FIRST REQUEST to its LAST STEP. That
    window contains two very different things:

      1. Admission wait  - time queued before ThunderAgent admits the program.
                           Correlates +0.75 to +0.87 with program SIZE.
                           Median ~140s, up to 400s+.
      2. Post-admission  - actual execution, where eviction happens.
                           Median step latency ~0.6s.

    Since admission wait is huge and size-driven, it swamps eviction effects
    and contaminates any "did eviction cost time?" comparison.

Method:
    Split each trajectory at the moment step 0 completes:
      t_admitted       = wallclock when step 0 finished (program is now in)
      admission_s      = step 0 latency (the queue wait)
      post_admission_s = t_done - t_admitted (execution, where eviction bites)

    Re-run the harm analysis on post_admission_s ONLY. Any remaining
    correlation with eviction count is a clean eviction effect.

Validated on synthetic data with a known +15s injected eviction effect:
recovers +14.9s, and shows the contaminated metric reports -37.0s (wrong sign).

Usage:
    python3 analysis/separate_effects.py --runs s21 s22 s23
    python3 analysis/separate_effects.py --runs grpo_run:grpo grpo101 grpo102
    (json:log syntax when the log name differs from the json name)
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PAUSE_RE = re.compile(r"paused (?:ACTING|REASONING) program (\S+) \(tokens=(\d+)\)")
MARK_RE = re.compile(r"marked (?:ACTING|REASONING) program (\S+) for pause \(tokens=(\d+)\)")
DEC_RE = re.compile(r"CHURN_DECISION .* chosen=(\S+) tokens=")


def load(json_tag, log_tag, results_dir="results"):
    cj = Path(results_dir) / f"{json_tag}.json"
    log = Path(results_dir) / f"scheduler_{log_tag}.log"
    if not (cj.exists() and log.exists()):
        return None

    evict = defaultdict(int)
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

    trajs = {}
    for pid, evs in per.items():
        evs.sort(key=lambda e: e["wallclock"])
        s0 = evs[0]
        t_admitted = s0["wallclock"]
        t_done = evs[-1]["wallclock"] + evs[-1]["latency_s"]
        trajs[pid] = {
            "start_tokens": s0["start_tokens"],
            "group": s0.get("group"),
            "evictions": evict.get(pid, 0),
            "admission_s": s0["latency_s"],
            "t_admitted": t_admitted,
            "t_done": t_done,
            "post_admission_s": max(t_done - t_admitted, 0.0),
            "total_s": s0["latency_s"] + max(t_done - t_admitted, 0.0),
        }
    return trajs


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def fmt(c):
    return "n/a" if c is None else f"{c:+.3f}"


def within_band(trajs_list, field, n_bands=3, label=""):
    rows = []
    for t in trajs_list:
        for v in t.values():
            rows.append((v["start_tokens"], v["evictions"], v[field]))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    per_band = max(len(rows) // n_bands, 1)

    print(f"\n  {label}")
    print(f"  {'size band':<22}{'n':>4}{'evicted':>9}{'clean':>7}"
          f"{'ev mean':>10}{'clean mean':>12}{'delta':>10}")
    print("  " + "-" * 74)
    deltas = []
    for b in range(n_bands):
        chunk = rows[b * per_band:(b + 1) * per_band] if b < n_bands - 1 else rows[b * per_band:]
        if not chunk:
            continue
        ev = [r for r in chunk if r[1] > 0]
        cl = [r for r in chunk if r[1] == 0]
        lo, hi = chunk[0][0], chunk[-1][0]
        if ev and cl:
            m_ev, m_cl = float(np.mean([r[2] for r in ev])), float(np.mean([r[2] for r in cl]))
            deltas.append(m_ev - m_cl)
            print(f"  {f'{lo:,}-{hi:,}':<22}{len(chunk):>4}{len(ev):>9}{len(cl):>7}"
                  f"{m_ev:>10.1f}{m_cl:>12.1f}{m_ev-m_cl:>+10.1f}")
        else:
            print(f"  {f'{lo:,}-{hi:,}':<22}{len(chunk):>4}{len(ev):>9}{len(cl):>7}"
                  f"{'--':>10}{'--':>12}{'--':>10}")
    if deltas:
        print(f"  {'':<22}{'':>4}{'':>9}{'':>7}{'':>10}{'MEAN':>12}{np.mean(deltas):>+10.1f}")
        return float(np.mean(deltas))
    return None


def group_decomposition(trajs_list):
    print(f"\n  {'grp':>4}{'size':>8}{'evict':>7}{'admit_span':>12}"
          f"{'post_span':>11}{'total':>9}{'admit %':>9}")
    print("  " + "-" * 62)
    rows = []
    for trajs in trajs_list:
        groups = defaultdict(list)
        for pid, v in trajs.items():
            if v["group"] is not None:
                groups[v["group"]].append(v)
        for g, mem in sorted(groups.items()):
            size = int(np.mean([m["start_tokens"] for m in mem]))
            ev = sum(m["evictions"] for m in mem)
            last_admit = max(m["t_admitted"] for m in mem)
            first_req = min(m["t_admitted"] - m["admission_s"] for m in mem)
            done = max(m["t_done"] for m in mem)
            admit_span = last_admit - first_req
            post_span = done - last_admit
            total = done - first_req
            rows.append({"g": g, "size": size, "ev": ev, "admit": admit_span,
                         "post": post_span, "total": total})
            print(f"  {g:>4}{size:>8}{ev:>7}{admit_span:>12.1f}"
                  f"{post_span:>11.1f}{total:>9.1f}"
                  f"{admit_span/total*100 if total else 0:>8.0f}%")
    if rows:
        a = np.array([r["admit"] for r in rows])
        p = np.array([r["post"] for r in rows])
        tt = np.array([r["total"] for r in rows])
        print(f"\n  Admission phase is {a.sum()/tt.sum()*100:.0f}% of all group time")
        print(f"  corr(group size, admission span)      = "
              f"{fmt(corr([r['size'] for r in rows], a))}")
        print(f"  corr(group size, POST-admission span) = "
              f"{fmt(corr([r['size'] for r in rows], p))}")
        print(f"  corr(group evictions, POST-adm span)  = "
              f"{fmt(corr([r['ev'] for r in rows], p))}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    loaded = []
    for spec in args.runs:
        jt, lt = spec.split(":", 1) if ":" in spec else (spec, spec)
        t = load(jt, lt)
        if t:
            loaded.append(t)
            print(f"  loaded {jt} ({len(t)} programs, "
                  f"{sum(v['evictions'] for v in t.values())} evictions)")
        else:
            print(f"  MISSING {jt} / scheduler_{lt}.log")
    if not loaded:
        print("nothing loaded")
        return

    sizes, adm, post, tot, ev = [], [], [], [], []
    for t in loaded:
        for v in t.values():
            sizes.append(v["start_tokens"]); adm.append(v["admission_s"])
            post.append(v["post_admission_s"]); tot.append(v["total_s"])
            ev.append(v["evictions"])

    print("\n" + "=" * 78)
    print("  DECOMPOSITION: where does a program's wall-clock time go?")
    print("=" * 78)
    adm_a, post_a, tot_a = np.array(adm), np.array(post), np.array(tot)
    print(f"\n  n = {len(adm)} programs")
    print(f"  admission wait   mean={adm_a.mean():7.1f}s  median={np.median(adm_a):7.1f}s")
    print(f"  post-admission   mean={post_a.mean():7.1f}s  median={np.median(post_a):7.1f}s")
    print(f"  total            mean={tot_a.mean():7.1f}s  median={np.median(tot_a):7.1f}s")
    print(f"\n  Admission wait is {adm_a.sum()/tot_a.sum()*100:.0f}% of all program time.")

    print("\n" + "=" * 78)
    print("  WHICH EFFECT DRIVES WHICH PHASE?")
    print("=" * 78)
    print(f"\n  corr(size, admission wait)        = {fmt(corr(sizes, adm))}   <- admission bias")
    print(f"  corr(size, post-admission)        = {fmt(corr(sizes, post))}")
    print(f"  corr(evictions, admission wait)   = {fmt(corr(ev, adm))}")
    print(f"  corr(evictions, post-admission)   = {fmt(corr(ev, post))}   <- eviction effect")

    print("\n" + "=" * 78)
    print("  HARM RE-MEASURED WITH ADMISSION WAIT REMOVED")
    print("=" * 78)
    d_tot = within_band(loaded, "total_s", label="(a) OLD metric: total time, admission INCLUDED")
    d_post = within_band(loaded, "post_admission_s", label="(b) NEW metric: post-admission time ONLY")
    if d_tot is not None and d_post is not None:
        print(f"\n  Old harm estimate (contaminated): {d_tot:+.1f}s")
        print(f"  Clean harm estimate (eviction only): {d_post:+.1f}s")
        print(f"  Difference attributable to admission wait: {d_tot - d_post:+.1f}s")
        if d_tot * d_post < 0:
            print(f"  NOTE: the two estimates have OPPOSITE SIGNS. The old metric")
            print(f"  was not merely inflated, it pointed the wrong way, because")
            print(f"  size-driven admission wait dominated it.")
        elif abs(d_post) < abs(d_tot):
            print(f"  => admission wait accounted for "
                  f"{(1 - abs(d_post)/abs(d_tot))*100:.0f}% of the old number.")

    if any(v["group"] is not None for t in loaded for v in t.values()):
        print("\n" + "=" * 78)
        print("  GROUP-LEVEL DECOMPOSITION")
        print("=" * 78)
        group_decomposition(loaded)


if __name__ == "__main__":
    main()
