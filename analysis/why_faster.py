"""
Why do LARGER programs have SHORTER post-admission time?

Observed: corr(size, post_admission_s) = -0.43 to -0.49.
Counter-intuitive: bigger contexts should be slower, not faster.

Hypothesis (causal chain):
    bigger size -> admitted LATER (admission is smallest-first)
                -> fewer programs still running by then
                -> less contention
                -> shorter execution

Tests:
  T1. corr(size, t_admitted)              is bigger really admitted later?
  T2. corr(t_admitted, post_admission)    do later admits run faster?
  T3. corr(concurrent_at_admit, post)     is contention the driver?
  T4. PARTIAL corr(size, post | concurrent)
        The decisive test. If controlling for concurrent load collapses the
        size correlation toward zero, the chain is confirmed and size has no
        direct effect on execution speed.

Partial-correlation math validated on synthetic cases: recovers +0.06 when
no direct effect exists, +0.57 when one does.

Usage:
    python3 analysis/why_faster.py --runs grpo_run:grpo grpo101 grpo102
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

    rows = []
    for pid, evs in per.items():
        evs.sort(key=lambda e: e["wallclock"])
        s0 = evs[0]
        t_adm = s0["wallclock"]
        t_done = evs[-1]["wallclock"] + evs[-1]["latency_s"]
        rows.append({
            "pid": pid, "size": s0["start_tokens"],
            "evictions": evict.get(pid, 0),
            "t_admitted": t_adm, "t_done": t_done,
            "admission_s": s0["latency_s"],
            "post_admission_s": max(t_done - t_adm, 0.0),
        })

    for r in rows:
        r["concurrent_at_admit"] = sum(
            1 for o in rows
            if o["pid"] != r["pid"]
            and o["t_admitted"] <= r["t_admitted"] <= o["t_done"]
        )
    t0 = min(r["t_admitted"] - r["admission_s"] for r in rows)
    for r in rows:
        r["admitted_at_rel"] = r["t_admitted"] - t0
    return rows


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def partial_corr(x, y, z):
    rxy, rxz, ryz = corr(x, y), corr(x, z), corr(y, z)
    if None in (rxy, rxz, ryz):
        return None
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if denom == 0:
        return None
    return float((rxy - rxz * ryz) / denom)


def f(c):
    return "n/a" if c is None else f"{c:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    allrows = []
    for spec in args.runs:
        jt, lt = spec.split(":", 1) if ":" in spec else (spec, spec)
        r = load(jt, lt)
        if r:
            allrows.extend(r)
            print(f"  loaded {jt} ({len(r)} programs)")
        else:
            print(f"  MISSING {jt}")
    if not allrows:
        return

    size = [r["size"] for r in allrows]
    post = [r["post_admission_s"] for r in allrows]
    adm_t = [r["admitted_at_rel"] for r in allrows]
    conc = [r["concurrent_at_admit"] for r in allrows]

    print("\n" + "=" * 76)
    print("  WHY DO LARGER PROGRAMS EXECUTE FASTER? (n = %d)" % len(allrows))
    print("=" * 76)

    print(f"\n  T1. corr(size, admission time)          = {f(corr(size, adm_t))}")
    print(f"      (positive => bigger programs admitted LATER)")
    print(f"\n  T2. corr(admission time, post-adm)      = {f(corr(adm_t, post))}")
    print(f"      (negative => later admits execute FASTER)")
    print(f"\n  T3. corr(concurrent at admit, post-adm) = {f(corr(conc, post))}")
    print(f"      (positive => more contention means slower execution)")
    print(f"\n  T4. corr(size, post-admission)          = {f(corr(size, post))}"
          f"   <- the puzzle")
    print(f"      PARTIAL, holding concurrency fixed    = "
          f"{f(partial_corr(size, post, conc))}   <- the answer")
    print(f"      PARTIAL, holding admit-time fixed     = "
          f"{f(partial_corr(size, post, adm_t))}")

    pc, raw = partial_corr(size, post, conc), corr(size, post)
    if pc is not None and raw is not None and raw:
        drop = (1 - abs(pc) / abs(raw)) * 100
        print(f"\n  Controlling for concurrency removes {drop:.0f}% of the size effect.")
        if abs(pc) < 0.15:
            print(f"  => CONFIRMED: size has no direct effect on execution speed.")
            print(f"     Large programs run faster only because they are admitted")
            print(f"     later, into a system that has already drained.")
        elif abs(pc) < abs(raw) * 0.6:
            print(f"  => PARTIALLY confirmed: contention explains most of it, but")
            print(f"     a residual size effect remains.")
        else:
            print(f"  => NOT confirmed: the size effect survives. Another")
            print(f"     mechanism is at work.")

    print("\n" + "=" * 76)
    print("  BY ADMISSION ORDER (thirds)")
    print("=" * 76)
    srt = sorted(allrows, key=lambda r: r["admitted_at_rel"])
    n3 = max(len(srt) // 3, 1)
    print(f"\n  {'cohort':<16}{'n':>4}{'mean size':>11}{'admitted at':>13}"
          f"{'concurrent':>12}{'post-adm':>11}{'evictions':>11}")
    print("  " + "-" * 74)
    for i, name in enumerate(["first third", "middle third", "last third"]):
        chunk = srt[i * n3:(i + 1) * n3] if i < 2 else srt[2 * n3:]
        if not chunk:
            continue
        print(f"  {name:<16}{len(chunk):>4}"
              f"{np.mean([r['size'] for r in chunk]):>11.0f}"
              f"{np.mean([r['admitted_at_rel'] for r in chunk]):>12.0f}s"
              f"{np.mean([r['concurrent_at_admit'] for r in chunk]):>12.1f}"
              f"{np.mean([r['post_admission_s'] for r in chunk]):>10.1f}s"
              f"{np.mean([r['evictions'] for r in chunk]):>11.2f}")

    print(f"\n  Chain holds if the last-third cohort has the LARGEST sizes,")
    print(f"  the LOWEST concurrency, and the SHORTEST post-admission time.")


if __name__ == "__main__":
    main()
