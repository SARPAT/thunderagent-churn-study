"""
Paired A/B analysis across multiple seeds.

For each seed we ran the SAME workload twice: once with beta=0 (which
short-circuits to upstream's shortest-first) and once with beta=1 (fatigue-
aware). Pairing by seed means workload variation cancels out, so the
per-pair delta isolates the policy effect.

beta=0 MUST show 0 divergences. If it doesn't, the control is broken and
nothing else in the table can be trusted.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DEC_RE = re.compile(
    r"CHURN_DECISION phase=(\w+) beta=([\d.]+) n=(\d+) chosen=(\S+) "
    r"tokens=(\d+) prior_evictions=(\d+) baseline=(\S+)"
)


def gini(c):
    a = np.sort(np.asarray(c, float))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


def load_run(tag, results_dir="results"):
    log = Path(results_dir) / f"scheduler_{tag}.log"
    cj = Path(results_dir) / f"{tag}.json"
    if not (log.exists() and cj.exists()):
        return None

    decs = []
    for line in log.read_text(errors="ignore").splitlines():
        m = DEC_RE.search(line)
        if m:
            decs.append({"n": int(m.group(3)), "chosen": m.group(4),
                         "beta": float(m.group(2)), "baseline": m.group(7)})

    data = json.loads(cj.read_text())
    starts = {}
    for e in data["events"]:
        starts.setdefault(e["program_id"], e["start_tokens"])

    counts = defaultdict(int)
    for d in decs:
        counts[d["chosen"]] += 1
    vals = [counts[p] for p in starts]
    ok = sum(1 for e in data["events"] if e["ok"])

    return {
        "tag": tag,
        "n_decisions": len(decs),
        "diverged": sum(1 for d in decs if d["chosen"] != d["baseline"]),
        "max_evict": max(vals) if vals else 0,
        "gini": gini(vals),
        "n_bearing": sum(1 for v in vals if v > 0),
        "throughput": ok / data["elapsed_s"] * 60,
        "elapsed": data["elapsed_s"],
        "counts": dict(counts),
        "starts": starts,
    }


def main(seeds=(21, 22, 23)):
    pairs = []
    for s in seeds:
        a, b = load_run(f"s{s}_b0"), load_run(f"s{s}_b1")
        if a and b:
            pairs.append((s, a, b))
        else:
            print(f"seed {s}: missing files, skipped")

    if not pairs:
        print("No complete pairs found.")
        return

    bad = [s for s, a, b in pairs if a["diverged"] != 0]
    print("CONTROL CHECK: beta=0 must never diverge from upstream")
    if bad:
        print(f"  FAILED for seeds {bad} -- control is not upstream. Stop here.")
    else:
        print(f"  OK: 0 divergences across all {len(pairs)} control runs\n")

    hdr = (f"{'seed':>5} {'decisions':>19} {'max evict':>13} "
           f"{'Gini':>15} {'throughput':>17} {'diverg':>7}")
    print(hdr)
    print(f"{'':>5} {'b0':>9}{'b1':>10} {'b0':>6}{'b1':>7} "
          f"{'b0':>7}{'b1':>8} {'b0':>8}{'b1':>9} {'(b1)':>7}")
    print("-" * len(hdr))

    for s, a, b in pairs:
        print(f"{s:>5} {a['n_decisions']:>9}{b['n_decisions']:>10} "
              f"{a['max_evict']:>6}{b['max_evict']:>7} "
              f"{a['gini']:>7.3f}{b['gini']:>8.3f} "
              f"{a['throughput']:>8.1f}{b['throughput']:>9.1f} "
              f"{b['diverged']:>7}")

    def arr(key, idx):
        return np.array([p[idx][key] for p in pairs], float)

    m0, m1 = arr("max_evict", 1), arr("max_evict", 2)
    g0, g1 = arr("gini", 1), arr("gini", 2)
    t0, t1 = arr("throughput", 1), arr("throughput", 2)
    n0, n1 = arr("n_bearing", 1), arr("n_bearing", 2)

    print("\n" + "=" * 64)
    print("  AGGREGATE (mean +/- sd across seeds)")
    print("=" * 64)

    def line(name, a, b, fmt="{:.2f}", pct=True):
        d = b.mean() - a.mean()
        p = f"  ({d/a.mean()*100:+.1f}%)" if pct and a.mean() else ""
        print(f"  {name:<28} {fmt.format(a.mean())} +/- {fmt.format(a.std())}"
              f"   ->   {fmt.format(b.mean())} +/- {fmt.format(b.std())}"
              f"   delta={d:+.2f}{p}")

    line("max evictions / program", m0, m1)
    line("Gini", g0, g1, "{:.3f}")
    line("programs bearing evictions", n0, n1)
    line("throughput (steps/min)", t0, t1, "{:.1f}")

    dmax, dg = m1 - m0, g1 - g0
    dt = (t1 - t0) / t0 * 100
    print(f"\n  Paired deltas (per seed, b1 - b0):")
    print(f"    max evictions : {dmax.tolist()}")
    print(f"    Gini          : {[round(x,3) for x in dg.tolist()]}")
    print(f"    throughput %  : {[round(x,1) for x in dt.tolist()]}")
    print(f"\n  Consistent direction?  max evict {'YES' if all(d<=0 for d in dmax) else 'NO'}"
          f" | Gini {'YES' if all(d<=0 for d in dg) else 'NO'}")
    print(f"  Throughput change: {dt.mean():+.1f}% +/- {dt.std():.1f}%")
    if abs(dt.mean()) < dt.std() * 2:
        print(f"    -> within noise; report as 'no large penalty observed',")
        print(f"       NOT as a precise cost figure.")

    Path("results/sweep_summary.json").write_text(json.dumps([
        {"seed": s, "beta0": {k: v for k, v in a.items() if k not in ("starts", "counts")},
         "beta1": {k: v for k, v in b.items() if k not in ("starts", "counts")}}
        for s, a, b in pairs], indent=2))
    print("\nSaved -> results/sweep_summary.json")


if __name__ == "__main__":
    main()
