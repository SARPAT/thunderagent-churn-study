"""
A/B comparison: does fatigue-aware eviction spread the burden, and what
does it cost?

Both runs use the SAME workload seed. The only difference is CHURN_BETA.
beta=0 short-circuits to candidates[0], i.e. exactly upstream behaviour.

Reads the CHURN_DECISION lines the patched router emits:
  CHURN_DECISION phase=acting beta=1.00 n=11 chosen=prog-007 tokens=2100
                 prior_evictions=2 baseline=prog-003
"chosen" is what the policy picked; "baseline" is what upstream would have
picked -- so we can count divergences directly, not by replay.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DEC_RE = re.compile(
    r"CHURN_DECISION phase=(\w+) beta=([\d.]+) n=(\d+) chosen=(\S+) "
    r"tokens=(\d+) prior_evictions=(\d+) baseline=(\S+)"
)


def parse(log):
    out = []
    for line in Path(log).read_text(errors="ignore").splitlines():
        m = DEC_RE.search(line)
        if m:
            out.append({
                "phase": m.group(1), "beta": float(m.group(2)),
                "n": int(m.group(3)), "chosen": m.group(4),
                "tokens": int(m.group(5)), "prior": int(m.group(6)),
                "baseline": m.group(7),
            })
    return out


def gini(c):
    a = np.sort(np.asarray(c, float))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


def summarize(log, client, label):
    decs = parse(log)
    data = json.loads(Path(client).read_text())
    starts = {}
    for e in data["events"]:
        starts.setdefault(e["program_id"], e["start_tokens"])

    counts = defaultdict(int)
    for d in decs:
        counts[d["chosen"]] += 1
    vals = [counts[p] for p in starts]

    diverged = sum(1 for d in decs if d["chosen"] != d["baseline"])
    ok = sum(1 for e in data["events"] if e["ok"])
    elapsed = data["elapsed_s"]

    return {
        "label": label,
        "beta": decs[0]["beta"] if decs else None,
        "n_decisions": len(decs),
        "mean_candidates": float(np.mean([d["n"] for d in decs])) if decs else 0,
        "diverged": diverged,
        "max_evict": max(vals) if vals else 0,
        "gini": gini(vals),
        "n_evicted_progs": sum(1 for v in vals if v > 0),
        "elapsed_s": elapsed,
        "throughput": ok / elapsed * 60,
        "counts": dict(counts),
        "starts": starts,
    }


def main():
    a = summarize("results/scheduler_beta0.log", "results/beta0.json", "beta=0 (upstream)")
    b = summarize("results/scheduler_beta1.log", "results/beta1.json", "beta=1 (fatigue)")

    print(f"{'metric':<34}{'beta=0':>14}{'beta=1':>14}{'change':>12}")
    print("-" * 74)

    def row(name, ka, kb, fmt="{:.0f}", pct=False):
        va, vb = a[ka], b[ka] if kb is None else b[kb]
        if pct and va:
            ch = f"{(vb-va)/va*100:+.1f}%"
        else:
            ch = f"{vb-va:+.2f}" if isinstance(va, float) else f"{vb-va:+d}"
        print(f"{name:<34}{fmt.format(va):>14}{fmt.format(vb):>14}{ch:>12}")

    row("eviction decisions", "n_decisions", None)
    row("mean candidates/decision", "mean_candidates", None, "{:.1f}")
    row("decisions diverged from upstream", "diverged", None)
    print("-" * 74)
    row("MAX evictions on one program", "max_evict", None)
    row("Gini of eviction counts", "gini", None, "{:.3f}")
    row("programs bearing evictions", "n_evicted_progs", None)
    print("-" * 74)
    row("elapsed (s)", "elapsed_s", None, "{:.0f}", pct=True)
    row("throughput (steps/min)", "throughput", None, "{:.1f}", pct=True)

    print("\nPer-program eviction counts (programs evicted in either run):")
    allp = sorted(set(list(a["counts"]) + list(b["counts"])),
                  key=lambda p: a["starts"].get(p, 0))
    print(f"  {'program':>10} {'start_tok':>10} {'beta=0':>7} {'beta=1':>7}")
    for p in allp:
        print(f"  {p:>10} {a['starts'].get(p,0):>10} "
              f"{a['counts'].get(p,0):>7} {b['counts'].get(p,0):>7}")

    Path("results/ab_summary.json").write_text(json.dumps(
        {"beta0": {k: v for k, v in a.items() if k != "starts"},
         "beta1": {k: v for k, v in b.items() if k != "starts"}}, indent=2))
    print("\nSaved -> results/ab_summary.json")


if __name__ == "__main__":
    main()
