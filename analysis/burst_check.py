"""
Does FIFO breach capacity in MORE, SMALLER bursts than smallest-first, or
FEWER, LARGER ones? Q1 in why_more_evictions.py refuted the "more resident KV"
hypothesis (FIFO's resident tokens were actually 2% LOWER at eviction time).
This tests the alternative: maybe it's breach FREQUENCY, not breach SIZE.

Groups pause events within 3s of each other into one "capacity breach burst"
and compares burst count vs evictions-per-burst across arms.

Usage:
    python3 analysis/burst_check.py
"""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

TS = r"^(\d+\.\d+)\s+"
PAUSE_RE = re.compile(TS + r".*paused (?:ACTING|REASONING) program (\S+) \(tokens=(\d+)\)")
MARK_RE = re.compile(TS + r".*marked (?:ACTING|REASONING) program (\S+) for pause \(tokens=(\d+)\)")
ARMS = ["smallest", "fifo", "largest"]
SEEDS = [100, 101, 102]


def load_pauses(arm, seed, results_dir="results"):
    log = Path(results_dir) / f"scheduler_admit_{arm}_s{seed}.log"
    if not log.exists():
        return None
    pauses = []
    for line in log.read_text(errors="ignore").splitlines():
        m = PAUSE_RE.match(line) or MARK_RE.match(line)
        if m:
            pauses.append(float(m.group(1)))
    return sorted(pauses)


def bursts(times, gap=3.0):
    if not times:
        return []
    out = [[times[0]]]
    for t in times[1:]:
        if t - out[-1][-1] <= gap:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def main():
    print(f"{'arm':<11}{'seed':>6}{'evictions':>11}{'bursts':>8}{'ev/burst':>10}")
    print("-" * 46)
    agg = defaultdict(lambda: {"ev": [], "bursts": [], "per": []})
    for arm in ARMS:
        for seed in SEEDS:
            p = load_pauses(arm, seed)
            if not p:
                continue
            b = bursts(p)
            nb = len(b)
            per = len(p) / nb
            print(f"{arm:<11}{seed:>6}{len(p):>11}{nb:>8}{per:>10.2f}")
            agg[arm]["ev"].append(len(p))
            agg[arm]["bursts"].append(nb)
            agg[arm]["per"].append(per)

    print(f"\n{'arm':<11}{'mean evict':>12}{'mean bursts':>13}{'mean ev/burst':>15}")
    print("-" * 51)
    for arm in ARMS:
        a = agg[arm]
        if not a["ev"]:
            continue
        print(f"{arm:<11}{np.mean(a['ev']):>12.1f}{np.mean(a['bursts']):>13.1f}"
              f"{np.mean(a['per']):>15.2f}")

    if agg["smallest"]["bursts"] and agg["fifo"]["bursts"]:
        s_b, f_b = np.mean(agg["smallest"]["bursts"]), np.mean(agg["fifo"]["bursts"])
        s_p, f_p = np.mean(agg["smallest"]["per"]), np.mean(agg["fifo"]["per"])
        print(f"\nFIFO vs smallest:")
        print(f"  breach bursts   : {f_b:.1f} vs {s_b:.1f}  ({(f_b-s_b)/s_b*100:+.0f}%)")
        print(f"  evictions/burst : {f_p:.2f} vs {s_p:.2f}  ({(f_p-s_p)/s_p*100:+.0f}%)")
        if f_b > s_b * 1.1 and abs(f_p - s_p) < 0.3:
            print("  => CONFIRMED: FIFO breaches capacity MORE OFTEN, not more")
            print("     severely. Each breach evicts a similar number of programs;")
            print("     there are just more breach events under FIFO.")
        elif f_p > s_p * 1.1:
            print("  => Breach SEVERITY is higher under FIFO (more evicted per")
            print("     breach), not frequency.")
        else:
            print("  => No clean pattern in burst structure either. Report the")
            print("     eviction-count difference as observed but unexplained.")


if __name__ == "__main__":
    main()
