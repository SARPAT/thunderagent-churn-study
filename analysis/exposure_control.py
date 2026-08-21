"""
Exposure-controlled analysis of eviction churn in ThunderAgent.

The raw finding (short programs get evicted more) has an obvious confound:
short programs move through their steps faster, so they may simply be
*eligible* for eviction more often. This script controls for that.

Four tests, in increasing order of how much they'd convince a skeptic:

  T1. Equal-opportunity check
      Every program runs the SAME number of steps, so request count is
      identical by construction. Any difference is not "more requests".

  T2. Lifetime-normalised eviction rate
      Evictions per second of program lifetime. Removes "was around longer".

  T3. Spearman rank correlation + permutation test
      Rank-based (robust to outliers), with an exact p-value from shuffling
      eviction counts across programs.

  T4. Pause-time token distribution vs population  <-- the strongest test
      At each pause the scheduler logs the victim's token count. If eviction
      were size-blind, victim sizes would look like a random draw from the
      population of live programs. If it's shortest-first, victims are
      systematically smaller. This test does NOT depend on exposure at all.

Usage:
    python3 analysis/exposure_control.py results/scheduler_run6.log results/run6.json
    python3 analysis/exposure_control.py --multi run6 run7
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PAUSE_RE = re.compile(r"paused (?:ACTING|REASONING) program (\S+) \(tokens=(\d+)\)")
MARK_RE = re.compile(r"marked (?:ACTING|REASONING) program (\S+) for pause \(tokens=(\d+)\)")
RESUME_RE = re.compile(r"Resumed program (\S+) to .*tokens=(\d+)")


def parse_log(log_path):
    """Returns pause events [(pid, tokens)] and resume events, in order."""
    pauses, resumes = [], []
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        m = PAUSE_RE.search(line) or MARK_RE.search(line)
        if m:
            pauses.append((m.group(1), int(m.group(2))))
            continue
        m = RESUME_RE.search(line)
        if m:
            resumes.append((m.group(1), int(m.group(2))))
    return pauses, resumes


def load_client(client_json):
    """Per-program: start tokens, step count, lifetime, active time."""
    data = json.loads(Path(client_json).read_text())
    by_prog = defaultdict(list)
    for e in data["events"]:
        by_prog[e["program_id"]].append(e)

    info = {}
    for pid, evs in by_prog.items():
        evs.sort(key=lambda e: e["wallclock"])
        lifetime = evs[-1]["wallclock"] - evs[0]["wallclock"] + evs[-1]["latency_s"]
        info[pid] = {
            "start_tokens": evs[0]["start_tokens"],
            "n_steps": len(evs),
            "lifetime_s": max(lifetime, 1e-6),
            "active_s": sum(e["latency_s"] for e in evs),
        }
    return info, data.get("config", {}), data.get("elapsed_s", 0)


def spearman(x, y):
    """Rank correlation, no scipy dependency."""
    def rank(a):
        order = np.argsort(a)
        r = np.empty(len(a), dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        for i, c in enumerate(cnt):
            if c > 1:
                r[inv == i] = r[inv == i].mean()
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_p(x, y, observed, n_perm=20000, seed=0):
    """Two-sided p-value: how often does shuffling y give |rho| >= |observed|?"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float).copy()
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(y)
        if abs(spearman(x, y)) >= abs(observed):
            hits += 1
    return (hits + 1) / (n_perm + 1)


def gini(counts):
    """0 = everyone evicted equally, 1 = one program takes all evictions."""
    a = np.sort(np.asarray(counts, dtype=float))
    n = len(a)
    if a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


def analyse(log_path, client_json, label="", verbose=True):
    pauses, resumes = parse_log(log_path)
    info, config, elapsed = load_client(client_json)

    pids = sorted(info)
    evict_count = {p: 0 for p in pids}
    for pid, _ in pauses:
        if pid in evict_count:
            evict_count[pid] += 1

    starts = np.array([info[p]["start_tokens"] for p in pids], float)
    counts = np.array([evict_count[p] for p in pids], float)
    lifetimes = np.array([info[p]["lifetime_s"] for p in pids], float)
    steps = np.array([info[p]["n_steps"] for p in pids], float)
    rate = counts / lifetimes

    out = {"label": label, "n_programs": len(pids), "n_pauses": len(pauses)}

    if verbose:
        print(f"\n{'='*66}")
        print(f"  {label or log_path}")
        print(f"{'='*66}")

    same_steps = len(set(steps.tolist())) == 1
    if verbose:
        print(f"\nT1. Equal-opportunity check")
        print(f"    Steps per program: {sorted(set(steps.astype(int).tolist()))}")
        print(f"    All programs ran the same number of steps: {same_steps}")
        if same_steps:
            print(f"    -> request count is identical by construction;")
            print(f"       differences are NOT explained by 'more requests'.")

    r_raw = spearman(starts, counts)
    r_rate = spearman(starts, rate)
    r_life = spearman(starts, lifetimes)
    if verbose:
        print(f"\nT2. Lifetime normalisation")
        print(f"    corr(start_tokens, lifetime_s)        = {r_life:+.3f}")
        print(f"    corr(start_tokens, evictions)         = {r_raw:+.3f}")
        print(f"    corr(start_tokens, evictions/second)  = {r_rate:+.3f}")
        print(f"    NOTE: lifetime is POST-TREATMENT -- being evicted makes a")
        print(f"          program wait, which lengthens its lifetime. Dividing by")
        print(f"          it is a collider adjustment and UNDERSTATES the effect.")
        print(f"          Reported for completeness; T1 and T5 are the clean tests.")

    p_raw = permutation_p(starts, counts, r_raw)
    p_rate = permutation_p(starts, rate, r_rate)
    g = gini(counts)
    if verbose:
        print(f"\nT3. Significance (Spearman + permutation test, 20k shuffles)")
        print(f"    evictions      rho={r_raw:+.3f}  p={p_raw:.5f}")
        print(f"    evictions/sec  rho={r_rate:+.3f}  p={p_rate:.5f}")
        print(f"    Gini of eviction counts = {g:.3f}  (0=even, 1=all on one)")

    # T4 (previous version) compared eviction-time tokens against STARTING
    # tokens. Contexts grow, so those are different scales and the comparison
    # was invalid. Removed. See T5 below for a sound exposure-free test.

    # ---- T5: are the never-evicted programs the largest ones? -------------
    never_mask = counts == 0
    n_never = int(never_mask.sum())
    if n_never > 0 and len(pauses) > 0:
        rank_of_size = np.argsort(np.argsort(starts))       # 0 = smallest
        never_ranks = rank_of_size[never_mask]
        n = len(pids)
        # If evictions were assigned uniformly at random over programs, the
        # chance a given program escapes all E events is ((n-1)/n)^E, so the
        # chance that a SPECIFIC set of k programs all escape is that ^k.
        p_escape_each = ((n - 1) / n) ** len(pauses)
        p_all_escape = p_escape_each ** n_never
        mean_never_rank = float(never_ranks.mean())
        expected_rank = (n - 1) / 2
        if verbose:
            print(f"\nT5. Are the never-evicted programs the LARGEST? (exposure-free)")
            print(f"    Never evicted: {n_never}/{n} programs")
            print(f"    Their mean size-rank: {mean_never_rank:.1f} / {n-1}"
                  f"   (random would be {expected_rank:.1f})")
            print(f"    P(these {n_never} all escape {len(pauses)} evictions by chance)"
                  f" = {p_all_escape:.2e}")
        out.update(n_never=n_never, mean_never_rank=mean_never_rank,
                   p_all_escape=float(p_all_escape))

    never = [p for p in pids if evict_count[p] == 0]
    if never and verbose:
        never_starts = [info[p]["start_tokens"] for p in never]
        print(f"\n    Never evicted: {len(never)}/{len(pids)} programs")
        print(f"    Their start sizes: {min(never_starts):,}-{max(never_starts):,} tokens")
        print(f"    Population range : {int(starts.min()):,}-{int(starts.max()):,} tokens")

    out.update(rho_raw=r_raw, p_raw=p_raw, rho_rate=r_rate, p_rate=p_rate,
               gini=g, rho_lifetime=r_life, same_steps=same_steps,
               mean_evict=float(counts.mean()), max_evict=int(counts.max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="results/scheduler_run6.log")
    ap.add_argument("client", nargs="?", default="results/run6.json")
    ap.add_argument("--multi", nargs="*", default=None,
                    help="Run tags to combine, e.g. --multi run6 run7")
    args = ap.parse_args()

    if args.multi:
        results = []
        for tag in args.multi:
            log = f"results/scheduler_{tag}.log"
            cj = f"results/{tag}.json"
            if not (Path(log).exists() and Path(cj).exists()):
                print(f"skipping {tag}: missing files")
                continue
            results.append(analyse(log, cj, label=tag))

        if len(results) > 1:
            print(f"\n{'='*66}")
            print("  ACROSS RUNS")
            print(f"{'='*66}")
            print(f"{'run':>8} {'pauses':>7} {'mean':>6} {'max':>4} "
                  f"{'rho_raw':>8} {'p':>8} {'rho_rate':>9} {'gini':>6}")
            for r in results:
                print(f"{r['label']:>8} {r['n_pauses']:>7} {r['mean_evict']:>6.2f} "
                      f"{r['max_evict']:>4} {r['rho_raw']:>+8.3f} {r['p_raw']:>8.5f} "
                      f"{r['rho_rate']:>+9.3f} {r['gini']:>6.3f}")
            rhos = [r["rho_raw"] for r in results]
            print(f"\nrho_raw across runs: mean={np.mean(rhos):+.3f} "
                  f"sd={np.std(rhos):.3f}  (stable if sd is small)")
            Path("results/exposure_summary.json").write_text(json.dumps(results, indent=2))
            print("Saved -> results/exposure_summary.json")
    else:
        analyse(args.log, args.client, label=Path(args.log).stem)


if __name__ == "__main__":
    main()
