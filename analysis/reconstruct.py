"""
Per-event reconstruction of ThunderAgent's eviction decisions.

Earlier tests showed short programs are evicted more OVERALL. This asks the
sharper question, one eviction at a time:

    At the moment of each eviction, was the victim the SMALLEST program
    among those currently live?

Method
------
The scheduler log (timestamped) tells us WHEN each pause/resume happened and
the victim's exact token count. The client JSON tells us when each program
started, finished, and completed each step. Combining them we can, for any
instant t, list the live programs and estimate each one's context size:

    size(p, t) = start_tokens[p] + steps_completed_before_t * per_step_growth

per_step_growth is calibrated from the log itself: we compare our estimate
for each VICTIM against the exact token count the scheduler logged, and
report the error. If the estimator is accurate for victims, we trust it for
the others.

Validated on synthetic data: recovers 100% when eviction truly is
shortest-first, and ~50th percentile when eviction is random.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TS_RE = re.compile(r"^(\d+\.\d+)\s+(.*)$")
PAUSE_RE = re.compile(r"paused (?:ACTING|REASONING) program (\S+) \(tokens=(\d+)\)")
MARK_RE = re.compile(r"marked (?:ACTING|REASONING) program (\S+) for pause \(tokens=(\d+)\)")
RESUME_RE = re.compile(r"Resumed program (\S+) to .*tokens=(\d+)")


def parse_timestamped_log(path):
    events = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = TS_RE.match(line)
        if not m:
            continue
        ts, rest = float(m.group(1)), m.group(2)
        mm = PAUSE_RE.search(rest) or MARK_RE.search(rest)
        if mm:
            events.append({"t": ts, "kind": "pause",
                           "pid": mm.group(1), "tokens": int(mm.group(2))})
            continue
        mm = RESUME_RE.search(rest)
        if mm:
            events.append({"t": ts, "kind": "resume",
                           "pid": mm.group(1), "tokens": int(mm.group(2))})
    events.sort(key=lambda e: e["t"])
    return events


def load_client(path):
    data = json.loads(Path(path).read_text())
    by_prog = defaultdict(list)
    for e in data["events"]:
        by_prog[e["program_id"]].append(e)
    for evs in by_prog.values():
        evs.sort(key=lambda e: e["wallclock"])
    return by_prog, data.get("config", {})


def steps_done_before(evs, t):
    n = 0
    for e in evs:
        if e["wallclock"] + e["latency_s"] <= t:
            n += 1
        else:
            break
    return n


def is_live(evs, t):
    start = evs[0]["wallclock"]
    end = evs[-1]["wallclock"] + evs[-1]["latency_s"]
    return start <= t <= end


def is_acting(evs, t):
    """ThunderAgent only evicts programs in ACTING state -- between the END of
    one request and the START of the next. A paused program shows up as a long
    ACTING gap, which is correct: it IS pause-eligible.

    Ranking a victim against all LIVE programs (including REASONING ones, which
    are NOT eviction candidates) scrambles ranks into a fake uniform
    distribution. That was a bug in the first version of this script."""
    for i, e in enumerate(evs):
        if i + 1 >= len(evs):
            continue
        done = e["wallclock"] + e["latency_s"]
        if done <= t < evs[i + 1]["wallclock"]:
            return True
    return False


def estimate_size(evs, t, per_step_growth):
    return evs[0]["start_tokens"] + steps_done_before(evs, t) * per_step_growth


def calibrate(log_events, by_prog, candidates):
    best, best_err = None, float("inf")
    for g in candidates:
        errs = []
        for ev in log_events:
            if ev["kind"] != "pause" or ev["pid"] not in by_prog:
                continue
            est = estimate_size(by_prog[ev["pid"]], ev["t"], g)
            errs.append(abs(est - ev["tokens"]))
        if errs:
            mae = float(np.mean(errs))
            if mae < best_err:
                best, best_err = g, mae
    return best, best_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", default="results/scheduler_run8.log", nargs="?")
    ap.add_argument("client", default="results/run8.json", nargs="?")
    args = ap.parse_args()

    log_events = parse_timestamped_log(args.log)
    by_prog, cfg = load_client(args.client)

    pauses = [e for e in log_events if e["kind"] == "pause"]
    print(f"Timestamped pause events: {len(pauses)}")
    print(f"Programs in client log  : {len(by_prog)}")
    if not pauses:
        print("No timestamped pause events found -- check the log format.")
        return

    growth, mae = calibrate(log_events, by_prog, candidates=list(range(200, 2001, 50)))
    print(f"\nCalibrated per-step growth: {growth} tokens "
          f"(mean abs error on victims: {mae:,.0f} tokens)")
    if mae > 3000:
        print("  WARNING: estimator error is large; ranks below are unreliable.")

    ranks, n_live_list, was_smallest, detail = [], [], 0, []
    for ev in pauses:
        t = ev["t"]
        # Eligible set = ACTING programs only (see is_acting docstring)
        live = [(pid, estimate_size(evs, t, growth))
                for pid, evs in by_prog.items() if is_acting(evs, t)]
        if len(live) < 2 or ev["pid"] not in dict(live):
            continue
        live.sort(key=lambda x: x[1])
        r = [pid for pid, _ in live].index(ev["pid"])
        ranks.append(r)
        n_live_list.append(len(live))
        if r == 0:
            was_smallest += 1
        detail.append({"t": t, "pid": ev["pid"], "rank": r,
                       "n_live": len(live), "logged_tokens": ev["tokens"]})

    if not ranks:
        print("Could not reconstruct any events (timing mismatch?).")
        return

    ranks = np.array(ranks)
    n_live = np.array(n_live_list)
    pct = ranks / np.maximum(n_live - 1, 1)

    print(f"\nReconstructed {len(ranks)} eviction decisions")
    print(f"  Mean ACTING (eligible) at eviction: {n_live.mean():.1f}")
    print(f"  Victim was SMALLEST eligible   : {was_smallest}/{len(ranks)} "
          f"({was_smallest/len(ranks)*100:.0f}%)")
    print(f"  Victim in smallest QUARTILE    : {(pct <= 0.25).sum()}/{len(ranks)} "
          f"({(pct<=0.25).mean()*100:.0f}%)")
    print(f"  Mean victim percentile         : {pct.mean()*100:.1f}% "
          f"(size-blind eviction would give ~50%)")

    print(f"\n  Victim rank histogram (0 = smallest ACTING program):")
    for r in range(0, min(8, int(ranks.max()) + 1)):
        c = int((ranks == r).sum())
        if c:
            print(f"    rank {r}: {'#' * c} ({c})")
    high = int((ranks >= 8).sum())
    if high:
        print(f"    rank 8+: {'#' * high} ({high})")

    Path("results/reconstruction_run8.json").write_text(json.dumps({
        "calibrated_growth": growth, "mae": mae,
        "n_reconstructed": len(ranks),
        "frac_smallest": float(was_smallest / len(ranks)),
        "mean_percentile": float(pct.mean()),
        "events": detail,
    }, indent=2))
    print("\nSaved -> results/reconstruction_run8.json")


if __name__ == "__main__":
    main()
