"""
Synthetic agentic workload for measuring eviction churn in ThunderAgent.

Each "program" imitates an agent: it alternates between
  REASONING  - an LLM call (context grows each turn)
  ACTING     - a simulated tool call (just a sleep, no LLM)
and repeats for a fixed number of steps.

Programs are deliberately given DIFFERENT starting context sizes so we can
ask: does ThunderAgent's shortest-first eviction policy keep evicting the
same (short) programs over and over?

Nothing here patches ThunderAgent -- we only drive it and read its logs.
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

THUNDERAGENT_URL = "http://localhost:9000/v1/chat/completions"
RELEASE_URL = "http://localhost:9000/programs/release"
MODEL = "Qwen/Qwen3-0.6B"
# vLLM runs with --max-model-len 8192; stay well under it (words ~= tokens/1.3)
MAX_CONTEXT_WORDS = int(6500 / 1.3)

FILLER = (
    "Consider the following system log entry and reason about it carefully. "
    "The scheduler observed memory pressure across backends and had to make "
    "a placement decision under uncertainty about future context growth. "
)


def make_padding(target_tokens: int) -> str:
    """~1.3 tokens per word for this filler, so scale words accordingly."""
    words_needed = int(target_tokens / 1.3)
    filler_words = FILLER.split()
    reps = (words_needed // len(filler_words)) + 1
    return " ".join((filler_words * reps)[:words_needed])


async def run_program(client, pid, start_tokens, n_steps, growth_tokens,
                      tool_time_range, events, sem, stagger_s=0.0,
                      max_tokens=24):
    """One agentic program: reason -> act -> reason -> act ..."""
    if stagger_s:
        await asyncio.sleep(stagger_s)
    context = make_padding(start_tokens)
    program_id = f"prog-{pid:03d}"

    for step in range(n_steps):
        t0 = time.time()
        try:
            async with sem:
                resp = await client.post(
                    THUNDERAGENT_URL,
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content":
                                      context + "\nAnswer in one short sentence."}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "program_id": program_id,
                    },
                    timeout=600.0,
                )
            ok = resp.status_code == 200
        except Exception as e:
            ok = False
            print(f"  {program_id} step {step} error: {type(e).__name__}")

        events.append({
            "program_id": program_id,
            "step": step,
            "start_tokens": start_tokens,
            "latency_s": time.time() - t0,
            "ok": ok,
            "wallclock": time.time(),
        })

        # Context grows like a real agent accumulating history, but we cap
        # it below vLLM's --max-model-len; otherwise requests 400 out.
        if len(context.split()) < MAX_CONTEXT_WORDS:
            context += " " + make_padding(growth_tokens)

        # ACTING phase: simulated tool call, no LLM involved
        await asyncio.sleep(random.uniform(*tool_time_range))

    try:
        await client.post(RELEASE_URL, json={"program_id": program_id}, timeout=30.0)
    except Exception as e:
        print(f"  {program_id} release failed: {type(e).__name__}")
    print(f"  {program_id} finished ({n_steps} steps, start={start_tokens} tok)")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-programs", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=8)
    ap.add_argument("--min-start-tokens", type=int, default=1500)
    ap.add_argument("--max-start-tokens", type=int, default=8000)
    ap.add_argument("--growth-tokens", type=int, default=400)
    ap.add_argument("--tool-time-min", type=float, default=1.0)
    ap.add_argument("--tool-time-max", type=float, default=6.0)
    ap.add_argument("--max-inflight", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=24,
                    help="Output tokens per step; higher = longer GPU residency")
    ap.add_argument("--stagger-s", type=float, default=0.0,
                    help="Seconds between program launches (0 = all at once)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="results/client_run1.json")
    args = ap.parse_args()

    random.seed(args.seed)

    # Spread starting context sizes evenly across the range, so short and long
    # programs coexist -- that's the setup the churn question needs.
    starts = [
        int(args.min_start_tokens
            + (args.max_start_tokens - args.min_start_tokens) * i / max(args.n_programs - 1, 1))
        for i in range(args.n_programs)
    ]
    random.shuffle(starts)

    print(f"Launching {args.n_programs} programs x {args.n_steps} steps")
    print(f"Start context sizes: {min(starts)}..{max(starts)} tokens")
    print(f"Rough peak demand: ~{sum(s + args.growth_tokens*args.n_steps for s in starts):,} tokens "
          f"vs 43,216 capacity\n")

    events = []
    sem = asyncio.Semaphore(args.max_inflight)
    t_start = time.time()

    async with httpx.AsyncClient(timeout=600.0) as client:
        await asyncio.gather(*[
            run_program(client, i, starts[i], args.n_steps, args.growth_tokens,
                        (args.tool_time_min, args.tool_time_max), events, sem,
                        stagger_s=i * args.stagger_s, max_tokens=args.max_tokens)
            for i in range(args.n_programs)
        ])

    elapsed = time.time() - t_start
    ok_count = sum(1 for e in events if e["ok"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "config": vars(args),
        "elapsed_s": elapsed,
        "events": events,
    }, indent=2))

    print(f"\nDone in {elapsed:.0f}s | {ok_count}/{len(events)} requests OK")
    print(f"Throughput: {ok_count/elapsed*60:.1f} steps/min")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
