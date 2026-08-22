"""
GRPO-shaped synthetic workload for ThunderAgent.

Real RL post-training (GRPO/DAPO/GSPO) generates G rollouts per prompt.
Advantages cannot be computed until ALL G members of a group finish, so the
unit that matters is GROUP completion time, not per-trajectory time.

ThunderAgent's scheduler has no concept of groups -- every trajectory is an
independent "program". This client creates the group structure so we can ask:
does group-blind eviction delay groups that were nearly finished?

Structural detail that matters: members of one GRPO group share a prompt, so
their context lengths are SIMILAR. Shortest-first eviction therefore does not
spread within a group -- it targets whole groups. We model that with small
within-group jitter.

program_id format: "g{group:02d}t{member}"  -- group is parseable from the id.
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

TA_URL = "http://localhost:9000/v1/chat/completions"
RELEASE_URL = "http://localhost:9000/programs/release"
MODEL = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_WORDS = int(6500 / 1.3)

FILLER = (
    "Consider the following repository issue and reason carefully about the "
    "fix. The failing test exercises a code path that the scheduler must "
    "handle under memory pressure and uncertain future context growth. "
)


def make_padding(target_tokens: int) -> str:
    words_needed = max(int(target_tokens / 1.3), 1)
    fw = FILLER.split()
    reps = (words_needed // len(fw)) + 1
    return " ".join((fw * reps)[:words_needed])


async def run_trajectory(client, gid, mid, start_tokens, n_steps, growth,
                         tool_range, max_tokens, events, stagger_s):
    if stagger_s:
        await asyncio.sleep(stagger_s)

    program_id = f"g{gid:02d}t{mid}"
    context = make_padding(start_tokens)
    t_start = time.time()

    for step in range(n_steps):
        t0 = time.time()
        ok = False
        try:
            r = await client.post(TA_URL, json={
                "model": MODEL,
                "messages": [{"role": "user",
                              "content": context + "\nAnswer in one sentence."}],
                "max_tokens": max_tokens, "temperature": 0.0,
                "program_id": program_id,
            }, timeout=900.0)
            ok = r.status_code == 200
        except Exception as e:
            print(f"  {program_id} step {step}: {type(e).__name__}")

        events.append({
            "program_id": program_id, "group": gid, "member": mid,
            "step": step, "start_tokens": start_tokens,
            "latency_s": time.time() - t0, "ok": ok, "wallclock": time.time(),
        })

        if len(context.split()) < MAX_CONTEXT_WORDS:
            context += " " + make_padding(growth)
        await asyncio.sleep(random.uniform(*tool_range))

    try:
        await client.post(RELEASE_URL, json={"program_id": program_id}, timeout=30.0)
    except Exception:
        pass

    return {"program_id": program_id, "group": gid,
            "finish_wallclock": time.time(), "duration_s": time.time() - t_start}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-groups", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=8)
    ap.add_argument("--min-start-tokens", type=int, default=1500)
    ap.add_argument("--max-start-tokens", type=int, default=8000)
    ap.add_argument("--within-group-jitter", type=float, default=0.05)
    ap.add_argument("--growth-tokens", type=int, default=400)
    ap.add_argument("--tool-time-min", type=float, default=4.0)
    ap.add_argument("--tool-time-max", type=float, default=12.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--stagger-s", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", type=str, default="results/grpo_run.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)

    n_g = args.n_groups
    base = [int(args.min_start_tokens
                + (args.max_start_tokens - args.min_start_tokens) * i / max(n_g - 1, 1))
            for i in range(n_g)]
    rng.shuffle(base)

    tasks_meta = []
    for g in range(n_g):
        for m in range(args.group_size):
            jit = 1.0 + rng.uniform(-args.within_group_jitter, args.within_group_jitter)
            tasks_meta.append((g, m, max(int(base[g] * jit), 500)))

    total = len(tasks_meta)
    print(f"GRPO-shaped workload: {n_g} groups x {args.group_size} rollouts "
          f"= {total} programs")
    print(f"Group base sizes: {sorted(base)}")
    print(f"Within-group jitter: +/-{args.within_group_jitter*100:.0f}%")
    print(f"A group completes only when ALL {args.group_size} members finish.\n")

    events = []
    t0 = time.time()
    async with httpx.AsyncClient(timeout=900.0) as client:
        results = await asyncio.gather(*[
            run_trajectory(client, g, m, st, args.n_steps, args.growth_tokens,
                           (args.tool_time_min, args.tool_time_max),
                           args.max_tokens, events, i * args.stagger_s)
            for i, (g, m, st) in enumerate(tasks_meta)
        ])
    elapsed = time.time() - t0
    ok = sum(1 for e in events if e["ok"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "config": vars(args),
        "group_base_tokens": base,
        "elapsed_s": elapsed,
        "trajectories": results,
        "events": events,
    }, indent=2))

    print(f"\nDone in {elapsed:.0f}s | {ok}/{len(events)} requests OK")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
