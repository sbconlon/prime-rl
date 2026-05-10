"""Synthetic-load profiling harness for the Advantage Server.

Brings up a standalone Advantage Server on the local node (no orchestrator,
no vLLM, no four-process cascade) with PRIME_RL_ADV_PROF=1, fires N
synthetic compute requests, then exits and prints the log path. The
companion analyzer (parse_prof_log.py) ingests the resulting log and
prints per-stage timing breakdowns.

Synthetic payload approximates the Phase 10 canary\'s actual shape:
  - Single-turn: 1 sample per request
  - prompt_len ~ 150 (typical reverse-text + system prompt)
  - completion_len = 128 (canary\'s max_tokens)
  - K = 32 (top_k_action_set_size)
  - completion_top_k_token_ids: random valid token IDs

Usage:
    cd /home/3312841/prime-rl
    source /home/3312841/prime-rl/.canary_env

    # Free GPU 2 first if anything is on it
    bash scripts/canary/canary.sh cleanup

    # Run profile (default: 10 requests).
    uv run python scripts/canary/profile_advantage_server.py \\
        --num-requests 10 --port 8200

    # Then analyze (path is printed at the end of this script):
    uv run python scripts/canary/parse_prof_log.py /tmp/adv_prof_run/server.log
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path


def build_synthetic_request(
    *,
    prompt_len: int,
    completion_len: int,
    K: int,
    vocab_size: int = 151643,
) -> bytes:
    """Construct a msgspec-encoded ComputeAdvantagesRequest with shape that
    approximates the Phase 10 canary\'s reverse-text rollouts."""
    import msgspec

    from prime_rl.transport.types import TrainingSample
    from prime_rl.orchestrator.advantage_server_client import (
        ComputeAdvantagesRequest,
    )

    rng = random.Random(0)

    prompt_ids = [rng.randint(0, vocab_size - 1) for _ in range(prompt_len)]
    completion_ids = [rng.randint(0, vocab_size - 1) for _ in range(completion_len)]
    completion_mask = [True] * completion_len

    # Phase 5 compact layout: outer length = sum(completion_mask).
    # Each row\'s sampled token must be present in the row -- ensure that
    # by placing it at index 0.
    completion_top_k_token_ids = []
    for tok in completion_ids:
        row = [tok] + [rng.randint(0, vocab_size - 1) for _ in range(K - 1)]
        completion_top_k_token_ids.append(row)

    sample = TrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * prompt_len,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=[0.0] * completion_len,
        completion_temperatures=[1.0] * completion_len,
        completion_top_k_token_ids=completion_top_k_token_ids,
    )
    request = ComputeAdvantagesRequest(
        samples=[sample],
        episodic_reward=1.0,
        is_terminal=True,
        algorithm="arm",
    )
    return msgspec.msgpack.encode(request)


async def fire_requests(
    *,
    num_requests: int,
    base_url: str,
    payload_bytes: bytes,
    log_per_request: bool,
) -> dict[str, float | int]:
    """Fire `num_requests` POSTs sequentially to /compute_advantages_and_targets.
    Sequential, not concurrent: we want to characterize per-request server
    cost, not concurrency behavior."""
    import httpx

    durations_ms = []
    failures = 0
    async with httpx.AsyncClient(timeout=600.0) as client:
        for i in range(num_requests):
            start = time.perf_counter()
            try:
                response = await client.post(
                    f"{base_url.rstrip('/')}/compute_advantages_and_targets",
                    content=payload_bytes,
                    headers={"Content-Type": "application/x-msgpack"},
                )
                response.raise_for_status()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                durations_ms.append(elapsed_ms)
                if log_per_request:
                    print(f"  request {i+1}/{num_requests}: {elapsed_ms:8.1f} ms  status={response.status_code}")
            except Exception as exc:
                failures += 1
                print(f"  request {i+1}/{num_requests}: FAILED -- {exc}")

    if not durations_ms:
        return {"failures": failures}

    durations_ms.sort()
    n = len(durations_ms)
    return {
        "n": n,
        "failures": failures,
        "mean_ms": sum(durations_ms) / n,
        "median_ms": durations_ms[n // 2],
        "p95_ms": durations_ms[max(0, int(0.95 * n) - 1)],
        "min_ms": durations_ms[0],
        "max_ms": durations_ms[-1],
        "total_s": sum(durations_ms) / 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--prompt-len", type=int, default=150)
    parser.add_argument("--completion-len", type=int, default=128)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/adv_prof_run"))
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Don\'t spawn the AdvSrv; assume one is already running at --port.",
    )
    args = parser.parse_args()

    snap = os.environ.get("SNAP")
    if not snap and not args.no_spawn:
        print("ERROR: SNAP env var not set; source .canary_env first.", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    server_log = args.out_dir / "server.log"
    server_proc: subprocess.Popen | None = None

    if not args.no_spawn:
        # Build the AdvantageServerConfig as a CLI override.
        # Use the SNAP path for model_name; LoRA defaults are fine for canary.
        # PRIME_RL_ADV_PROF=1 turns on the prof logging.
        env = {
            **os.environ,
            "PRIME_RL_ADV_PROF": "1",
            # Keep the AdvSrv off the GPUs the canary uses; pin to GPU 0 here
            # since this is a standalone run (whichever GPU you have free).
            "CUDA_VISIBLE_DEVICES": os.environ.get("PROFILE_ADV_CUDA", "0"),
        }
        cmd = [
            "uv", "run", "advantage-server",
            "--model-name", snap,
            "--host", "127.0.0.1",
            "--port", str(args.port),
        ]
        print(f"==> spawning AdvSrv: {' '.join(cmd)}")
        print(f"    log: {server_log}")
        with open(server_log, "w") as logf:
            server_proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)

        # Wait for /health to be 200.
        import httpx
        base = f"http://127.0.0.1:{args.port}"
        print("==> waiting for AdvSrv /health ...")
        for i in range(180):
            try:
                with httpx.Client(timeout=2.0) as client:
                    r = client.get(f"{base}/health")
                    if r.status_code == 200:
                        print(f"    ready after {i+1}s")
                        break
            except Exception:
                pass
            time.sleep(1)
            if server_proc.poll() is not None:
                print(f"==> AdvSrv died during startup (exit code {server_proc.returncode})")
                print(f"==> log tail:")
                print(server_log.read_text()[-4000:])
                sys.exit(1)
        else:
            print("==> AdvSrv did not become ready within 180s")
            server_proc.send_signal(signal.SIGTERM)
            sys.exit(1)

    base_url = f"http://127.0.0.1:{args.port}"

    print(f"==> building synthetic payload")
    print(f"    prompt_len={args.prompt_len} completion_len={args.completion_len} K={args.K}")
    payload = build_synthetic_request(
        prompt_len=args.prompt_len,
        completion_len=args.completion_len,
        K=args.K,
    )
    print(f"    payload bytes: {len(payload):,}")

    print(f"==> firing {args.num_requests} requests sequentially")
    summary = asyncio.run(
        fire_requests(
            num_requests=args.num_requests,
            base_url=base_url,
            payload_bytes=payload,
            log_per_request=True,
        )
    )

    print()
    print("=== client-side wall-clock summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>12s}: {v:8.1f}")
        else:
            print(f"  {k:>12s}: {v}")

    if server_proc is not None:
        print()
        print(f"==> stopping AdvSrv (PID {server_proc.pid})")
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()

    print()
    print(f"==> done. Server log (with PROF lines):")
    print(f"    {server_log}")
    print(f"==> Analyze with:")
    print(f"    uv run python scripts/canary/parse_prof_log.py {server_log}")


if __name__ == "__main__":
    main()
