"""Diagnose why completion_top_k_token_ids is None on TrainingSamples
reaching the Advantage Server. Run on the compute node with the canary's
inference server up.

Usage:
    cd /home/3312841/prime-rl
    uv run python scripts/canary/diag_top_k.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys


async def main() -> None:
    from openai import AsyncOpenAI
    from verifiers.clients.openai_chat_completions_client import (
        _extract_top_k_token_ids_from_logprobs_content,
    )

    snap = os.environ.get("SNAP")
    if not snap:
        print("ERROR: SNAP env var not set; source $ENV_FILE first", file=sys.stderr)
        sys.exit(1)

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    client = AsyncOpenAI(base_url=base_url, api_key="dummy")

    extra_body = {
        # Mirror prime_rl/orchestrator/utils.py: when return_top_k_token_ids
        # is on, these three are set together.
        "logprobs": True,
        "top_logprobs": 32,
        "return_tokens_as_token_ids": True,
        # vLLM extension to also include token_ids on the choice (so we
        # can compare lengths to top_logprobs.content).
        "return_token_ids": True,
        # vLLM-only sampling tweaks the orchestrator sets.
        "top_k": -1,
        "min_p": 0.0,
    }

    print(f"--> POST {base_url}/chat/completions")
    print(f"    extra_body keys: {list(extra_body.keys())}")
    print()

    response = await client.chat.completions.create(
        model=snap,
        messages=[{"role": "user", "content": "Reverse this: hello world"}],
        max_completion_tokens=20,
        logprobs=True,
        top_logprobs=32,
        extra_body={
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "top_k": -1,
            "min_p": 0.0,
        },
    )

    print("=== response shape ===")
    choice = response.choices[0]
    print(f"finish_reason: {choice.finish_reason}")
    print(f"message.content (first 60 chars): {choice.message.content[:60]!r}")

    # Check the vLLM `token_ids` extension on the choice.
    token_ids = getattr(choice, "token_ids", None)
    print(f"choice.token_ids present: {token_ids is not None}")
    if token_ids is not None:
        print(f"choice.token_ids length: {len(token_ids)}; first 5: {token_ids[:5]}")

    logprobs = choice.logprobs
    if logprobs is None or logprobs.content is None:
        print("ERROR: logprobs / logprobs.content is None; vLLM did not return logprobs")
        print("  -> request reached vLLM but logprobs disabled at server, OR")
        print("  -> max_logprobs cap rejected our top_logprobs=32 (check server log)")
        return

    print(f"logprobs.content length: {len(logprobs.content)}")
    if len(logprobs.content) > 0:
        first = logprobs.content[0]
        print(f"first content[0]: token={first.token!r}, logprob={first.logprob}, "
              f"token_id={getattr(first, 'token_id', None)}")
        if first.top_logprobs:
            tl0 = first.top_logprobs[0]
            print(f"first content[0].top_logprobs[0]: token={tl0.token!r}, "
                  f"logprob={tl0.logprob}, token_id={getattr(tl0, 'token_id', None)}")
            print(f"top_logprobs[0] count: {len(first.top_logprobs)}")
        else:
            print("WARNING: first content[0].top_logprobs is empty -- request did not propagate")

    # Now run the verifiers extractor.
    print()
    print("=== extractor result ===")
    # The extractor's `is_dict` flag: when accessed via OpenAI Pydantic models,
    # entries are objects, not dicts -- so is_dict=False is correct here.
    result = _extract_top_k_token_ids_from_logprobs_content(
        logprobs.content, is_dict=False
    )
    if result is None:
        print("RESULT: None  <--- this is the bug; extractor did not get token IDs")
        print("  Probable cause: top_logprobs items have neither .token_id nor a")
        print("  'token_id:N'-encoded .token. Inspect first content[0].top_logprobs[0] above.")
    else:
        print(f"RESULT: list of {len(result)} positions, "
              f"first row length {len(result[0]) if result else 0}")
        if result:
            print(f"  first row (first 5 IDs): {result[0][:5]}")

        # Length-mismatch check (this is what chat_completions_client.py
        # uses to silently drop top-K).
        if token_ids is not None:
            if len(result) != len(token_ids):
                print(f"  LENGTH MISMATCH: top_k extracted {len(result)}, "
                      f"but choice.token_ids has {len(token_ids)}.")
                print("  This is what chat_completions_client.py:570-575 silently catches")
                print("  by setting completion_top_k_token_ids = None.")
            else:
                print(f"  lengths match ({len(result)} == {len(token_ids)}); top-K should reach orchestrator")


if __name__ == "__main__":
    asyncio.run(main())
