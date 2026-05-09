"""Diagnose top-K extraction via the orchestrator-shaped path:

  prime_rl.orchestrator.utils.get_sampling_args (build extra_body)
    -> verifiers.clients.openai_chat_completions_client.OpenAIChatCompletionsClient
        -> AsyncOpenAI client
            -> vLLM /v1/chat/completions

Compare this output against diag_top_k.py (which uses raw AsyncOpenAI).
If diag_top_k.py works but THIS one fails, the bug is in verifiers'
chat-client wrapper -- somewhere extra_body or return_tokens_as_token_ids
is getting stripped.

Usage (same as diag_top_k.py):
    cd /home/3312841/prime-rl
    source /home/3312841/prime-rl/.canary_env
    uv run python scripts/canary/diag_top_k_via_verifiers.py
"""

from __future__ import annotations

import asyncio
import os
import sys


async def main() -> None:
    snap = os.environ.get("SNAP")
    if not snap:
        print("ERROR: SNAP env var not set; source $ENV_FILE first", file=sys.stderr)
        sys.exit(1)

    from prime_rl.configs.orchestrator import SamplingConfig
    from prime_rl.orchestrator.utils import get_sampling_args
    from verifiers.clients.openai_chat_completions_client import (
        OpenAIChatCompletionsClient,
        _extract_top_k_token_ids_from_logprobs_content,
    )

    # 1. Build sampling_args via the orchestrator's actual function.
    sampling_config = SamplingConfig(
        max_tokens=128,
        return_top_k_token_ids=True,
        top_k_action_set_size=32,
    )
    sampling_args = get_sampling_args(
        sampling_config, temperature=1.0, is_vllm=True
    )
    print("=== sampling_args from prime_rl.orchestrator.utils.get_sampling_args ===")
    for k, v in sampling_args.items():
        print(f"  {k}: {v!r}")
    print()

    extra = sampling_args.get("extra_body", {})
    has_rt = extra.get("return_tokens_as_token_ids")
    has_tl = extra.get("top_logprobs")
    print(f"  extra_body['return_tokens_as_token_ids']: {has_rt!r}")
    print(f"  extra_body['top_logprobs']: {has_tl!r}")
    print()
    if has_rt is not True:
        print("--> ROOT CAUSE: orchestrator is NOT setting return_tokens_as_token_ids=True.")
        sys.exit(2)

    # 2. Construct verifiers' chat client via its standard ctor (calls
    # setup_client(config) internally to build the AsyncOpenAI). vLLM
    # ignores api_key but openai-python requires SOME value to be set.
    print("=== constructing verifiers OpenAIChatCompletionsClient ===")
    base = "http://localhost:8000/v1"
    os.environ.setdefault("VLLM_API_KEY", "dummy")

    from verifiers.types import ClientConfig as VFClientConfig
    vf_config = VFClientConfig(
        api_base_url=base,
        api_key_var="VLLM_API_KEY",
        timeout=600,
        connect_timeout=30,
    )
    client = OpenAIChatCompletionsClient(vf_config)
    print(f"  client.client type: {type(client.client).__name__}")
    print(f"  base: {base}")

    # 3. Fire the request via the verifiers client method -- same as the env does.
    print()
    print("=== firing request via OpenAIChatCompletionsClient.get_native_response ===")
    response = await client.get_native_response(
        prompt=[{"role": "user", "content": "Reverse: hi"}],
        model=snap,
        sampling_args=sampling_args,
    )

    choice = response.choices[0]
    print(f"finish_reason: {choice.finish_reason}")
    print(f"message.content (first 60 chars): {(choice.message.content or '')[:60]!r}")
    token_ids = getattr(choice, "token_ids", None)
    print(f"choice.token_ids present: {token_ids is not None}; "
          f"length: {len(token_ids) if token_ids is not None else 'N/A'}")

    if choice.logprobs is None or choice.logprobs.content is None:
        print("ERROR: logprobs.content is None even though we asked for it")
        sys.exit(3)

    content_list = choice.logprobs.content
    print(f"logprobs.content length: {len(content_list)}")
    if not content_list:
        print("WARNING: logprobs.content empty")
        return

    first = content_list[0]
    print(f"first content[0]: token={first.token!r}, logprob={first.logprob}")
    if first.top_logprobs:
        tl0 = first.top_logprobs[0]
        print(f"first content[0].top_logprobs[0]: token={tl0.token!r}, logprob={tl0.logprob}")
        print(f"top_logprobs[0] count: {len(first.top_logprobs)}")

        if isinstance(tl0.token, str) and tl0.token.startswith("token_id:"):
            print()
            print("--> return_tokens_as_token_ids REACHED vLLM (token_id:N encoding present).")
        else:
            print()
            print("--> return_tokens_as_token_ids did NOT reach vLLM.")
            print("    Token field is plain text, not 'token_id:N'.")
    else:
        print("first content[0].top_logprobs is empty")

    # 4. Run the extractor too and check len match.
    print()
    print("=== extractor result ===")
    result = _extract_top_k_token_ids_from_logprobs_content(content_list, is_dict=False)
    if result is None:
        print("RESULT: None")
    else:
        print(f"RESULT: list of {len(result)} positions, "
              f"first row length {len(result[0]) if result else 0}")
        if token_ids is not None and len(result) != len(token_ids):
            print(f"  LENGTH MISMATCH: top_k {len(result)} != token_ids {len(token_ids)} "
                  "(silently dropped at chat_completions_client.py:570-575)")
        else:
            print(f"  lengths match (top_k {len(result)} == token_ids {len(token_ids) if token_ids else '?'})")




async def main_from_native() -> None:
    """Extension: also test from_native_response (the verifiers method that
    wraps a ChatCompletionResponse into a vf.Response with tokens populated).
    This is what parse_response_tokens (used by MultiTurnEnv.add_model_response,
    which SingleTurnEnv inherits) ultimately reads."""
    snap = os.environ.get("SNAP")
    if not snap:
        return

    from prime_rl.configs.orchestrator import SamplingConfig
    from prime_rl.orchestrator.utils import get_sampling_args
    from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
    from verifiers.types import ClientConfig as VFClientConfig

    sampling_args = get_sampling_args(
        SamplingConfig(max_tokens=128, return_top_k_token_ids=True, top_k_action_set_size=32),
        temperature=1.0, is_vllm=True,
    )
    os.environ.setdefault("VLLM_API_KEY", "dummy")
    client = OpenAIChatCompletionsClient(VFClientConfig(
        api_base_url="http://localhost:8000/v1", api_key_var="VLLM_API_KEY",
        timeout=600, connect_timeout=30,
    ))

    raw = await client.get_native_response(
        prompt=[{"role": "user", "content": "Reverse: hi"}],
        model=snap, sampling_args=sampling_args,
    )

    print()
    print("=== from_native_response wrapping ===")
    has_pti = hasattr(raw, "prompt_token_ids")
    print(f"raw response has prompt_token_ids attr: {has_pti}")
    if has_pti:
        pti = raw.prompt_token_ids
        print(f"  prompt_token_ids type: {type(pti).__name__}, value None? {pti is None}, length: {len(pti) if pti else 'N/A'}")

    vf_response = await client.from_native_response(raw)
    msg = vf_response.message
    print(f"vf_response.message.tokens is None: {msg.tokens is None}")
    if msg.tokens is None:
        print("--> ROOT CAUSE: parse_tokens returned None inside from_native_response.")
        print("    Most likely: vLLM didn\'t set 'prompt_token_ids' on the response,")
        print("    so parse_tokens (line 508 of chat client) bailed early.")
    else:
        ttk = msg.tokens.completion_top_k_token_ids
        print(f"  completion_top_k_token_ids is None: {ttk is None}")
        if ttk is not None:
            print(f"  shape: {len(ttk)} positions x {len(ttk[0]) if ttk else 0}")
            print("--> verifiers Response.message.tokens HAS top-K populated.")
            print("    The bug is downstream of from_native_response.")
        else:
            print("--> top-K extraction inside parse_tokens failed.")
            print("    Likely: extractor returned non-None but length-mismatch fired,")
            print("    OR extractor returned None (extractor saw token_id-less top_logprobs).")


if __name__ == "__main__" and "--ext" in sys.argv:
    asyncio.run(main_from_native())
elif __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main_from_native())
