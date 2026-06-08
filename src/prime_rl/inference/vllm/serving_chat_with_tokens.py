import asyncio
import base64
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import ClassVar, Optional, Union

import numpy as np
from fastapi import Request
from pydantic import BaseModel, Field
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionResponse
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata
from vllm.entrypoints.openai.engine.serving import GenerationError
from vllm.entrypoints.utils import get_max_tokens
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.reasoning import ReasoningParser
from vllm.sampling_params import BeamSearchParams, SamplingParams

logger = init_logger(__name__)


class _RoutedExpertsCapture:
    def __init__(self, generator: AsyncGenerator[RequestOutput, None]):
        self._generator = generator
        self.routed_experts: dict[int, dict] = {}

    def _encode_routed_experts(self, arr: np.ndarray) -> dict:
        return {
            "data": base64.b85encode(arr.tobytes()).decode("ascii"),
            "shape": list(arr.shape),
        }

    async def __aiter__(self):
        async for request_output in self._generator:
            for output in request_output.outputs:
                if output.routed_experts is not None:
                    self.routed_experts[output.index] = self._encode_routed_experts(output.routed_experts)
            yield request_output

    def post_process(self, response: ChatCompletionResponse):
        for choice in response.choices:
            if choice.index in self.routed_experts:
                choice.routed_experts = self.routed_experts[choice.index]


class ChatCompletionRequestWithTokens(ChatCompletionRequest):
    field_names: ClassVar[Optional[set[str]]] = None
    tokens: list[int] = Field(description=("Prompt tokens to use for the request."))


class ScoreRequest(BaseModel):
    """Teacher-forcing scoring request for action-level ARM pi_hat.

    Each prompt is a full token-id sequence (e.g. [o, R_j, <action>a</action>]).
    For each prompt we return ONLY the summed logprob over its [start, end) span
    (the action suffix), not the full per-position prompt_logprobs -- the response
    is then one float per prompt instead of len(prompt) logprob dicts, which is the
    whole point: the full-prompt prompt_logprobs payload is cache-invariant and was
    the Phase 9 throughput wall.
    """

    model: str
    prompts: list[list[int]]
    spans: list[list[int]]  # [[start, end), ...] aligned 1:1 with prompts
    temperature: float = 1.0
    top_p: float = 1.0


def _sum_span_prompt_logprobs(prompt_logprobs, token_ids: list[int], start: int, end: int) -> float:
    """Sum the actual-token logprob over [start, end) from a RequestOutput's
    prompt_logprobs (list[Optional[dict[int, Logprob]]], keyed by int token-id at the
    engine level). Cached / missing positions contribute 0 (the scored span is the
    uncached action suffix, so its positions are always present)."""
    if not prompt_logprobs:
        return 0.0
    n = len(prompt_logprobs)
    total = 0.0
    for p in range(start, min(end, n)):
        entry = prompt_logprobs[p]
        if not entry:
            continue
        lp = entry.get(token_ids[p])
        if lp is None:
            continue
        total += lp.logprob if hasattr(lp, "logprob") else float(lp)
    return total


class OpenAIServingChatWithTokens(OpenAIServingChat):
    """OpenAI-compatible generate API that allows token-in and routed experts capture."""

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation,
        tokenizer,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        # We need to override the full_generator to be able to capture the routed experts
        # By default, VLLM does not save the routed experts into ChatCompletionResponse.choices, so we need to capture them manually
        # How this works:
        # 1. We create a custom generator that encapsulates the original result_generator in self._generator
        # 2. We override it's __aiter__ method to also capture the routed experts as an extra field in ChatCompletionResponse.choices
        # 3. We override the full_generator method to use the custom generator instead of the original one if expert routing is enabled
        if self.model_config.enable_return_routed_experts:
            capture = _RoutedExpertsCapture(result_generator)
            result_generator = capture
        else:
            capture = None

        response = await super().chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            reasoning_parser,
        )

        if capture and isinstance(response, ChatCompletionResponse):
            capture.post_process(response)

        return response

    async def create_chat_completion_with_tokens(
        self,
        request: ChatCompletionRequestWithTokens,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ChatCompletionResponse, ErrorResponse]:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        # Streaming response
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        reasoning_parser: ReasoningParser | None = None
        try:
            if self.reasoning_parser_cls:
                # Pass the same chat template kwargs as used in tokenization
                chat_template_kwargs = self._prepare_extra_chat_template_kwargs(
                    request.chat_template_kwargs,
                    self.default_chat_template_kwargs,
                )
                reasoning_parser = self.reasoning_parser_cls(
                    tokenizer,
                    chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
                )
        except RuntimeError as e:
            logger.exception("Error in reasoning parser creation.")
            return self.create_error_response(str(e))
        result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_prompts = result

        # We override prompt tokens directly.
        # VLM conversations use MITO (message-based) instead of TITO, so
        # multi_modal_data is not expected here.
        engine_prompts[0]["prompt_token_ids"] = request.tokens  # type: ignore

        request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

            model_name = self.models.model_name(lora_request)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.exception("Error preparing request components")
            return self.create_error_response(e)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                prompt_token_ids = self._extract_prompt_components(engine_prompt).token_ids

                # If we are creating sub requests for multiple prompts, ensure that they
                # have unique request ids.
                sub_request_id = request_id if len(engine_prompts) == 1 else f"{request_id}_{i}"

                prompt_len = self._extract_prompt_len(engine_prompt)
                if prompt_len >= max_model_len:
                    raise VLLMValidationError(
                        f"This model's maximum context length is "
                        f"{max_model_len} tokens. However, your request has "
                        f"{prompt_len} input tokens. Please reduce the length of "
                        "the input messages.",
                        parameter="input_tokens",
                        value=prompt_len,
                    )

                max_tokens = get_max_tokens(
                    max_model_len,
                    request.max_completion_tokens if request.max_completion_tokens is not None else request.max_tokens,
                    self._extract_prompt_len(engine_prompt),
                    self.default_sampling_params,
                    self.override_max_tokens,
                )

                sampling_params: SamplingParams | BeamSearchParams
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens,
                        self.default_sampling_params,
                    )

                self._log_inputs(
                    sub_request_id,
                    engine_prompt,
                    params=sampling_params,
                    lora_request=lora_request,
                )

                trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)

                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.beam_search(
                        prompt=engine_prompt,
                        request_id=sub_request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                    )
                else:
                    reasoning_ended = (
                        reasoning_parser.is_reasoning_end(prompt_token_ids or []) if reasoning_parser else None
                    )

                    generator = self.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        sub_request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                        data_parallel_rank=data_parallel_rank,
                        reasoning_ended=reasoning_ended,
                    )

                generators.append(generator)
        except ValueError as e:
            return self.create_error_response(e)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
            )

        try:
            return await self.chat_completion_full_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
            )
        except GenerationError as e:
            return self._convert_generation_error_to_response(e)
        except ValueError as e:
            return self.create_error_response(e)

    async def create_score(self, request: "ScoreRequest", raw_request: Optional[Request] = None):
        """Teacher-force-score token-id prompts and return ONLY each prompt's summed
        span logprob (action-level ARM pi_hat).

        Runs one engine request per prompt (max_tokens=1, prompt_logprobs=1),
        concurrently so vLLM continuous-batches them and prefix caching reuses the
        shared [o, R_j] prefix; then sums each prompt's prompt_logprobs over its
        [start, end) action span. Returns {"scores": [float, ...]} -- a tiny payload
        vs the full per-position prompt_logprobs the env would otherwise pull over
        /v1/completions. With the prefix cached, vLLM only computes logits for the
        uncached suffix, so this cuts BOTH the response size and the compute.
        """
        try:
            lora_request = self._maybe_get_adapters(request)
        except (ValueError, TypeError, RuntimeError) as e:
            return self.create_error_response(str(e))

        prompts = request.prompts
        spans = request.spans
        if len(prompts) != len(spans):
            return self.create_error_response(
                f"prompts ({len(prompts)}) and spans ({len(spans)}) length mismatch"
            )

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=1,
            prompt_logprobs=1,
        )
        base_request_id = f"score-{uuid.uuid4().hex}"

        async def _run(idx: int, token_ids: list[int]):
            gen = self.engine_client.generate(
                {"prompt_token_ids": list(token_ids)},
                sampling_params,
                f"{base_request_id}-{idx}",
                lora_request=lora_request,
            )
            final = None
            async for out in gen:
                final = out
            return idx, final

        finals = await asyncio.gather(*[_run(i, tk) for i, tk in enumerate(prompts)])

        scores: list[float] = [0.0] * len(prompts)
        for idx, final in finals:
            prompt_logprobs = getattr(final, "prompt_logprobs", None) if final is not None else None
            start, end = int(spans[idx][0]), int(spans[idx][1])
            scores[idx] = _sum_span_prompt_logprobs(prompt_logprobs, prompts[idx], start, end)

        return {"scores": scores}
