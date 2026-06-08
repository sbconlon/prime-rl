"""Advantage Server HTTP entrypoint.

FastAPI app exposing two endpoints:
  GET  /health
       Returns 200 OK once the ValueNetworkBackbone has finished loading;
       503 SERVICE UNAVAILABLE before that. The orchestrator polls this
       endpoint at startup.
  POST /compute_advantages_and_targets
       Accepts a msgspec.msgpack-encoded ComputeAdvantagesRequest body,
       dispatches to advantage_server.compute, returns a msgspec.msgpack-
       encoded ComputeAdvantagesResponse body.

The compute work is GPU-bound and synchronous; we wrap it in
`asyncio.run_in_executor` so a slow request does not block the event loop.

Phase 6 = naive forward path (slow). Phase 6.5 layers KV-cache /
batched-K-candidate optimizations.
"""

from __future__ import annotations

import asyncio
import io

import torch
import logging

import msgspec
import uvicorn
from fastapi import FastAPI, Request, Response
from transformers import AutoModel

from prime_rl.advantage_server.compute import compute_advantages_and_targets
from prime_rl.advantage_server._prof import (
    new_request_id,
    prof,
    set_request_id,
)
from prime_rl.configs.advantage_server import AdvantageServerConfig
from prime_rl.orchestrator.advantage_server_client import (
    ComputeAdvantagesRequest,
    ComputeAdvantagesResponse,
    PairedSample,
)
from prime_rl.orchestrator.value_networks import ValueNetworkBackbone

_LOGGER = logging.getLogger("prime_rl.advantage_server")


def create_app(config: AdvantageServerConfig) -> FastAPI:
    """Build a FastAPI app for the Advantage Server.

    Loading the value-network backbone happens lazily on app startup via the
    lifespan handler. The /health endpoint reports 503 until load is complete.
    """
    app = FastAPI(title="Advantage Server")
    app.state.config = config
    app.state.backbone = None
    # Tokenizer for action-level ARM (tokenizing admissible actions for Q+).
    # Loaded lazily alongside the backbone; None for PPO/token-level runs.
    app.state.tokenizer = None
    app.state.ready = False
    app.state.gamma = config.gamma
    app.state.lam = config.lam
    app.state.n_step = config.n_step
    # CFR+ regret-accumulation decay for action-level ARM q_plus_target. <1 bounds
    # the regret accumulation (the run-001 collapse was effective phi_decay=1.0).
    app.state.arm_phi_decay = config.arm_phi_decay
    # Tracks how many weight broadcasts the AdvServer has received from the
    # AdvTrainer. Initialized to 0 (= initial zero-init backbone, no broadcast
    # yet). Incremented by /update_weights AFTER the state dict is loaded.
    # Surfaced via /weight_status so the orchestrator can gate its per-step
    # compute calls on `weight_step >= step` (enforces that orchestrator
    # step N's targets are computed under value weights post-AdvTrainer
    # training on batch N-1; closes the cross-step race in the
    # AdvTrainer<->orchestrator pipeline).
    app.state.weight_step = 0

    @app.on_event("startup")
    async def _load_backbone() -> None:
        cfg: AdvantageServerConfig = app.state.config
        # Phase 10 perf: serve on GPU in BF16 when available. Loading on CPU
        # forces the Q+ K-candidate kernel onto the FP32 Python attention
        # fallback (~31s/request); on CUDA+BF16 it uses flash_attn_with_kvcache.
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16
        else:
            device = torch.device("cpu")
            dtype = torch.float32
        _LOGGER.info(
            "Loading base model %s for ValueNetworkBackbone (device=%s, dtype=%s)",
            cfg.model_name, device, dtype,
        )
        base = AutoModel.from_pretrained(cfg.model_name, dtype=dtype)
        backbone = ValueNetworkBackbone(
            base, lora_config=cfg.lora, polyak_tau=cfg.polyak_tau
        )
        backbone = backbone.to(device=device, dtype=dtype)
        backbone.eval()
        app.state.backbone = backbone
        # Action-level ARM tokenizer (best-effort). PPO / token-level runs don't
        # need it, so a load failure is non-fatal here -- the action path raises a
        # clear error only if it is actually invoked without a tokenizer.
        try:
            from transformers import AutoTokenizer

            app.state.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        except Exception as exc:  # noqa: BLE001 -- non-fatal; PPO doesn't need it
            _LOGGER.warning(
                "Could not load tokenizer for %s; action-level ARM will fail if used: %s",
                cfg.model_name, exc,
            )
            app.state.tokenizer = None
        app.state.ready = True
        _LOGGER.info("Advantage Server ready (host=%s, port=%d)", cfg.host, cfg.port)

    @app.get("/health")
    async def health() -> Response:
        if app.state.ready:
            return Response(status_code=200, content=b'{"status":"ok"}', media_type="application/json")
        return Response(
            status_code=503, content=b'{"status":"loading"}', media_type="application/json"
        )

    @app.get("/weight_status")
    async def weight_status() -> Response:
        """Report the number of weight broadcasts received from the AdvTrainer.

        Starts at 0 (initial zero-init backbone). Incremented by /update_weights
        each time the AdvTrainer successfully broadcasts a new state_dict.
        The orchestrator polls this between steps to enforce the
        lockstep-with-AdvTrainer invariant: before issuing step N's
        /compute_advantages_and_targets calls, the orchestrator waits until
        weight_step >= N (= AdvTrainer has applied the broadcast resulting
        from training on batch N-1).
        """
        body = msgspec.json.encode({"weight_step": int(app.state.weight_step)})
        return Response(status_code=200, content=body, media_type="application/json")

    @app.post("/compute_advantages_and_targets")
    async def compute_endpoint(request: Request) -> Response:
        if not app.state.ready:
            return Response(status_code=503, content=b'{"error":"not ready"}', media_type="application/json")

        # Phase 10 perf instrumentation: tag every PROF line emitted during
        # this request with a short request id so the analyzer can group them.
        # No-op when PRIME_RL_ADV_PROF env var is unset.
        rid = new_request_id()
        set_request_id(rid)

        with prof("endpoint.TOTAL"):
            with prof("endpoint.body_read"):
                body = await request.body()
            with prof("endpoint.decode_request", body_bytes=len(body)):
                decoded = msgspec.msgpack.decode(body, type=ComputeAdvantagesRequest)

            backbone: ValueNetworkBackbone = app.state.backbone
            loop = asyncio.get_event_loop()

            def _do_compute():
                # Re-set the request id inside the executor thread; ContextVar
                # is not propagated across threads automatically.
                set_request_id(rid)
                kwargs: dict[str, float | int] = {"gamma": app.state.gamma}
                if decoded.algorithm == "ppo":
                    kwargs["lam"] = app.state.lam
                else:
                    kwargs["n_step"] = app.state.n_step
                    kwargs["arm_phi_decay"] = app.state.arm_phi_decay
                return compute_advantages_and_targets(
                    samples=decoded.samples,
                    episodic_reward=decoded.episodic_reward,
                    is_terminal=decoded.is_terminal,
                    algorithm=decoded.algorithm,
                    backbone=backbone,
                    tokenizer=app.state.tokenizer,
                    **kwargs,
                )

            with prof("endpoint.run_in_executor", sync_cuda=True):
                paired = await loop.run_in_executor(None, _do_compute)
            with prof("endpoint.build_response"):
                paired_samples = [
                    PairedSample(llm_sample=llm, advantage_sample=adv) for llm, adv in paired
                ]
                response = ComputeAdvantagesResponse(paired_samples=paired_samples)
            with prof("endpoint.encode_response"):
                encoded = msgspec.msgpack.encode(response)
            return Response(content=encoded, media_type="application/x-msgpack")

    @app.post("/update_weights")
    async def update_weights(request: Request) -> Response:
        """Phase 7c: receive a torch.save'd state_dict from the Advantage Trainer.

        Body: raw bytes (output of torch.save(state_dict, BytesIO)).
        State dict contains LoRA adapters + value head weights only (not the
        frozen base). load_state_dict(..., strict=False) tolerates the missing
        base parameters.
        """
        if not app.state.ready:
            return Response(status_code=503, content=b'{"error":"not ready"}', media_type="application/json")

        body = await request.body()
        backbone: ValueNetworkBackbone = app.state.backbone
        loop = asyncio.get_event_loop()

        def _do_load() -> tuple[int, int]:
            state_dict = torch.load(
                io.BytesIO(body), map_location="cpu", weights_only=True
            )
            # strict=False because the trainer only sends LoRA + heads, not the
            # frozen base. Returns the LISTS of missing/unexpected keys; the
            # missing list will be the (expected) frozen-base keys.
            result = backbone.load_state_dict(state_dict, strict=False)
            return len(result.missing_keys), len(result.unexpected_keys)

        n_missing, n_unexpected = await loop.run_in_executor(None, _do_load)
        if n_unexpected > 0:
            _LOGGER.warning(
                "update_weights: %d unexpected keys in state_dict (likely a "
                "config mismatch between Trainer and Server)",
                n_unexpected,
            )
        # Increment AFTER the state_dict load completes synchronously inside
        # the executor. This guarantees that when the AdvTrainer's POST
        # returns 200, app.state.weight_step has advanced AND the new
        # weights are live in the backbone. The orchestrator's
        # wait_for_weight_step gate can therefore rely on the counter as
        # an "applied-and-active" signal, not a "received-but-pending" one.
        app.state.weight_step = app.state.weight_step + 1
        _LOGGER.debug(
            "update_weights: applied state_dict (weight_step=%d, %d missing, %d unexpected)",
            app.state.weight_step, n_missing, n_unexpected,
        )
        return Response(
            status_code=200,
            content=b'{"status":"ok"}',
            media_type="application/json",
        )

    return app


def main() -> None:
    """CLI entrypoint: parse AdvantageServerConfig from CLI args / TOML, start uvicorn."""
    from prime_rl.utils.config import cli

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = cli(AdvantageServerConfig)
    app = create_app(config)
    # log_config=None: don't override our basicConfig with uvicorn's defaults.
    uvicorn.run(app, host=config.host, port=config.port, log_config=None)


if __name__ == "__main__":
    main()
