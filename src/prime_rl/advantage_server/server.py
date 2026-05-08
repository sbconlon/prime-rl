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
    app.state.ready = False
    app.state.gamma = config.gamma
    app.state.lam = config.lam
    app.state.n_step = config.n_step

    @app.on_event("startup")
    async def _load_backbone() -> None:
        cfg: AdvantageServerConfig = app.state.config
        _LOGGER.info("Loading base model %s for ValueNetworkBackbone", cfg.model_name)
        base = AutoModel.from_pretrained(cfg.model_name)
        backbone = ValueNetworkBackbone(
            base, lora_config=cfg.lora, polyak_tau=cfg.polyak_tau
        )
        backbone.eval()
        app.state.backbone = backbone
        app.state.ready = True
        _LOGGER.info("Advantage Server ready (host=%s, port=%d)", cfg.host, cfg.port)

    @app.get("/health")
    async def health() -> Response:
        if app.state.ready:
            return Response(status_code=200, content=b'{"status":"ok"}', media_type="application/json")
        return Response(
            status_code=503, content=b'{"status":"loading"}', media_type="application/json"
        )

    @app.post("/compute_advantages_and_targets")
    async def compute_endpoint(request: Request) -> Response:
        if not app.state.ready:
            return Response(status_code=503, content=b'{"error":"not ready"}', media_type="application/json")

        body = await request.body()
        decoded = msgspec.msgpack.decode(body, type=ComputeAdvantagesRequest)

        backbone: ValueNetworkBackbone = app.state.backbone
        loop = asyncio.get_event_loop()

        def _do_compute():
            kwargs: dict[str, float | int] = {"gamma": app.state.gamma}
            if decoded.algorithm == "ppo":
                kwargs["lam"] = app.state.lam
            else:
                kwargs["n_step"] = app.state.n_step
            return compute_advantages_and_targets(
                samples=decoded.samples,
                episodic_reward=decoded.episodic_reward,
                is_terminal=decoded.is_terminal,
                algorithm=decoded.algorithm,
                backbone=backbone,
                **kwargs,
            )

        paired = await loop.run_in_executor(None, _do_compute)
        paired_samples = [
            PairedSample(llm_sample=llm, advantage_sample=adv) for llm, adv in paired
        ]
        response = ComputeAdvantagesResponse(paired_samples=paired_samples)
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
        _LOGGER.debug(
            "update_weights: applied state_dict (%d missing, %d unexpected)",
            n_missing,
            n_unexpected,
        )
        app.state.weight_step = getattr(app.state, "weight_step", 0) + 1
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
