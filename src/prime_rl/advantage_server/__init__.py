"""Advantage Server module: forward-only HTTP service that owns the
ValueNetworkBackbone (V/Q+/V_target) and produces per-token advantages
and Advantage Trainer regression targets per-rollout.

Phase 6a: compute logic + client + tests.
Phase 6b: HTTP server + orchestrator integration + launch script wiring.
"""
