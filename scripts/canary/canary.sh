#!/usr/bin/env bash
# Phase 10 ARM canary -- single-script bootstrap + run + cleanup.
#
# Captures every fault hit during the first cluster bring-up:
#   - Compute nodes have no internet (HF cache, uv resolve, model pre-fetch).
#   - flash-attn + reverse-text + verifiers must all be installed; uv sync
#     alone doesn't pull the optional flash-attn extra, and the PrimeIntellect
#     index isn't in pyproject.toml's main resolve.
#   - wandb 0.24.2 + protobuf 6 are incompatible (pin protobuf<6).
#   - uv sync wipes editable verifiers; UV_NO_SYNC=1 is per-shell, so it
#     must be set in every shell where we run the canary.
#   - uv run with a piped tee buffers stdout silently; PYTHONUNBUFFERED=1
#     fixes that.
#
# Usage:
#
#   # (1) On slnode01 (or any login node with internet)
#   bash scripts/canary/canary.sh bootstrap
#
#   # (2) Allocate GPUs
#   salloc --partition=stud --account=3312841 --qos=stud \
#          --nodes=1 --gres=gpu:4 --cpus-per-task=8 --mem=200G \
#          --time=03:00:00
#   srun --jobid=<JOBID> --pty bash -l
#
#   # (3) On the compute node
#   bash scripts/canary/canary.sh run
#
#   # If a prior run wedged the node:
#   bash scripts/canary/canary.sh cleanup
#
# Idempotent. Safe to re-run bootstrap if anything in the venv drifts.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration -- override via env if your paths / model differ.
# ---------------------------------------------------------------------------

PRIME_RL_DIR="${PRIME_RL_DIR:-/home/3312841/prime-rl}"
VERIFIERS_DIR="${VERIFIERS_DIR:-/home/3312841/verifiers}"
VERIFIERS_BRANCH="${VERIFIERS_BRANCH:-feature/arm-ppo}"
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_REPO="${MODEL_REPO:-PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT}"
PRIME_INDEX_URL="${PRIME_INDEX_URL:-https://hub.primeintellect.ai/primeintellect/simple/}"
ENV_FILE="${ENV_FILE:-$PRIME_RL_DIR/.canary_env}"
RUNS_DIR="${RUNS_DIR:-/home/3312841/runs}"
CONFIG_TOML="${CONFIG_TOML:-configs/ci/integration/rl_arm/start.toml}"

# Colors for readable status (degrade to plain if not a TTY)
if [ -t 1 ]; then
    C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_INFO=$'\033[36m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
    C_OK=""; C_ERR=""; C_INFO=""; C_DIM=""; C_OFF=""
fi

step() { echo "${C_INFO}==> $*${C_OFF}"; }
ok()   { echo "${C_OK}    OK${C_OFF}: $*"; }
fail() { echo "${C_ERR}    FAIL${C_OFF}: $*" >&2; exit 1; }
note() { echo "${C_DIM}    $*${C_OFF}"; }

usage() {
    cat <<EOF
Usage: $0 {bootstrap|run|cleanup}

  bootstrap   Run on a login node (with internet). Sets up the venv
              with consistent versions of every dep, pre-fetches the HF
              model, installs the reverse-text env + verifiers fork
              editable, and writes $ENV_FILE for compute-node sessions
              to source.

  run         Run on a compute node (post-srun). Sources $ENV_FILE,
              checks the venv is still healthy, and launches the
              canary with stdout redirected to a fresh log file under
              $RUNS_DIR. Tails the log so you see live progress.

  cleanup     Run on a compute node when a prior canary wedged the GPU.
              Sends SIGTERM (then SIGKILL if needed) to your own
              compute processes that are still holding GPU memory.

Env overrides:
  PRIME_RL_DIR        ($PRIME_RL_DIR)
  VERIFIERS_DIR       ($VERIFIERS_DIR)
  VERIFIERS_BRANCH    ($VERIFIERS_BRANCH)
  HF_HOME             ($HF_HOME_DIR)
  MODEL_REPO          ($MODEL_REPO)
  ENV_FILE            ($ENV_FILE)
  RUNS_DIR            ($RUNS_DIR)
  CONFIG_TOML         ($CONFIG_TOML)
EOF
    exit 1
}

# ---------------------------------------------------------------------------
# Subcommand: bootstrap (login node)
# ---------------------------------------------------------------------------

cmd_bootstrap() {
    step "Pre-flight checks"
    [ -d "$PRIME_RL_DIR" ] || fail "PRIME_RL_DIR ($PRIME_RL_DIR) does not exist"
    [ -d "$VERIFIERS_DIR" ] || fail "VERIFIERS_DIR ($VERIFIERS_DIR) does not exist"
    command -v uv >/dev/null 2>&1 || fail "uv is not on PATH"
    if ! curl -sSf --max-time 5 https://github.com >/dev/null 2>&1; then
        fail "no internet -- run bootstrap from a login node, not a compute node"
    fi
    ok "internet reachable, uv installed, paths exist"

    step "verifiers checkout: ensure branch '$VERIFIERS_BRANCH'"
    cd "$VERIFIERS_DIR"
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" != "$VERIFIERS_BRANCH" ]; then
        note "switching from '$current_branch' to '$VERIFIERS_BRANCH'"
        git fetch origin --quiet
        git checkout "$VERIFIERS_BRANCH"
    fi
    git fetch origin --quiet
    git pull origin "$VERIFIERS_BRANCH" --ff-only
    note "verifiers HEAD: $(git log -1 --oneline)"
    ok "verifiers on $VERIFIERS_BRANCH"

    step "prime-rl checkout"
    cd "$PRIME_RL_DIR"
    note "prime-rl HEAD: $(git log -1 --oneline) on branch $(git rev-parse --abbrev-ref HEAD)"

    step "uv sync (clean lockfile resolve, with flash-attn extra)"
    unset UV_NO_SYNC
    uv sync --extra flash-attn
    ok "uv sync done"

    # Install reverse-text BEFORE pinning protobuf<6 so the resolver
    # can satisfy prime-sandboxes' transitive protobuf>=6.31.1 marker
    # against the post-sync protobuf 6.x already in the venv. Use
    # --extra-index-url (NOT --index-url, which is exclusive) so PyPI
    # remains available for verifiers + prime-sandboxes lookups.
    step "install reverse-text env (PrimeIntellect index + PyPI)"
    uv pip install reverse-text \
        --extra-index-url "$PRIME_INDEX_URL" \
        --prerelease=allow >/dev/null
    ok "reverse-text installed"

    step "install verifiers fork (editable, on top of synced venv)"
    uv pip install -e "$VERIFIERS_DIR" --force-reinstall >/dev/null
    ok "verifiers editable from $VERIFIERS_DIR"

    # Pin protobuf to >=6.31.1,<7 LAST. The verifiers editable install
    # bumps protobuf to 7.x, which breaks wandb 0.24.2\'s pb2 stubs
    # ("Imports" missing from wandb_telemetry_pb2 at runtime). The
    # floor 6.31.1 is what prime-sandboxes 0.2.23 was gencoded against
    # (per the protobuf cross-version guarantee, runtime must be >=
    # gencode). Pinning into the 6.x range keeps both wandb and
    # prime-sandboxes happy.
    step "pin protobuf>=6.31.1,<7 (wandb pb2 + prime-sandboxes gencode floor)"
    uv pip install --force-reinstall 'protobuf>=6.31.1,<7' >/dev/null
    ok "protobuf pinned to 6.x"

    # Same pattern as the protobuf pin: the verifiers editable install
    # bumped starlette 0.50 -> 1.0, but fastapi 0.124.4 (lockfile pin)
    # still passes `on_startup` to starlette\'s Router which 1.0
    # dropped, breaking the Advantage Server\'s FastAPI() init.
    # Single-package --force-reinstall keeps fastapi at 0.124.4 and
    # rolls starlette back into the working range.
    step "pin starlette<1 (fastapi 0.124.4 needs starlette 0.x Router API)"
    uv pip install --force-reinstall 'starlette<1' >/dev/null
    ok "starlette pinned to 0.x"

    # Lock the venv NOW: every `uv run python` from this point on must
    # NOT trigger an implicit sync (which would wipe the editable
    # verifiers, reverse-text, and the protobuf+starlette pins we
    # just set up).
    export UV_NO_SYNC=1

    step "pre-fetch HF model: $MODEL_REPO"
    SNAP=$(uv run python - <<EOF
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id="$MODEL_REPO"))
EOF
)
    [ -d "$SNAP" ] || fail "snapshot_download did not return a directory: $SNAP"
    ok "model snapshot at $SNAP"

    step "verify venv coherence (UV_NO_SYNC=1 already set; subsequent uv run must NOT re-sync)"
    uv run python - <<'EOF'
import sys
import google.protobuf
proto_v = google.protobuf.__version__
assert proto_v.startswith("6."), f"protobuf must be 6.x, got {proto_v}"

import wandb
import wandb.proto.wandb_telemetry_pb2 as t
assert hasattr(t, "Imports"), "wandb proto stubs broken (missing Imports)"

import verifiers as vf
env = vf.load_environment("reverse-text")
assert env is not None

import verifiers.clients.openai_chat_completions_client as m
import inspect
src = inspect.getsource(m._extract_top_k_token_ids_from_logprobs_content)
assert "token_id:" in src and "_resolve_token_id" in src, \
    "verifiers fork is missing the Phase 5b vLLM-0.17 followup patch -- " \
    "ensure feature/arm-ppo HEAD includes it"

import flash_attn
import fastapi
import starlette
starlette_v = starlette.__version__
assert starlette_v.startswith("0."), \
    f"starlette must be 0.x (fastapi 0.124.4 incompat with starlette 1.0+), got {starlette_v}"
# Smoke-build a FastAPI app to catch the on_startup Router mismatch
# at bootstrap time, not in the live Advantage Server subprocess.
from fastapi import FastAPI
FastAPI(title="bootstrap-smoke")
print(
    f"protobuf={proto_v} | wandb={wandb.__version__} | "
    f"flash_attn={flash_attn.__version__} | "
    f"fastapi={fastapi.__version__} | starlette={starlette_v} | "
    f"verifiers={m.__file__}"
)
EOF
    ok "venv coherent"

    step "write $ENV_FILE for compute-node sessions"
    cat > "$ENV_FILE" <<EOF
# Sourced by 'canary.sh run' on compute nodes.
# Re-source in every new shell -- these env vars are per-shell.

# uv discipline (do NOT let implicit syncs wipe the editable verifiers install).
export UV_NO_SYNC=1

# HF lookup must stay local; compute nodes have no internet.
export HF_HOME='$HF_HOME_DIR'
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Make Python flush stdout/stderr eagerly so 'tee' shows live progress.
export PYTHONUNBUFFERED=1

# Resolved snapshot path -- avoids transformers' is_base_mistral
# online check (which fires even when HF_HUB_OFFLINE=1 is set).
export SNAP='$SNAP'

# Wandb: pick ONE.
# - If you have a key, fill it in (or export WANDB_API_KEY before sourcing).
# - Otherwise leave WANDB_MODE=offline so the trainer logs locally.
: "\${WANDB_API_KEY:=}"
: "\${WANDB_MODE:=offline}"
export WANDB_API_KEY WANDB_MODE

# Bocconi gnodes are MIG-partitioned -- adjacent slices are hardware-
# isolated, so the launcher\'s "existing processes on GPUs" pre-flight
# check (entrypoints/rl.py: check_gpus_available) does not apply. The
# error message itself recommends this env var for MIG systems.
export PRIME_RL_SKIP_GPU_CHECK=1
EOF
    ok "wrote $ENV_FILE"

    cat <<EOF

${C_OK}Bootstrap complete.${C_OFF}

Next:
  1. Allocate GPUs (or attach to existing salloc):
     ${C_DIM}salloc --partition=stud --account=3312841 --qos=stud \\
            --nodes=1 --gres=gpu:4 --cpus-per-task=8 --mem=200G --time=03:00:00${C_OFF}
     ${C_DIM}srun --jobid=<JOBID> --pty bash -l${C_OFF}

  2. On the compute node:
     ${C_DIM}cd $PRIME_RL_DIR${C_OFF}
     ${C_DIM}bash scripts/canary/canary.sh run${C_OFF}

EOF
}

# ---------------------------------------------------------------------------
# Subcommand: run (compute node)
# ---------------------------------------------------------------------------

cmd_run() {
    step "Pre-flight: env + venv health"
    [ -f "$ENV_FILE" ] || fail "missing $ENV_FILE -- run 'canary.sh bootstrap' from slnode01 first"
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    [ -d "$SNAP" ] || fail "SNAP from env file does not exist: $SNAP -- re-run bootstrap"
    ok "env loaded; SNAP=$SNAP"

    cd "$PRIME_RL_DIR"

    step "venv sanity (UV_NO_SYNC=1 should keep this fast and side-effect-free)"
    uv run python - <<'EOF'
import google.protobuf
assert google.protobuf.__version__.startswith("6."), \
    f"protobuf {google.protobuf.__version__} -- bootstrap pinned to 6.x, did UV_NO_SYNC slip?"
import wandb.proto.wandb_telemetry_pb2 as t
assert hasattr(t, "Imports"), "wandb proto broken; rerun bootstrap"
import starlette
assert starlette.__version__.startswith("0."), \
    f"starlette {starlette.__version__} -- bootstrap pinned to 0.x, did UV_NO_SYNC slip?"
from fastapi import FastAPI
FastAPI(title="run-smoke")  # catches the on_startup Router mismatch early
import verifiers.clients.openai_chat_completions_client as m
import inspect
src = inspect.getsource(m._extract_top_k_token_ids_from_logprobs_content)
assert "token_id:" in src and "_resolve_token_id" in src, "verifiers patch missing"
print("venv OK")
EOF
    ok "venv coherent"

    step "GPU pre-flight"
    if command -v nvidia-smi >/dev/null 2>&1; then
        n_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
        note "GPUs visible: $n_gpus"
        # Warn (don't fail) if anything else is using a GPU on this node.
        existing=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v '^$' || true)
        if [ -n "$existing" ]; then
            echo "${C_DIM}    other compute processes on these GPUs:${C_OFF}"
            echo "$existing" | sed 's/^/      /'
            echo "${C_DIM}    if any are yours from a prior run, run 'canary.sh cleanup' first${C_OFF}"
        fi
    else
        note "nvidia-smi not on PATH -- did you 'module load cuda/12.8'?"
    fi

    step "module load cuda/12.8 (best-effort; ignore if module command unavailable)"
    if command -v module >/dev/null 2>&1; then
        module load cuda/12.8 2>/dev/null || note "cuda/12.8 module load failed; continuing"
    fi

    step "launch canary"
    OUT="$RUNS_DIR/arm-canary-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$OUT"
    LOG="$OUT/launcher.log"
    note "output dir: $OUT"
    note "log file:   $LOG"

    # Background launch so we can tail without ^C-killing it.
    (
        cd "$PRIME_RL_DIR"
        uv run rl @ "$CONFIG_TOML" \
            --model.name "$SNAP" \
            --advantage-server.model-name "$SNAP" \
            --advantage-trainer.model.base-model-name "$SNAP" \
            --output-dir "$OUT" > "$LOG" 2>&1
        echo "EXIT=$?" >> "$LOG"
    ) &
    LAUNCHER_PID=$!
    note "launcher PID: $LAUNCHER_PID (logging to $LOG)"
    note "tail will follow until the launcher exits; ^C only stops the tail (not the launcher)"
    note "to kill the launcher: kill $LAUNCHER_PID  (or 'canary.sh cleanup' for stuck GPU procs)"

    # Wait briefly for the log file to appear, then tail-follow until the
    # launcher exits.
    while [ ! -s "$LOG" ] && kill -0 "$LAUNCHER_PID" 2>/dev/null; do
        sleep 0.5
    done
    tail -F --pid="$LAUNCHER_PID" "$LOG" || true

    if grep -q '^EXIT=0$' "$LOG"; then
        ok "canary exited with status 0"
    else
        echo "${C_ERR}canary did not exit cleanly. Diagnose:${C_OFF}"
        echo "    tail $OUT/logs/orchestrator.stdout"
        echo "    tail $OUT/logs/advantage_server.stdout"
        echo "    tail $OUT/logs/advantage_trainer.stdout"
        echo "    tail $OUT/logs/inference.stdout"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Subcommand: cleanup (compute node)
# ---------------------------------------------------------------------------

cmd_cleanup() {
    step "Killing leftover compute processes owned by $(id -un)"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        fail "nvidia-smi not on PATH"
    fi

    my_uid=$(id -u)
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -v '^$' || true)
    if [ -z "$pids" ]; then
        ok "no compute processes on any GPU"
        return 0
    fi

    targets=""
    for pid in $pids; do
        if [ "$(stat -c %u /proc/"$pid" 2>/dev/null)" = "$my_uid" ]; then
            targets="$targets $pid"
        fi
    done

    if [ -z "$targets" ]; then
        ok "no leftover processes owned by you (other GPU users present though -- left untouched)"
        return 0
    fi

    note "SIGTERM ->$targets"
    kill -TERM $targets 2>/dev/null || true
    sleep 5
    survivors=""
    for pid in $targets; do
        if kill -0 "$pid" 2>/dev/null; then
            survivors="$survivors $pid"
        fi
    done
    if [ -n "$survivors" ]; then
        note "SIGKILL ->$survivors"
        kill -9 $survivors 2>/dev/null || true
    fi

    sleep 2
    remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -v '^$' || true)
    if [ -z "$remaining" ]; then
        ok "all GPUs free"
    else
        note "GPUs still occupied (likely other users' processes):"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | sed 's/^/      /'
    fi
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

case "${1:-}" in
    bootstrap) cmd_bootstrap ;;
    run)       cmd_run ;;
    cleanup)   cmd_cleanup ;;
    -h|--help|"") usage ;;
    *) echo "Unknown subcommand: $1" >&2; usage ;;
esac
