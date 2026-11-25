"""Resuming an ARM run: the value side.

Three things have to hold after a restart, and none did before:

  1. The Advantage Trainer reads on from its checkpoint step. The orchestrator
     files each batch under its own step; the reader used to start again at
     step_0, a file the clean-up loop has long deleted (the value networks then
     stop training, with no error) or, if it is still there, an old batch.
  2. The Advantage Server gets the loaded value weights and their step. It
     restarts with the warm-start weights and weight_step 0, while the resumed
     orchestrator waits for weight_step >= its own step before it scores.
  3. A batch that can no longer arrive stops the run instead of freezing it.

Plus the launcher side: a resume appends to the process logs.

The end-to-end cases run the real train() entrypoint on the tiny CPU backbone
(harness of test_online_optimizer.py), with the real server app behind the
trainer's HTTP client.
"""

from __future__ import annotations

import inspect
import io
import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

import prime_rl.advantage_trainer.train as tr
import prime_rl.entrypoints.rl as rl_entry
from prime_rl.configs.shared import FileSystemTransportConfig
from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.transport.advantage_batch import (
    FileSystemAdvantageTrainingBatchReceiver,
    FileSystemAdvantageTrainingBatchSender,
    setup_advantage_training_batch_receiver,
)
from prime_rl.transport.types import AdvantageTrainingBatch
from tests.unit.advantage_server.test_server_endpoint import _build_tiny_app
from tests.unit.advantage_trainer.test_online_optimizer import _one, _online, _saved, _trainable, _warm_state
from tests.unit.advantage_trainer.test_train_step_action import _backbone

URL = "http://test-server:8200"


def _send(transport: Path, steps) -> None:
    sender = FileSystemAdvantageTrainingBatchSender(transport)
    for s in steps:
        sender.send(AdvantageTrainingBatch(examples=_one(s).examples, step=s))


def _no_waiting(monkeypatch, limit: int = 50) -> None:
    """A trainer that waits for a batch which is not there would hang the test:
    make the wait fail instead."""
    calls = {"n": 0}

    def sleep(_seconds):
        calls["n"] += 1
        if calls["n"] > limit:
            raise AssertionError("the trainer is waiting: no batch at the step it expects")

    monkeypatch.setattr(tr.time, "sleep", sleep)


_REAL_CLIENT = httpx.Client


def _http(monkeypatch, handler) -> None:
    """Route the trainer's httpx.Client through `handler` (the latest call wins)."""
    transport = httpx.MockTransport(handler)

    class _Client(_REAL_CLIENT):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(tr.httpx, "Client", _Client)


def _query_step(request: httpx.Request) -> str | None:
    return request.url.params.get("weight_step")


# ---------------------------------------------------------------- 1. the reader


def test_reader_starts_at_the_given_step(tmp_path):
    _send(tmp_path, range(5))
    fresh = setup_advantage_training_batch_receiver(FileSystemTransportConfig(), tmp_path)
    assert [b.step for b in fresh.receive()] == [0, 1, 2, 3, 4]

    resumed = setup_advantage_training_batch_receiver(FileSystemTransportConfig(), tmp_path, start_step=3)
    assert resumed.next_step == 3
    assert [b.step for b in resumed.receive()] == [3, 4]
    assert resumed.next_step == 5 and resumed.receive() == []


def test_reader_waits_for_its_step_and_not_for_step_0(tmp_path):
    reader = FileSystemAdvantageTrainingBatchReceiver(tmp_path, start_step=30)
    assert not reader.can_receive()
    _send(tmp_path, [0, 29])  # earlier steps are never read again
    assert not reader.can_receive() and reader.receive() == []
    _send(tmp_path, [30])
    assert reader.can_receive()
    assert [b.step for b in reader.receive()] == [30]


def test_find_gap(tmp_path):
    reader = FileSystemAdvantageTrainingBatchReceiver(tmp_path, start_step=2)
    assert reader.find_gap() is None  # nothing on disk yet
    _send(tmp_path, [0, 1])
    assert reader.find_gap() is None  # only earlier steps: the next one may still come
    (tmp_path / "rollouts" / "step_9").mkdir()  # a step dir without its batch file
    (tmp_path / "rollouts" / "step_x").mkdir()
    assert reader.find_gap() is None
    _send(tmp_path, [5, 4])
    assert reader.find_gap() == 4  # step_2 is missing and later batches exist
    _send(tmp_path, [2])
    assert reader.find_gap() is None  # the expected batch is there
    assert [b.step for b in reader.receive()] == [2]
    assert reader.find_gap() == 4  # now step_3 is the missing one


# ---------------------------------------------------------------- 2. the gap watch


class _Reader:
    next_step = 25

    def __init__(self, gaps):
        self.gaps = list(gaps)
        self.asked = 0

    def find_gap(self):
        self.asked += 1
        return self.gaps.pop(0)


def test_gap_watch_raises_on_the_second_sighting_in_a_row():
    watch = tr._GapWatch(interval=30.0)
    reader = _Reader([28, 28])
    watch.check(reader, 100.0)  # the wait starts
    watch.check(reader, 120.0)  # too early to look
    assert reader.asked == 0
    watch.check(reader, 130.0)  # first sighting
    assert reader.asked == 1
    with pytest.raises(RuntimeError, match=r"step 25.*step 28.*will not arrive"):
        watch.check(reader, 160.0)


def test_gap_watch_forgives_a_gap_that_closes():
    watch = tr._GapWatch(interval=1.0)
    reader = _Reader([28, None, 28, 28])
    for now in (0.0, 1.0, 2.0, 3.0):  # gap, no gap, gap: never two in a row
        watch.check(reader, now)
    watch.reset()  # a batch arrived
    watch.check(reader, 10.0)
    watch.check(reader, 11.0)  # first sighting after the reset
    assert reader.asked == 4


# ---------------------------------------------------------------- 3. the server's counter


def _body(backbone) -> bytes:
    buf = io.BytesIO()
    torch.save({n: p.detach().cpu() for n, p in backbone.named_parameters() if p.requires_grad}, buf)
    return buf.getvalue()


def test_server_takes_an_explicit_weight_step(monkeypatch):
    app = _build_tiny_app(monkeypatch)
    with TestClient(app) as client:
        body = _body(app.state.backbone)
        post = lambda **kw: client.post(  # noqa: E731
            "/update_weights", content=body, headers={"Content-Type": "application/octet-stream"}, **kw
        )
        assert post().status_code == 200
        assert client.get("/weight_status").json() == {"weight_step": 1}  # a broadcast still counts one
        assert post(params={"weight_step": "30"}).status_code == 200
        assert client.get("/weight_status").json() == {"weight_step": 30}  # the resume hand-over sets it
        assert post().status_code == 200
        assert client.get("/weight_status").json() == {"weight_step": 31}  # and counting goes on from there
        for bad in ("-1", "abc", "2.5"):
            assert post(params={"weight_step": bad}).status_code == 400
        assert client.get("/weight_status").json() == {"weight_step": 31}


def test_server_rejects_a_bad_step_before_touching_the_weights(monkeypatch):
    app = _build_tiny_app(monkeypatch)
    with TestClient(app) as client:
        before = _trainable(app.state.backbone)
        other = _backbone(randomize_heads=True)
        r = client.post("/update_weights", content=_body(other), params={"weight_step": "-3"})
        assert r.status_code == 400
        after = _trainable(app.state.backbone)
        assert all(torch.equal(before[n], after[n]) for n in before)


# ---------------------------------------------------------------- 4. the hand-over


def test_post_weights_sends_the_step_only_when_asked(monkeypatch):
    seen = []

    def handler(request):
        seen.append((request.url.path, _query_step(request)))
        return httpx.Response(200, json={"status": "ok"})

    _http(monkeypatch, handler)
    bb = _backbone()
    assert tr._post_weights(bb, URL, timeout=5.0) is None
    assert tr._post_weights(bb, URL + "/", timeout=5.0, weight_step=7) is None
    assert tr._broadcast_weights(bb, URL) is True
    assert seen == [("/update_weights", None), ("/update_weights", "7"), ("/update_weights", None)]


def test_push_resume_weights_retries_until_the_server_is_up(monkeypatch):
    replies = [503, 503, 200]
    steps = []

    def handler(request):
        steps.append(_query_step(request))
        return httpx.Response(replies.pop(0), content=b"{}")

    _http(monkeypatch, handler)
    slept = []
    attempts = tr._push_resume_weights(
        _backbone(), URL, 30, timeout=100.0, retry_interval=5.0, sleep=slept.append, clock=lambda: 0.0
    )
    assert attempts == 3 and slept == [5.0, 5.0]
    assert steps == ["30", "30", "30"]  # every attempt carries the step


def test_push_resume_weights_gives_up_after_the_timeout(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("connection refused")

    _http(monkeypatch, handler)
    now = {"t": 0.0}

    def sleep(seconds):
        now["t"] += seconds

    with pytest.raises(RuntimeError, match=r"did not take the value weights of checkpoint step 30"):
        tr._push_resume_weights(
            _backbone(), URL, 30, timeout=12.0, retry_interval=5.0, sleep=sleep, clock=lambda: now["t"]
        )
    assert now["t"] == 15.0  # tries at 0, 5, 10 and 15 s; the last is past the deadline


# ---------------------------------------------------------------- 5. train() end to end


CKPT = CheckpointConfig(interval=1)
RESUME = CheckpointConfig(interval=1, resume_step=-1)


def test_resumed_trainer_reads_on_after_the_old_batches_are_deleted(tmp_path, monkeypatch):
    """The fault itself: after a resume the reader waited for step_0, which the
    launcher's clean-up loop had deleted, and never trained again."""
    out = tmp_path / "run"
    common = dict(warm_start_path=_warm_state(tmp_path), n_epochs=1)
    _online(monkeypatch, out, [_one(0), _one(1)], max_steps=2, ckpt=CKPT, **common)
    assert _saved(out, 2)["step"] == 2
    for old in (0, 1):  # what the clean-up loop does
        shutil.rmtree(out / "transport" / "rollouts" / f"step_{old}")

    _no_waiting(monkeypatch)
    resumed = _online(monkeypatch, out, [_one(2)], first_step=2, max_steps=3, ckpt=RESUME, **common)
    assert [line.split(" ")[0] for line in resumed.lines] == ["step=2"]
    assert _saved(out, 3)["step"] == 3
    before, after = _saved(out, 2)["trainable_state_dict"], _saved(out, 3)["trainable_state_dict"]
    assert any(not torch.equal(before[n], after[n]) for n in after)  # it trained


def test_resume_does_not_train_old_batches_again(tmp_path, monkeypatch):
    """Without the clean-up loop step_0 is still on disk: the reader must not
    start there again (run-001, June 2026: value step 55 at orchestrator step 25).
    The resumed run equals the uninterrupted one."""
    settings = dict(warm_start_path=_warm_state(tmp_path), n_epochs=1, learning_rate=1e-3)
    batches = [_one(i) for i in range(4)]
    _online(monkeypatch, tmp_path / "full", batches, max_steps=4, ckpt=CKPT, **settings)

    _online(monkeypatch, tmp_path / "cut", batches[:2], max_steps=2, ckpt=CKPT, **settings)
    # steps 2 and 3 were already written when the run was interrupted: read in order
    _no_waiting(monkeypatch)
    resumed = _online(monkeypatch, tmp_path / "cut", batches[2:], first_step=2, max_steps=4, ckpt=RESUME, **settings)
    assert [line.split(" ")[0] for line in resumed.lines] == ["step=2", "step=3"]

    a, b = _saved(tmp_path / "full", 4), _saved(tmp_path / "cut", 4)
    for name in a["trainable_state_dict"]:
        assert torch.equal(a["trainable_state_dict"][name], b["trainable_state_dict"][name]), name


def _same(served: torch.Tensor, saved: torch.Tensor) -> bool:
    """A served weight equals a saved one in the server's own dtype (bfloat16 on a GPU)."""
    return torch.equal(served.cpu(), saved.to(served.dtype).cpu())


def _server_behind(monkeypatch, client: TestClient, log: list) -> None:
    """The trainer's HTTP client talks to the real server app."""

    def handler(request):
        log.append(_query_step(request))
        r = client.post(
            request.url.path,
            params=dict(request.url.params),
            content=request.content,
            headers={"Content-Type": "application/octet-stream"},
        )
        return httpx.Response(r.status_code, content=r.content)

    _http(monkeypatch, handler)


def test_resume_hands_weights_and_step_to_the_server(tmp_path, monkeypatch):
    out = tmp_path / "run"
    common = dict(warm_start_path=_warm_state(tmp_path), n_epochs=1, advantage_server_url=URL)

    # the job that is interrupted after two cycles
    posts: list = []
    with TestClient(_build_tiny_app(monkeypatch)) as client:
        _server_behind(monkeypatch, client, posts)
        _online(monkeypatch, out, [_one(0), _one(1)], max_steps=2, ckpt=CKPT, **common)
        assert posts == [None, None]  # a fresh run only broadcasts
        assert client.get("/weight_status").json() == {"weight_step": 2}
    ckpt = _saved(out, 2)["trainable_state_dict"]

    # everything restarts: a new server (weight_step 0, not the trained weights)
    posts = []
    app = _build_tiny_app(monkeypatch)
    with TestClient(app) as client:
        assert client.get("/weight_status").json() == {"weight_step": 0}
        served = _trainable(app.state.backbone)
        assert any(not torch.equal(served[n].cpu(), ckpt[n].cpu()) for n in ckpt)
        _server_behind(monkeypatch, client, posts)

        seen_at_first_cycle = {}
        real_step = tr._train_step_action

        def spy(**kw):
            if not seen_at_first_cycle:
                seen_at_first_cycle["weight_step"] = client.get("/weight_status").json()["weight_step"]
                seen_at_first_cycle["weights"] = _trainable(app.state.backbone)
            return real_step(**kw)

        monkeypatch.setattr(tr, "_train_step_action", spy)
        _no_waiting(monkeypatch)
        resumed = _online(monkeypatch, out, [_one(2)], first_step=2, max_steps=3, ckpt=RESUME, **common)

        # before the first cycle after the resume the server holds checkpoint 2 at step 2
        assert seen_at_first_cycle["weight_step"] == 2
        for name, value in ckpt.items():
            assert _same(seen_at_first_cycle["weights"][name], value), name
        # then one ordinary broadcast: step 3, the trainer's final weights
        assert posts == ["2", None]
        assert client.get("/weight_status").json() == {"weight_step": 3}
        final, served = _trainable(resumed.backbone), _trainable(app.state.backbone)
        for name, value in final.items():
            assert _same(served[name], value), name


def test_a_fresh_run_never_sets_the_step(tmp_path, monkeypatch):
    """resume_step = -1 with no checkpoint is a fresh run (the run configs always
    carry it): warm start, reader at step 0, no hand-over."""
    posts: list = []
    with TestClient(_build_tiny_app(monkeypatch)) as client:
        _server_behind(monkeypatch, client, posts)
        _no_waiting(monkeypatch)
        run = _online(
            monkeypatch, tmp_path / "run", [_one(0)], max_steps=1, ckpt=RESUME,
            warm_start_path=_warm_state(tmp_path), n_epochs=1, advantage_server_url=URL,
        )
        assert posts == [None]
        assert run.lines[0].startswith("step=0 ")
        assert client.get("/weight_status").json() == {"weight_step": 1}


def test_a_batch_that_cannot_arrive_stops_the_trainer(tmp_path, monkeypatch):
    """Checkpoint at step 2, but the batches left on disk start at step 4: the
    trainer used to wait for ever. Now it stops with the reason."""
    out = tmp_path / "run"
    common = dict(warm_start_path=_warm_state(tmp_path), n_epochs=1)
    _online(monkeypatch, out, [_one(0), _one(1)], max_steps=2, ckpt=CKPT, **common)
    shutil.rmtree(out / "transport" / "rollouts")

    monkeypatch.setattr(tr, "_GapWatch", lambda: _RealGapWatch(interval=0.0))
    _no_waiting(monkeypatch, limit=20)  # the watch must stop the wait well before that
    with pytest.raises(RuntimeError, match=r"step 2.*step 4.*will not arrive"):
        _online(monkeypatch, out, [_one(4)], first_step=4, max_steps=5, ckpt=RESUME, **common)


_RealGapWatch = tr._GapWatch


def test_a_half_written_checkpoint_is_not_resumed_from(tmp_path, monkeypatch):
    """The job died while it saved checkpoint 2: the step dir is there, its STABLE
    marker is not. The trainer resumes from checkpoint 1 and trains batch 1 again
    from disk (the launcher's check counts the same way)."""
    out = tmp_path / "run"
    common = dict(warm_start_path=_warm_state(tmp_path), n_epochs=1)
    _online(monkeypatch, out, [_one(0), _one(1)], max_steps=2, ckpt=CKPT, **common)
    whole = _saved(out, 2)["trainable_state_dict"]
    (out / "checkpoints" / "step_2" / "STABLE").unlink()
    (out / "checkpoints" / "step_2" / "trainer" / "value_state.pt").write_bytes(b"half a file")

    _no_waiting(monkeypatch)
    resumed = _online(monkeypatch, out, [], first_step=2, max_steps=2, ckpt=RESUME, **common)
    assert [line.split(" ")[0] for line in resumed.lines] == ["step=1"]
    assert (out / "checkpoints" / "step_2" / "STABLE").exists()
    again = _saved(out, 2)
    assert again["step"] == 2
    for name, value in whole.items():  # the same cycle on the same batch: the same weights
        assert torch.equal(again["trainable_state_dict"][name], value), name


# ---------------------------------------------------------------- 6. the process logs


def _launch_config(out: Path, resume_step, ckpt_dir: Path | None = None):
    ckpt = None if resume_step == "no ckpt section" else SimpleNamespace(resume_step=resume_step, output_dir=ckpt_dir)
    return SimpleNamespace(output_dir=out, ckpt=ckpt)


def test_continues_existing_run(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    assert rl_entry.continues_existing_run(_launch_config(out, "no ckpt section")) is False
    assert rl_entry.continues_existing_run(_launch_config(out, None)) is False
    # resume_step = -1 in a run directory without checkpoints: a fresh run
    assert rl_entry.continues_existing_run(_launch_config(out, -1)) is False
    (out / "checkpoints").mkdir()
    assert rl_entry.continues_existing_run(_launch_config(out, -1)) is False  # empty
    (out / "checkpoints" / "step_5").mkdir()
    assert rl_entry.continues_existing_run(_launch_config(out, -1)) is True
    assert rl_entry.continues_existing_run(_launch_config(out, 5)) is True
    assert rl_entry.continues_existing_run(_launch_config(out, None)) is False  # not resuming: truncate
    # checkpoints kept in their own directory
    elsewhere = tmp_path / "ckpts"
    (elsewhere / "checkpoints" / "step_5").mkdir(parents=True)
    assert rl_entry.continues_existing_run(_launch_config(tmp_path / "other", -1, elsewhere)) is True


def test_every_process_log_follows_the_resume_mode():
    source = inspect.getsource(rl_entry.rl_local)
    assert 'log_mode = "a" if continues_existing_run(config) else "w"' in source
    assert source.count('.stdout", log_mode)') == 6
    assert '.stdout", "w")' not in source
