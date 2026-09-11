"""Superseded jobs must not run: queued stale jobs are skipped, running ones abort."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fastmatch.types import MatchParams
from fastmatch.worker import MatchWorker


class _Engine:
    def __init__(self, on_poll=None):
        self.calls = 0
        self.polls = 0
        self._on_poll = on_poll

    def match(self, template, params, *, exclude_box=None, mask=None, cancel=None, progress=None):
        self.calls += 1
        for _ in range(100):  # 100 "tiles"
            self.polls += 1
            if self._on_poll:
                self._on_poll(self.polls)
            if cancel():
                return []
        return ["done"]


def _collect(worker):
    got = []
    worker.finished.connect(lambda m, j: got.append((m, j)))
    return got


def test_queued_stale_job_is_skipped():
    eng = _Engine()
    w = MatchWorker(eng)
    got = _collect(w)
    w.latest_job_id = 5          # the controller already dispatched job 5
    w.run(None, MatchParams(), None, None, None, 3)
    assert eng.calls == 0 and got == []
    w.run(None, MatchParams(), None, None, None, 5)
    assert eng.calls == 1 and got == [(["done"], 5)]


def test_running_job_aborts_when_superseded():
    w = MatchWorker(None)
    # A newer job is dispatched while job 1 is on its 10th tile.
    eng = _Engine(on_poll=lambda n: setattr(w, "latest_job_id", 2) if n == 10 else None)
    w._engine = eng
    got = _collect(w)
    w.latest_job_id = 1
    w.run(None, MatchParams(), None, None, None, 1)
    assert eng.polls == 10 and got == [([], 1)]
