import concurrent.futures
import threading

import pytest

import litellm
from litellm.batch_completion.main import batch_completion_models, batch_completion_models_all_responses


def test_batch_completion_models_all_responses_submits_before_waiting(monkeypatch):
    """
    Regression test for issue #20704.
    Ensures all model calls are submitted to the thread pool before waiting on results.
    """
    models = ["model-a", "model-b", "model-c"]
    called_models = []

    class _AssertingFuture:
        def __init__(self, result, executor, expected_submissions):
            self._result = result
            self._executor = executor
            self._expected_submissions = expected_submissions

        def result(self):
            if self._executor.submit_count != self._expected_submissions:
                raise AssertionError(
                    "Not all model calls were submitted before waiting"
                )
            return self._result

    class _RecordingThreadPoolExecutor:
        def __init__(self, max_workers, *args, **kwargs):
            self.max_workers = max_workers
            self.submit_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            self.submit_count += 1
            result = fn(*args, **kwargs)
            return _AssertingFuture(
                result=result,
                executor=self,
                expected_submissions=len(models),
            )

    def _mock_completion(*args, model, **kwargs):
        called_models.append(model)
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)
    monkeypatch.setattr(
        concurrent.futures, "ThreadPoolExecutor", _RecordingThreadPoolExecutor
    )

    responses = batch_completion_models_all_responses(
        models=models,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert sorted(called_models) == sorted(models)
    assert len(responses) == len(models)
    assert sorted(response["model"] for response in responses) == sorted(models)


def test_batch_completion_models_all_responses_continues_on_model_error(monkeypatch):
    models = ["model-a", "model-error", "model-b"]

    def _mock_completion(*args, model, **kwargs):
        if model == "model-error":
            raise RuntimeError("simulated model failure")
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    responses = batch_completion_models_all_responses(
        models=models,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert len(responses) == 2
    assert sorted(response["model"] for response in responses) == ["model-a", "model-b"]


def test_batch_completion_models_all_responses_returns_empty_for_empty_models(
    monkeypatch,
):
    called = False

    def _mock_completion(*args, model, **kwargs):
        nonlocal called
        called = True
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    responses = batch_completion_models_all_responses(
        models=[],
        messages=[{"role": "user", "content": "hello"}],
    )

    assert responses == []
    assert called is False


def test_batch_completion_models_all_responses_accepts_single_model_string(monkeypatch):
    called_models = []

    def _mock_completion(*args, model, **kwargs):
        called_models.append(model)
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    responses = batch_completion_models_all_responses(
        models="model-a",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert called_models == ["model-a"]
    assert responses == [{"model": "model-a"}]


def test_batch_completion_models_returns_the_first_completed_response(monkeypatch):
    """Regression test for issue #44918: a slow first-listed model must not win over a faster one."""
    release_slow_model = threading.Event()

    def _mock_completion(*args, model, **kwargs):
        if model == "model-slow":
            release_slow_model.wait(timeout=5)
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    try:
        response = batch_completion_models(
            models=["model-slow", "model-fast"],
            messages=[{"role": "user", "content": "hello"}],
        )
        slow_model_still_running = not release_slow_model.is_set()
    finally:
        release_slow_model.set()

    assert response == {"model": "model-fast"}
    assert slow_model_still_running


def test_batch_completion_models_skips_a_failed_model(monkeypatch):
    def _mock_completion(*args, model, **kwargs):
        if model == "model-error":
            raise RuntimeError("simulated model failure")
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    response = batch_completion_models(
        models=["model-error", "model-ok"],
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response == {"model": "model-ok"}


def test_batch_completion_models_raises_when_every_model_fails(monkeypatch):
    def _mock_completion(*args, model, **kwargs):
        raise RuntimeError(f"{model} failed")

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    with pytest.raises(RuntimeError, match="failed"):
        batch_completion_models(
            models=["model-a", "model-b"],
            messages=[{"role": "user", "content": "hello"}],
        )
