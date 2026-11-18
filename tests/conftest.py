import pytest
from app.config import settings
from app.cache import query_cache

@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch):
    """
    Enforces deterministic offline testing per MoveInSync specification:
    'Your tests may use a fake LLM. Only the demo needs a real one, so a quota error should never cost you a passing test.'
    """
    query_cache.clear()
    monkeypatch.setattr(settings, "LLM_PROVIDER", "fake")
    monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 2.5)
