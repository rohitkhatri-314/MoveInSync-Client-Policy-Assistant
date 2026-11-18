import asyncio
import pytest
from unittest.mock import patch
from app.ingestion import IngestionEngine
from app.models import UserClaims
from app.retriever import PolicyRetriever
from app.service import CommuteService

@pytest.fixture(scope="module")
def commute_service():
    engine = IngestionEngine()
    engine.ingest_policy_file("policies/nexuspay_v2.md", "NexusPay", 2)
    retriever = PolicyRetriever(chroma_client=engine.client)
    return CommuteService(retriever=retriever)

@pytest.mark.asyncio
async def test_max_call_and_chunk_limits(commute_service):
    user = UserClaims(user_id="emp_01", company_id="NexusPay", role="employee")
    res = await commute_service.answer_question("Am I eligible for a 9:00 PM drop with escort?", user=user)
    
    # Assert at most 2 LLM calls (Rule: 'At most 2 LLM calls per question')
    assert res.llm_calls <= 2
    # Assert at most 3 cited chunks (Rule: 'no more than 3 cited chunks per answer')
    assert len(res.citations) <= 3
    # Assert latency under SLA (< 3000 ms)
    assert res.latency_ms < 3000.0

@pytest.mark.asyncio
async def test_sla_timeout_resilience(commute_service):
    user = UserClaims(user_id="emp_01", company_id="NexusPay", role="employee")
    
    # Simulate an LLM call that takes longer than LLM_TIMEOUT_SECONDS (2.5s)
    def slow_llm_generate(*args, **kwargs):
        import time
        time.sleep(3.0)
        return "Slow response", [], 1

    with patch.object(commute_service.llm, "generate", side_effect=slow_llm_generate):
        res = await commute_service.answer_question("What is the cutoff timing?", user=user)
        
        # Must return timeout fallback gracefully, not crash or raise 500
        assert res.status == "timeout_fallback"
        assert "latency timeout" in res.answer.lower() or "taking longer" in res.answer.lower()
        assert "transport desk" in res.answer.lower()
