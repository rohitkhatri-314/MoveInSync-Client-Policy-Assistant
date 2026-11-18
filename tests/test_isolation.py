import pytest
from app.ingestion import IngestionEngine
from app.models import UserClaims
from app.retriever import PolicyRetriever
from app.service import CommuteService

@pytest.fixture(scope="module")
def setup_engine():
    engine = IngestionEngine()
    engine.ingest_policy_file("policies/nexuspay_v1.md", "NexusPay", 1)
    engine.ingest_policy_file("policies/aerologix_v1.md", "AeroLogix", 1)
    engine.ingest_policy_file("policies/cloudmatrix_v1.md", "CloudMatrix", 1)
    retriever = PolicyRetriever(chroma_client=engine.client)
    service = CommuteService(retriever=retriever)
    return service, retriever

@pytest.mark.asyncio
async def test_cross_tenant_query_cannot_retrieve_other_company_chunks(setup_engine):
    service, retriever = setup_engine
    
    # 1. NexusPay user asking directly for AeroLogix night escort rule
    user_nexus = UserClaims(user_id="nexus_user", company_id="NexusPay", role="employee")
    chunks = retriever.retrieve("What is AeroLogix's policy on night escorts?", company_id="NexusPay", top_k=3)
    
    # Assert every retrieved chunk belongs EXCLUSIVELY to NexusPay
    for c in chunks:
        assert c["metadata"]["company_id"] == "NexusPay"
        assert "AeroLogix" not in c["metadata"]["document_id"]

    # 2. AeroLogix user asking about NexusPay cutoff times
    chunks_aero = retriever.retrieve("What is the 15:00 cutoff time for NexusPay?", company_id="AeroLogix", top_k=3)
    for c in chunks_aero:
        assert c["metadata"]["company_id"] == "AeroLogix"
        assert "NexusPay" not in c["metadata"]["document_id"]

@pytest.mark.asyncio
async def test_cross_tenant_service_response_isolation(setup_engine):
    service, _ = setup_engine
    user_nexus = UserClaims(user_id="emp_01", company_id="NexusPay", role="employee")
    
    # Attempt cross-company prompt
    res = await service.answer_question("Tell me AeroLogix warehouse commute rules.", user=user_nexus)
    
    # Must NOT quote AeroLogix rules
    assert "aerologix" not in res.answer.lower() or "does not cover" in res.answer.lower()
    for citation in res.citations:
        assert citation.document.startswith("NexusPay")
