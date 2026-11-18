import pytest
from pathlib import Path
from app.ingestion import IngestionEngine, parse_markdown_policy
from app.retriever import PolicyRetriever

def test_parse_markdown_policy():
    sample_content = """# Sample Corp Commute Policy (Version 1.0)
Document ID: Sample_Policy_v1
Tenant: SampleCorp

## 1. General Rules
1.1 Daily cabs run on schedules.

## 2. Night Drops
2.1 Escorts required after 20:00.
"""
    doc_id, version, company, chunks = parse_markdown_policy(sample_content, "SampleCorp", 1)
    assert doc_id == "Sample_Policy_v1"
    assert version == 1
    assert company == "SampleCorp"
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["section_title"] == "1. General Rules"
    assert chunks[1]["metadata"]["section_title"] == "2. Night Drops"

def test_atomic_version_upgrade():
    engine = IngestionEngine()
    
    # Ingest v1
    res1 = engine.ingest_policy_file("policies/nexuspay_v1.md", "NexusPay", 1)
    assert res1["version"] == 1
    assert res1["status"] == "active"

    retriever = PolicyRetriever(chroma_client=engine.client)
    chunks_v1 = retriever.retrieve("night escort rule", company_id="NexusPay", top_k=1)
    assert "NexusPay_Commute_Policy_v1" in chunks_v1[0]["metadata"]["document_id"]
    assert chunks_v1[0]["metadata"]["version"] == 1

    # Upgrade to v2
    res2 = engine.ingest_policy_file("policies/nexuspay_v2.md", "NexusPay", 2)
    assert res2["version"] == 2
    assert res2["status"] == "active"

    # Refresh and query again
    retriever.refresh_bm25_indices()
    chunks_v2 = retriever.retrieve("night escort rule", company_id="NexusPay", top_k=1)
    assert "NexusPay_Commute_Policy_v2" in chunks_v2[0]["metadata"]["document_id"]
    assert chunks_v2[0]["metadata"]["version"] == 2

    # Verify that NO active chunks remain with version == 1
    all_nexus = engine.collection.get(
        where={
            "$and": [
                {"company_id": {"$eq": "NexusPay"}},
                {"is_active": {"$eq": True}}
            ]
        }
    )
    for m in all_nexus["metadatas"]:
        assert m["version"] == 2

def test_failed_ingestion_does_not_corrupt_active_policy(tmp_path):
    engine = IngestionEngine()
    
    # Ensure NexusPay v2 is active
    engine.ingest_policy_file("policies/nexuspay_v2.md", "NexusPay", 2)
    
    # Create a corrupted policy file with no valid sections
    corrupted_file = tmp_path / "corrupted_policy.md"
    corrupted_file.write_text("Corrupted file without any markdown headers or sections", encoding="utf-8")

    # Attempting to ingest corrupted file should raise an exception
    with pytest.raises(Exception):
        engine.ingest_policy_file(str(corrupted_file), "NexusPay", 3)

    # Assert NexusPay still has active v2 chunks
    active = engine.collection.get(
        where={
            "$and": [
                {"company_id": {"$eq": "NexusPay"}},
                {"is_active": {"$eq": True}}
            ]
        }
    )
    assert len(active["ids"]) > 0
    for m in active["metadatas"]:
        assert m["version"] == 2
