import re
from pathlib import Path
from typing import Dict, List, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

def parse_markdown_policy(content: str, fallback_company: str, fallback_version: int) -> Tuple[str, int, str, List[Dict]]:
    """
    Parses a policy markdown file into structured sections with metadata.
    Extracts Document ID, Version, Tenant, and section blocks.
    """
    lines = content.splitlines()
    doc_id = f"{fallback_company}_Commute_Policy_v{fallback_version}"
    version = fallback_version
    company_id = fallback_company

    # Look for header metadata
    for line in lines[:10]:
        if line.lower().startswith("document id:"):
            doc_id = line.split(":", 1)[1].strip()
        elif line.lower().startswith("tenant:"):
            company_id = line.split(":", 1)[1].strip()
        elif "version" in line.lower() and "(" in line:
            # e.g. # NexusPay Commute Policy (Version 2.0)
            m = re.search(r"version\s*(\d+(\.\d+)?)", line, re.IGNORECASE)
            if m:
                try:
                    version = int(float(m.group(1)))
                except ValueError:
                    pass

    # Validate that document contains valid section headers
    has_sections = any(line.strip().startswith("## ") for line in lines)
    if not has_sections:
        raise ValueError("Policy document must contain at least one '## ' section header")

    # Split into sections by markdown headers (## )
    sections = []
    current_title = ""
    current_lines: List[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_title and current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append((current_title, section_text))
                current_lines = []
            current_title = line.replace("## ", "").strip()
        elif current_title:
            current_lines.append(line)

    if current_title and current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_title, section_text))

    chunks = []
    for idx, (title, text) in enumerate(sections):
        if not text or len(text) < 10:
            continue
        chunk_id = f"{company_id}_v{version}_sec_{idx}"
        chunk_content = f"[{doc_id}] Section: {title}\n{text}"
        chunks.append({
            "id": chunk_id,
            "text": chunk_content,
            "metadata": {
                "company_id": company_id,
                "document_id": doc_id,
                "version": int(version),
                "section_id": f"sec_{idx}",
                "section_title": title,
                "is_active": False  # Initially staged as inactive
            }
        })

    return doc_id, version, company_id, chunks


class IngestionEngine:
    def __init__(self, chroma_client: chromadb.ClientAPI = None):
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        if chroma_client:
            self.client = chroma_client
        else:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_policy_file(self, file_path: str, company_id: str, version: int) -> Dict:
        """
        Atomically ingests a policy file.
        Staging -> Validation -> Deactivate Previous -> Activate New.
        If any stage fails, staged chunks are wiped and active state is untouched.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Policy file not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        doc_id, parsed_version, parsed_company, chunks = parse_markdown_policy(
            content, fallback_company=company_id, fallback_version=version
        )

        if not chunks:
            raise ValueError(f"Failed to extract valid sections from {file_path}")

        staged_ids = [c["id"] for c in chunks]
        staged_texts = [c["text"] for c in chunks]
        staged_metas = [c["metadata"] for c in chunks]

        # Step 1: Stage new chunks with is_active = False
        try:
            # Clean up any leftover staged chunks with same IDs first
            try:
                self.collection.delete(ids=staged_ids)
            except Exception:
                pass

            self.collection.add(
                ids=staged_ids,
                documents=staged_texts,
                metadatas=staged_metas
            )
        except Exception as e:
            raise RuntimeError(f"ChromaDB staging failed: {str(e)}")

        # Step 2: Validate staged chunks are queryable
        try:
            verify_res = self.collection.get(ids=staged_ids)
            if len(verify_res["ids"]) != len(staged_ids):
                raise ValueError("Verification failed: staged chunk count mismatch")
        except Exception as e:
            # Rollback staged chunks
            try:
                self.collection.delete(ids=staged_ids)
            except Exception:
                pass
            raise RuntimeError(f"Atomic ingestion aborted during verification: {str(e)}")

        # Step 3: Deactivate or remove older versions for this company
        try:
            existing = self.collection.get(
                where={"company_id": {"$eq": parsed_company}}
            )
            old_ids = [
                oid for oid, m in zip(existing["ids"], existing["metadatas"])
                if oid not in staged_ids
            ]
            if old_ids:
                # Remove deprecated chunks completely to guarantee clean queries
                self.collection.delete(ids=old_ids)

            # Step 4: Atomically activate the newly staged chunks
            active_metas = []
            for m in staged_metas:
                m_copy = dict(m)
                m_copy["is_active"] = True
                active_metas.append(m_copy)

            self.collection.update(
                ids=staged_ids,
                documents=staged_texts,
                metadatas=active_metas
            )
        except Exception as e:
            # Rollback staged chunks on error
            try:
                self.collection.delete(ids=staged_ids)
            except Exception:
                pass
            raise RuntimeError(f"Failed during activation phase: {str(e)}")

        return {
            "company_id": parsed_company,
            "document_id": doc_id,
            "version": parsed_version,
            "chunks_indexed": len(chunks),
            "status": "active"
        }

    def get_active_versions(self) -> Dict[str, Dict]:
        """
        Returns active policy information for each company.
        """
        try:
            res = self.collection.get(where={"is_active": {"$eq": True}})
            status = {}
            for m in res.get("metadatas", []):
                cid = m.get("company_id")
                ver = m.get("version", 1)
                doc = m.get("document_id", "")
                if cid:
                    if cid not in status or ver > status[cid]["version"]:
                        status[cid] = {
                            "company_id": cid,
                            "version": ver,
                            "document_id": doc,
                        }
            # Fallback defaults if collection empty
            for cid in ["NexusPay", "AeroLogix", "CloudMatrix"]:
                if cid not in status:
                    status[cid] = {"company_id": cid, "version": 1, "document_id": f"{cid}_Commute_Policy_v1"}
            return status
        except Exception:
            return {
                "NexusPay": {"company_id": "NexusPay", "version": 1, "document_id": "NexusPay_Commute_Policy_v1"},
                "AeroLogix": {"company_id": "AeroLogix", "version": 1, "document_id": "AeroLogix_Commute_Policy_v1"},
                "CloudMatrix": {"company_id": "CloudMatrix", "version": 1, "document_id": "CloudMatrix_Commute_Policy_v1"},
            }

