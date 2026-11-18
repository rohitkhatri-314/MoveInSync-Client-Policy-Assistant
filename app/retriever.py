import re
from typing import Dict, List, Optional, Tuple
import chromadb
from rank_bm25 import BM25Okapi

from app.config import settings

def tokenize(text: str) -> List[str]:
    """Lightweight regex word tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())

class PolicyRetriever:
    def __init__(self, chroma_client: Optional[chromadb.ClientAPI] = None):
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        if chroma_client:
            self.client = chroma_client
        else:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        self._bm25_indices: Dict[str, BM25Okapi] = {}
        self._bm25_corpus: Dict[str, List[Dict]] = {}
        self.refresh_bm25_indices()

    def refresh_bm25_indices(self):
        """Builds in-memory BM25 inverted indices partitioned strictly by company_id."""
        try:
            all_records = self.collection.get()
            tenant_chunks: Dict[str, List[Dict]] = {}
            
            for doc_id, text, meta in zip(
                all_records["ids"], all_records["documents"], all_records["metadatas"]
            ):
                if meta.get("is_active"):
                    cid = meta.get("company_id")
                    if cid not in tenant_chunks:
                        tenant_chunks[cid] = []
                    tenant_chunks[cid].append({
                        "id": doc_id,
                        "text": text,
                        "metadata": meta
                    })
                    
            self._bm25_corpus = tenant_chunks
            self._bm25_indices = {}
            for cid, chunks in tenant_chunks.items():
                tokenized_corpus = [tokenize(c["text"]) for c in chunks]
                if tokenized_corpus:
                    self._bm25_indices[cid] = BM25Okapi(tokenized_corpus)
        except Exception as e:
            # If collection is empty, gracefully proceed
            pass

    def retrieve(
        self,
        query: str,
        company_id: str,
        top_k: int = 3,
        mode: Optional[str] = None
    ) -> List[Dict]:
        """
        Multi-tenant secure retrieval with strict server-enforced boundary.
        Enforces at most top_k (<= 3) chunks.
        Supports 'dense' or 'hybrid' (Dense + BM25 via Reciprocal Rank Fusion).
        """
        # Hard cap at max cited chunks constraint
        k = min(top_k, settings.MAX_CITED_CHUNKS)
        retrieval_mode = mode or settings.DEFAULT_RETRIEVAL_MODE

        # 1. Dense Vector Search (ChromaDB) strictly isolated by company_id
        dense_results = []
        try:
            res = self.collection.query(
                query_texts=[query],
                n_results=k * 2,  # candidate pool
                where={
                    "$and": [
                        {"company_id": {"$eq": company_id}},
                        {"is_active": {"$eq": True}}
                    ]
                }
            )
            
            if res["ids"] and res["ids"][0]:
                for doc_id, doc_text, meta, dist in zip(
                    res["ids"][0],
                    res["documents"][0],
                    res["metadatas"][0],
                    res["distances"][0] if "distances" in res and res["distances"] else [0.5] * len(res["ids"][0])
                ):
                    # Cosine distance to similarity: 1 - distance
                    sim = 1.0 - float(dist) if dist is not None else 0.5
                    dense_results.append({
                        "id": doc_id,
                        "text": doc_text,
                        "metadata": meta,
                        "dense_score": sim
                    })
        except Exception:
            dense_results = []

        if retrieval_mode == "dense" or company_id not in self._bm25_indices:
            # Sort by dense score descending
            dense_results.sort(key=lambda x: x.get("dense_score", 0), reverse=True)
            return dense_results[:k]

        # 2. Sparse Keyword Search (BM25) strictly isolated by company_id
        bm25_index = self._bm25_indices.get(company_id)
        corpus = self._bm25_corpus.get(company_id, [])
        query_tokens = tokenize(query)
        bm25_results = []

        if bm25_index and corpus and query_tokens:
            scores = bm25_index.get_scores(query_tokens)
            scored_corpus = []
            for item, score in zip(corpus, scores):
                if score > 0:
                    scored_corpus.append((score, item))
            scored_corpus.sort(key=lambda x: x[0], reverse=True)
            bm25_results = scored_corpus[:k * 2]

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score = sum(1 / (60 + rank))
        rrf_scores: Dict[str, float] = {}
        item_lookup: Dict[str, Dict] = {}

        # Dense ranks
        for rank, item in enumerate(dense_results, start=1):
            doc_id = item["id"]
            item_lookup[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))

        # BM25 ranks
        for rank, (score, item) in enumerate(bm25_results, start=1):
            doc_id = item["id"]
            if doc_id not in item_lookup:
                item_lookup[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))

        # Sort combined results by RRF score
        sorted_docs = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

        final_chunks = []
        for doc_id, rrf_score in sorted_docs[:k]:
            chunk_data = dict(item_lookup[doc_id])
            chunk_data["rrf_score"] = rrf_score
            final_chunks.append(chunk_data)

        return final_chunks
