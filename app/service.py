import asyncio
import time
from typing import Optional
from app.config import settings
from app.llm import CONTACT_DESKS, get_llm
from app.models import AskResponse, Citation, UserClaims
from app.retriever import PolicyRetriever

class CommuteService:
    def __init__(self, retriever: Optional[PolicyRetriever] = None):
        self.retriever = retriever or PolicyRetriever()
        self.llm = get_llm()

    async def answer_question(
        self,
        question: str,
        user: UserClaims,
        mode: Optional[str] = None
    ) -> AskResponse:
        """
        Executes tenant-isolated query resolution under strict SLA and call limits.
        - Enforces server-side company_id from JWT.
        - Enforces max 3 cited chunks.
        - Enforces <= 2 LLM calls (1 on standard path, 0 on silence early-exit).
        - Enforces < 3.0s p95 latency.
        """
        start_time = time.perf_counter()
        company_id = user.company_id
        contact = CONTACT_DESKS.get(company_id, f"desk@{company_id.lower()}.com")
        retrieval_mode = mode or settings.DEFAULT_RETRIEVAL_MODE

        # 1. Multi-Tenant Retrieval (server-enforced company_id)
        chunks = self.retriever.retrieve(
            query=question,
            company_id=company_id,
            top_k=settings.MAX_CITED_CHUNKS,
            mode=retrieval_mode
        )

        # 2. Early-Exit Silence Detection (Zero-LLM optimization)
        # If no chunks found or top chunk has negligible relevance score
        is_silent = False
        if not chunks:
            is_silent = True
        elif retrieval_mode == "hybrid":
            top_score = chunks[0].get("rrf_score", 0.0)
            if top_score < settings.RELEVANCE_THRESHOLD:
                is_silent = True
        elif retrieval_mode == "dense":
            top_score = chunks[0].get("dense_score", 0.0)
            if top_score < 0.20:
                is_silent = True

        if is_silent:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            silence_text = (
                f"The commute policy for {company_id} does not cover this topic. "
                f"Please contact your company Transport Desk at {contact} for assistance."
            )
            return AskResponse(
                answer=silence_text,
                citations=[],
                company_id=company_id,
                latency_ms=elapsed_ms,
                llm_calls=0,
                chunks_retrieved=0,
                retrieval_mode=retrieval_mode,
                status="silent_fallback"
            )

        # 3. Grounded LLM Generation with SLA Circuit-Breaker
        try:
            # Wrap in async timeout guard
            answer, raw_citations, call_count = await asyncio.wait_for(
                asyncio.to_thread(self.llm.generate, question, company_id, chunks),
                timeout=settings.LLM_TIMEOUT_SECONDS
            )

            citations = [
                Citation(
                    document=c["document"],
                    section=c["section"],
                    excerpt=c["excerpt"]
                )
                for c in raw_citations[:settings.MAX_CITED_CHUNKS]
            ]

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            status_val = "silent_fallback" if "does not cover" in answer.lower() else "success"

            return AskResponse(
                answer=answer,
                citations=citations,
                company_id=company_id,
                latency_ms=elapsed_ms,
                llm_calls=call_count,
                chunks_retrieved=len(chunks),
                retrieval_mode=retrieval_mode,
                status=status_val
            )

        except asyncio.TimeoutError:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            timeout_msg = (
                f"The transport assistant encountered a latency timeout while querying {company_id}'s policy. "
                f"Please try again in a few moments or contact the Transport Desk directly at {contact}."
            )
            return AskResponse(
                answer=timeout_msg,
                citations=[],
                company_id=company_id,
                latency_ms=elapsed_ms,
                llm_calls=1,
                chunks_retrieved=len(chunks),
                retrieval_mode=retrieval_mode,
                status="timeout_fallback"
            )
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_msg = (
                f"An unexpected error occurred while processing your request. "
                f"Please try again or contact the Transport Desk at {contact}."
            )
            return AskResponse(
                answer=err_msg,
                citations=[],
                company_id=company_id,
                latency_ms=elapsed_ms,
                llm_calls=1,
                chunks_retrieved=0,
                retrieval_mode=retrieval_mode,
                status="error_fallback"
            )
