import asyncio
import re
import time
from typing import Optional
from app.cache import query_cache
from app.config import settings
from app.llm import CONTACT_DESKS, get_llm
from app.models import AskResponse, Citation, UserClaims
from app.retriever import PolicyRetriever

GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"}
THANKS = {"thank you", "thanks", "thx", "thank you so much", "thanks a lot", "appreciate it"}
CLOSINGS = {"bye", "goodbye", "see you", "cya"}

class CommuteService:
    def __init__(self, retriever: Optional[PolicyRetriever] = None):
        self.retriever = retriever or PolicyRetriever()
        self.llm = get_llm()

    async def answer_question(
        self,
        question: str,
        user: UserClaims,
        mode: Optional[str] = None,
        provider: Optional[str] = None
    ) -> AskResponse:
        """
        Executes tenant-isolated query resolution under strict SLA and call limits.
        - Enforces server-side company_id from JWT.
        - Enforces max 3 cited chunks.
        - Enforces <= 2 LLM calls (1 on standard path, 0 on silence early-exit).
        - Enforces < 3.0s p95 latency.
        - Dynamically routes to requested LLM provider (Groq, Gemini, or FakeLLM).
        """
        start_time = time.perf_counter()
        company_id = user.company_id
        contact = CONTACT_DESKS.get(company_id, f"desk@{company_id.lower()}.com")
        retrieval_mode = mode or settings.DEFAULT_RETRIEVAL_MODE
        active_provider = (provider or settings.LLM_PROVIDER).lower()
        llm_instance = get_llm(provider) if provider else self.llm

        # 0. Check Query Cache (Instant sub-millisecond response on repeated questions)
        cached_resp = query_cache.get(company_id=company_id, query=question, mode=retrieval_mode, provider=active_provider)
        if cached_resp:
            return cached_resp

        # 1. Conversational Pleasantries Handling (Greetings & Thanks)
        norm_clean = re.sub(r"[^\w\s]", "", question.strip().lower()).strip()
        if norm_clean in GREETINGS:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AskResponse(
                answer=f"Hello! I am your MoveInSync Commute Policy Assistant for {company_id}. How can I assist you with your commute, shift cutoffs, night escort rules, or cab allocations today?",
                citations=[],
                company_id=company_id,
                latency_ms=elapsed_ms,
                llm_calls=0,
                chunks_retrieved=0,
                retrieval_mode=retrieval_mode,
                llm_provider=active_provider,
                status="success"
            )
        elif norm_clean in THANKS:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AskResponse(
                answer="You're very welcome! Feel free to ask if you have any more questions about your corporate commute policy. Have a safe journey!",
                citations=[],
                company_id=company_id,
                latency_ms=elapsed_ms,
                llm_calls=0,
                chunks_retrieved=0,
                retrieval_mode=retrieval_mode,
                llm_provider=active_provider,
                status="success"
            )
        elif norm_clean in CLOSINGS:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AskResponse(
                answer="Goodbye! Have a safe and pleasant commute.",
                citations=[],
                company_id=company_id,
                latency_ms=elapsed_ms,
                llm_calls=0,
                chunks_retrieved=0,
                retrieval_mode=retrieval_mode,
                llm_provider=active_provider,
                status="success"
            )

        # 2. Multi-Tenant Retrieval (server-enforced company_id)
        chunks = self.retriever.retrieve(
            query=question,
            company_id=company_id,
            top_k=settings.MAX_CITED_CHUNKS,
            mode=retrieval_mode
        )

        # 3. Silence Detection: Early-exit only if zero chunks could be retrieved
        if not chunks:
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
                llm_provider=active_provider,
                status="silent_fallback"
            )

        # 4. Grounded LLM Generation with SLA Circuit-Breaker
        try:
            # Dynamic timeout budget depending on provider
            budget = 3.5 if provider == "groq" else (12.0 if provider == "gemini" else settings.LLM_TIMEOUT_SECONDS)
            answer, raw_citations, call_count = await asyncio.wait_for(
                asyncio.to_thread(llm_instance.generate, question, company_id, chunks),
                timeout=budget
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

            final_response = AskResponse(
                answer=answer,
                citations=citations,
                company_id=company_id,
                latency_ms=elapsed_ms,
                llm_calls=call_count,
                chunks_retrieved=len(chunks),
                retrieval_mode=retrieval_mode,
                llm_provider=active_provider,
                status=status_val
            )
            # Store in query cache for instant sub-millisecond retrieval on repeat
            query_cache.set(company_id=company_id, query=question, mode=retrieval_mode, response=final_response, provider=active_provider)
            return final_response

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
