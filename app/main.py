from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.auth import create_access_token, get_current_tenant
from app.cache import query_cache
from app.config import settings
from app.ingestion import IngestionEngine
from app.models import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    TokenRequest,
    TokenResponse,
    UserClaims,
)
from app.retriever import PolicyRetriever
from app.service import CommuteService

# Global service singletons
ingestion_engine: IngestionEngine = None
policy_retriever: PolicyRetriever = None
commute_service: CommuteService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes ChromaDB, auto-ingests baseline policies, and prepares indices."""
    global ingestion_engine, policy_retriever, commute_service
    
    # 1. Initialize Ingestion and Retriever
    ingestion_engine = IngestionEngine()
    
    # 2. Seed baseline company policies if not already present
    policies_dir = settings.BASE_DIR / "policies"
    seed_files = [
        ("NexusPay", policies_dir / "nexuspay_v1.md", 1),
        ("AeroLogix", policies_dir / "aerologix_v1.md", 1),
        ("CloudMatrix", policies_dir / "cloudmatrix_v1.md", 1),
    ]
    
    for cid, file_path, ver in seed_files:
        if file_path.is_file():
            try:
                ingestion_engine.ingest_policy_file(str(file_path), cid, ver)
            except Exception as e:
                print(f"[WARN] Startup seed failed for {cid}: {e}")
                
    # 3. Instantiate retriever and commute service
    policy_retriever = PolicyRetriever(chroma_client=ingestion_engine.client)
    commute_service = CommuteService(retriever=policy_retriever)
    
    yield
    # Shutdown logic (if any)

app = FastAPI(
    title=settings.APP_NAME,
    description="Multi-tenant Commute Policy Assistant for MoveInSync with zero-leakage, grounded citations, and SLA enforcement.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def generate_token(req: TokenRequest):
    """
    Issues a signed JWT token bound to the specified company.
    In production, this is issued by corporate SSO / IDP.
    """
    token = create_access_token(
        user_id=req.user_id,
        company_id=req.company_id,
        role=req.role
    )
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        company_id=req.company_id,
        user_id=req.user_id,
        expires_in_minutes=settings.JWT_EXPIRY_MINUTES
    )

@app.post("/api/policies/ingest", response_model=IngestResponse, tags=["Admin / Policy Ingestion"])
async def ingest_policy(req: IngestRequest):
    """
    Atomically ingests a new version of a company's commute policy.
    Once activated, only the new version answers queries.
    """
    global ingestion_engine, policy_retriever
    if not ingestion_engine:
        ingestion_engine = IngestionEngine()
        
    try:
        res = ingestion_engine.ingest_policy_file(req.file_path, req.company_id, req.version)
        # Refresh BM25 sparse index and invalidate query cache after new ingestion
        if policy_retriever:
            policy_retriever.refresh_bm25_indices()
        query_cache.invalidate_tenant(req.company_id)
        return IngestResponse(
            company_id=res["company_id"],
            document_id=res["document_id"],
            version=res["version"],
            chunks_indexed=res["chunks_indexed"],
            status="active"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {str(e)}")

@app.post("/api/ask", response_model=AskResponse, tags=["Commute Assistant"])
async def ask_question(
    req: AskRequest,
    user: UserClaims = Depends(get_current_tenant)
):
    """
    Answers an employee question strictly grounded in the authenticated company's policy.
    Multi-tenancy is enforced at the server layer via the validated JWT claims.
    Client cannot supply or override company_id.
    """
    global commute_service
    if not commute_service:
        commute_service = CommuteService()
        
    return await commute_service.answer_question(
        question=req.question,
        user=user,
        mode=req.mode
    )

@app.get("/api/health", tags=["System"])
async def health_check():
    """Returns system status, active collection count, and SLA configurations."""
    global ingestion_engine
    count = 0
    if ingestion_engine:
        try:
            count = ingestion_engine.collection.count()
        except Exception:
            pass
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "indexed_chunks": count,
        "max_cited_chunks": settings.MAX_CITED_CHUNKS,
        "max_llm_calls": settings.MAX_LLM_CALLS_PER_QUERY,
        "sla_timeout_seconds": settings.LLM_TIMEOUT_SECONDS,
        "default_retrieval_mode": settings.DEFAULT_RETRIEVAL_MODE
    }

# Mount static files for modern web console
static_path = settings.BASE_DIR / "static"
if static_path.is_dir():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
