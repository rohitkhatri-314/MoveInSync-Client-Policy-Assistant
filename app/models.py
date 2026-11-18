from typing import List, Optional
from pydantic import BaseModel, Field

class TokenRequest(BaseModel):
    company_id: str = Field(..., description="Target company/tenant (e.g., NexusPay, AeroLogix, CloudMatrix)")
    user_id: str = Field(default="emp_001", description="Employee identifier")
    role: str = Field(default="employee", description="Employee or transport admin role")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    company_id: str
    user_id: str
    expires_in_minutes: int

class UserClaims(BaseModel):
    user_id: str
    company_id: str
    role: str = "employee"

class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="Employee commute question")
    mode: Optional[str] = Field(default=None, description="Retrieval mode: 'dense' or 'hybrid' (defaults to system config)")
    provider: Optional[str] = Field(default=None, description="LLM provider: 'groq', 'gemini', or 'fake' (defaults to system config)")

class Citation(BaseModel):
    document: str = Field(..., description="Source policy document ID or name")
    section: str = Field(..., description="Section number and title")
    excerpt: str = Field(..., description="Relevant text excerpt providing grounding")

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    company_id: str
    latency_ms: float
    llm_calls: int
    chunks_retrieved: int
    retrieval_mode: str
    llm_provider: Optional[str] = None
    status: str  # 'success', 'silent_fallback', 'timeout_fallback', 'error_fallback'

class IngestRequest(BaseModel):
    company_id: str
    file_path: str
    version: int

class IngestResponse(BaseModel):
    company_id: str
    document_id: str
    version: int
    chunks_indexed: int
    status: str
