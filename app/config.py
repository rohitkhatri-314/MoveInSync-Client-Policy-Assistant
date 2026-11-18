import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "MoveInSync Commute Policy Assistant"
    APP_ENV: str = os.getenv("APP_ENV", "development")
    
    # Security / JWT
    JWT_SECRET: str = os.getenv("JWT_SECRET", "moveinsync-campus-case-study-b-secure-jwt-key-2026")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 60 * 24  # 24 hours
    
    # Storage / Chroma
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma_db"))
    CHROMA_COLLECTION_NAME: str = "commute_policies"
    
    # LLM & Retrieval Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "fake")  # 'fake' by default for deterministic tests/Docker
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEFAULT_RETRIEVAL_MODE: str = os.getenv("DEFAULT_RETRIEVAL_MODE", "hybrid")  # 'dense' or 'hybrid'
    
    # Constraints
    MAX_CITED_CHUNKS: int = 3
    MAX_LLM_CALLS_PER_QUERY: int = 2
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "5.0" if os.getenv("LLM_PROVIDER") == "gemini" else "2.5"))
    RELEVANCE_THRESHOLD: float = 0.01  # Hybrid RRF score threshold for out-of-scope silence (RRF max is ~0.033)
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
