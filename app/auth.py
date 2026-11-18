import datetime
from typing import Optional
import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.models import UserClaims

security = HTTPBearer(auto_error=False)

def create_access_token(
    user_id: str,
    company_id: str,
    role: str = "employee",
    expires_minutes: Optional[int] = None
) -> str:
    """Generate a signed JWT token containing tenant claims."""
    if expires_minutes is None:
        expires_minutes = settings.JWT_EXPIRY_MINUTES
        
    now = datetime.datetime.now(datetime.timezone.utc)
    expire = now + datetime.timedelta(minutes=expires_minutes)
    
    payload = {
        "sub": user_id,
        "company_id": company_id,
        "role": role,
        "iat": now,
        "exp": expire,
    }
    
    encoded_jwt = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> UserClaims:
    """Validate and decode the JWT token, strictly verifying signature and claims."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id = payload.get("sub")
        company_id = payload.get("company_id")
        role = payload.get("role", "employee")
        
        if not user_id or not company_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject or company_id claim",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return UserClaims(user_id=user_id, company_id=company_id, role=role)
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please authenticate again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials: invalid token signature or format",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_tenant(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> UserClaims:
    """FastAPI dependency to enforce tenant authorization on secure endpoints."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token in request header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_access_token(credentials.credentials)
