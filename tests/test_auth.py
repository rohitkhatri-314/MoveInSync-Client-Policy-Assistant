import datetime
import jwt
import pytest
from fastapi import HTTPException
from app.auth import create_access_token, decode_access_token
from app.config import settings

def test_token_creation_and_decoding():
    token = create_access_token(user_id="emp_test", company_id="NexusPay", role="employee")
    claims = decode_access_token(token)
    assert claims.user_id == "emp_test"
    assert claims.company_id == "NexusPay"
    assert claims.role == "employee"

def test_expired_token():
    # Create token with -5 minutes expiry
    now = datetime.datetime.now(datetime.timezone.utc)
    expired_payload = {
        "sub": "emp_expired",
        "company_id": "NexusPay",
        "role": "employee",
        "iat": now - datetime.timedelta(minutes=10),
        "exp": now - datetime.timedelta(minutes=5),
    }
    token = jwt.encode(expired_payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()

def test_tampered_token():
    token = create_access_token(user_id="emp_test", company_id="NexusPay")
    tampered_token = token[:-5] + "abcde"
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(tampered_token)
    assert exc_info.value.status_code == 401
    assert "invalid token" in exc_info.value.detail.lower()

def test_missing_company_id_claim():
    now = datetime.datetime.now(datetime.timezone.utc)
    invalid_payload = {
        "sub": "emp_no_company",
        "exp": now + datetime.timedelta(minutes=60),
    }
    token = jwt.encode(invalid_payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token)
    assert exc_info.value.status_code == 401
    assert "missing" in exc_info.value.detail.lower()
