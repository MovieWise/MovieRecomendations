from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any
from urllib.parse import parse_qsl

try:
    from fastapi import HTTPException, status
except ModuleNotFoundError:
    class status:
        HTTP_401_UNAUTHORIZED = 401

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail


def _b64url_encode(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _b64url_decode(payload: str) -> bytes:
    padding = "=" * (-len(payload) % 4)
    return base64.urlsafe_b64decode(payload + padding)


def validate_telegram_init_data(
    init_data: str,
    bot_token: str,
    max_age_seconds: int = 86400,
    now: int | None = None,
) -> dict[str, Any]:
    if not bot_token:
        raise HTTPException(status_code=500, detail="telegram_bot_token_missing")

    pairs = dict(parse_qsl(init_data, keep_blank_values=True))
    received_hash = pairs.pop("hash", None)
    if not received_hash:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="telegram_hash_missing")

    data_check_string = "\n".join(f"{key}={pairs[key]}" for key in sorted(pairs))
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    calculated_hash = hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(calculated_hash, received_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="telegram_hash_invalid")

    auth_date = int(pairs.get("auth_date", "0") or "0")
    current_time = int(time.time()) if now is None else now
    if max_age_seconds > 0 and current_time - auth_date > max_age_seconds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="telegram_auth_expired")

    user_payload = pairs.get("user")
    if not user_payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="telegram_user_missing")
    try:
        user = json.loads(user_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="telegram_user_invalid") from exc
    if "id" not in user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="telegram_user_id_missing")
    return user


def create_access_token(payload: dict[str, Any], secret: str, ttl_seconds: int) -> str:
    now = int(time.time())
    body = {"iat": now, "exp": now + ttl_seconds, **payload}
    header_segment = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body_segment = _b64url_encode(json.dumps(body, separators=(",", ":")).encode())
    signing_input = f"{header_segment}.{body_segment}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_segment}.{body_segment}.{_b64url_encode(signature)}"


def decode_access_token(token: str, secret: str) -> dict[str, Any]:
    try:
        header_segment, body_segment, signature_segment = token.split(".")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token_invalid") from exc

    signing_input = f"{header_segment}.{body_segment}".encode("ascii")
    expected_signature = _b64url_encode(hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest())
    if not hmac.compare_digest(expected_signature, signature_segment):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token_invalid")

    try:
        payload = json.loads(_b64url_decode(body_segment))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token_invalid") from exc
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token_expired")
    return payload
