import hashlib
import hmac
import json
from urllib.parse import urlencode

from recommendation_service.core.security import create_access_token, decode_access_token, validate_telegram_init_data


def build_init_data(bot_token: str, user: dict, auth_date: int = 1000) -> str:
    pairs = {
        "auth_date": str(auth_date),
        "query_id": "abc",
        "user": json.dumps(user, separators=(",", ":")),
    }
    data_check = "\n".join(f"{key}={pairs[key]}" for key in sorted(pairs))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    pairs["hash"] = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()
    return urlencode(pairs)


def test_validate_telegram_init_data():
    token = "123:secret"
    init_data = build_init_data(token, {"id": 42, "first_name": "Misha"})
    user = validate_telegram_init_data(init_data, token, max_age_seconds=0)
    assert user["id"] == 42


def test_access_token_roundtrip():
    token = create_access_token({"sub": "7"}, "secret", ttl_seconds=60)
    payload = decode_access_token(token, "secret")
    assert payload["sub"] == "7"

