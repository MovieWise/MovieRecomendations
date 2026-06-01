from __future__ import annotations

import asyncio
import json
from urllib.parse import urlencode
from urllib.request import urlopen


class OmdbClient:
    def __init__(self, api_key: str, base_url: str = "https://www.omdbapi.com/", timeout: float = 5.0) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    async def fetch_by_imdb_id(self, imdb_id: str) -> dict | None:
        if not self.api_key or not imdb_id:
            return None
        query = urlencode({"i": imdb_id, "apikey": self.api_key, "plot": "short"})
        url = f"{self.base_url}?{query}"
        return await asyncio.to_thread(self._fetch_json, url)

    def _fetch_json(self, url: str) -> dict | None:
        with urlopen(url, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("Response") != "True":
            return None
        return payload

