from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict


def call_vlm(model_cfg: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = _chat_url(model_cfg.base_url)
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if model_cfg.api_key:
        headers["Authorization"] = f"Bearer {model_cfg.api_key}"
    last_error: Exception | None = None
    for attempt in range(max(1, int(model_cfg.max_retries) + 1)):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=float(model_cfg.timeout)) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < int(model_cfg.max_retries):
                time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"VLM request failed: {last_error}")


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"
