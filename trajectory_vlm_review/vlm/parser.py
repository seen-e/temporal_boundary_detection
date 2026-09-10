from __future__ import annotations

import json
import re
from typing import Any, Dict


def parse_vlm_response(response: Dict[str, Any]) -> Dict[str, Any]:
    content = response
    if isinstance(response, dict) and "choices" in response:
        content = response["choices"][0]["message"].get("content", "")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ValueError("VLM response content is not a string or JSON object")
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)
