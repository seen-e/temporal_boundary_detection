from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..core.models import ReviewSlice
from ..prompts.image_instructions import GLOBAL_IMAGE_INSTRUCTION, LOCAL_IMAGE_INSTRUCTION
from ..prompts import system_prompt, user_prompt
from ..prompts.three_stage_image_instructions import (
    PASS1_GLOBAL_IMAGE_INSTRUCTION,
    PASS1_LOCAL_IMAGE_INSTRUCTION,
    PASS2_LOCAL_IMAGE_INSTRUCTION,
    PASS3_GLOBAL_IMAGE_INSTRUCTION,
    PASS3_LOCAL_IMAGE_INSTRUCTION,
)
from ..prompts.three_stage_prompts import (
    COMMON_CONCEPTS_PROMPT,
    PASS1_SYSTEM_PROMPT,
    PASS1_USER_PROMPT,
    PASS2_SYSTEM_PROMPT,
    PASS2_USER_PROMPT,
    PASS3_SYSTEM_PROMPT,
    PASS3_USER_PROMPT,
)


TASK_INSTRUCTION = """<task>
请按以下顺序审核当前单夹爪轨迹：

1. 先看 GLOBAL，建立当前 LOCAL 的宏观背景；
2. 再看 LOCAL，划分局部轨迹结构；
3. 再结合 metadata 审核 target keyframes 和 owned intervals；
4. 最后严格按照给定 JSON schema 输出结果。
</task>"""


def build_messages(review_slice: ReviewSlice) -> List[Dict[str, Any]]:
    if not review_slice.image_path:
        raise ValueError("review_slice.image_path is required")
    if not review_slice.global_image_path:
        raise ValueError("review_slice.global_image_path is required")
    global_image_data = base64.b64encode(Path(review_slice.global_image_path).read_bytes()).decode("ascii")
    local_image_data = base64.b64encode(Path(review_slice.image_path).read_bytes()).decode("ascii")
    metadata_text, review_rules_text, output_schema_text = render_user_prompt_parts(review_slice)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TASK_INSTRUCTION},
                {"type": "text", "text": GLOBAL_IMAGE_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{global_image_data}"}},
                {"type": "text", "text": LOCAL_IMAGE_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{local_image_data}"}},
                {"type": "text", "text": metadata_text},
                {"type": "text", "text": review_rules_text},
                {"type": "text", "text": output_schema_text},
            ],
        },
    ]


def build_payload(model_cfg: Any, review_slice: ReviewSlice) -> Dict[str, Any]:
    return _payload(model_cfg, build_messages(review_slice))


def build_pass1_payload(model_cfg: Any, review_slice: ReviewSlice) -> Dict[str, Any]:
    if not review_slice.image_path:
        raise ValueError("pass1 review_slice.image_path is required")
    if not review_slice.global_image_path:
        raise ValueError("pass1 review_slice.global_image_path is required")
    global_image_data = base64.b64encode(Path(review_slice.global_image_path).read_bytes()).decode("ascii")
    local_image_data = base64.b64encode(Path(review_slice.image_path).read_bytes()).decode("ascii")
    context = {
        "episode_id": review_slice.episode_id,
        "episode_index": review_slice.episode_index,
        "arm": review_slice.arm,
        "slice_id": review_slice.slice_id,
        "time_range_sec": [round(review_slice.time_range[0], 4), round(review_slice.time_range[1], 4)],
        "target_time_range_sec": [round(review_slice.targets[0].time, 4), round(review_slice.targets[-1].time, 4)] if review_slice.targets else None,
        "has_left_context": bool(review_slice.context_left),
        "has_right_context": bool(review_slice.context_right),
    }
    context_text = _xml_json_block("context_without_candidates", context)
    messages = [
        {"role": "system", "content": _system_content(PASS1_SYSTEM_PROMPT)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Pass 1：先看 GLOBAL，再看 LOCAL；当前阶段不显示候选关键帧。"},
                {"type": "text", "text": PASS1_GLOBAL_IMAGE_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{global_image_data}"}},
                {"type": "text", "text": PASS1_LOCAL_IMAGE_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{local_image_data}"}},
                {"type": "text", "text": context_text},
                {"type": "text", "text": _task_prompt(PASS1_USER_PROMPT, ["context_without_candidates"])},
            ],
        },
    ]
    return _payload(model_cfg, messages)


def build_pass2_payload(model_cfg: Any, review_slice: ReviewSlice, pass1_review: Dict[str, Any]) -> Dict[str, Any]:
    if not review_slice.image_path:
        raise ValueError("pass2 review_slice.image_path is required")
    local_image_data = base64.b64encode(Path(review_slice.image_path).read_bytes()).decode("ascii")
    frozen_segments = (pass1_review.get("region") or {}).get("segments") or []
    metadata = review_slice.metadata()
    context_text = "\n\n".join(
        [
            _xml_json_block("frozen_segments", frozen_segments),
            _xml_json_block("local_metadata", metadata),
        ]
    )
    messages = [
        {"role": "system", "content": _system_content(PASS2_SYSTEM_PROMPT)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Pass 2：Pass 1 segments 已冻结；请根据整个 LOCAL 的 candidates 选择真实结构边界。"},
                {"type": "text", "text": PASS2_LOCAL_IMAGE_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{local_image_data}"}},
                {"type": "text", "text": context_text},
                {"type": "text", "text": _task_prompt(PASS2_USER_PROMPT, ["frozen_segments", "metadata"])},
            ],
        },
    ]
    return _payload(model_cfg, messages)


def build_pass3_payload(model_cfg: Any, gap_metadata: Dict[str, Any]) -> Dict[str, Any]:
    global_image_path = gap_metadata.get("global_image_path")
    local_image_path = gap_metadata.get("local_image_path")
    if not global_image_path:
        raise ValueError("pass3 gap_metadata.global_image_path is required")
    if not local_image_path:
        raise ValueError("pass3 gap_metadata.local_image_path is required")
    global_image_data = base64.b64encode(Path(global_image_path).read_bytes()).decode("ascii")
    local_image_data = base64.b64encode(Path(local_image_path).read_bytes()).decode("ascii")
    metadata = dict(gap_metadata)
    metadata.pop("global_image_path", None)
    metadata.pop("local_image_path", None)
    context_text = "\n\n".join(
        [
            _xml_json_block("frozen_segments", metadata.get("frozen_segments", [])),
            _xml_json_block("pass2_segments", metadata.get("pass2_segments", [])),
            _xml_json_block("cleaned_keyframes", metadata.get("cleaned_keyframes", [])),
            _xml_json_block("internal_boundaries", metadata.get("internal_boundaries", [])),
            _xml_json_block("boundaries_to_check", metadata.get("boundaries_to_check", [])),
        ]
    )
    messages = [
        {"role": "system", "content": _system_content(PASS3_SYSTEM_PROMPT)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Pass 3：检查 Pass 1 已冻结的 internal boundaries 是否已被 Pass 2 cleaned keyframes 覆盖。"},
                {"type": "text", "text": PASS3_GLOBAL_IMAGE_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{global_image_data}"}},
                {"type": "text", "text": PASS3_LOCAL_IMAGE_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{local_image_data}"}},
                {"type": "text", "text": context_text},
                {"type": "text", "text": _task_prompt(PASS3_USER_PROMPT, ["frozen_segments", "cleaned_keyframes", "internal_boundaries", "boundaries_to_check"])},
            ],
        },
    ]
    return _payload(model_cfg, messages)


def _system_content(pass_system_prompt: str) -> str:
    return f"{COMMON_CONCEPTS_PROMPT.strip()}\n\n{pass_system_prompt.strip()}"


def _xml_json_block(tag: str, value: Any) -> str:
    return f"<{tag}>\n{json.dumps(value, ensure_ascii=False, indent=2)}\n</{tag}>"


def _task_prompt(prompt: str, remove_blocks: Iterable[str]) -> str:
    cleaned = prompt
    for block in remove_blocks:
        cleaned = _remove_xml_block(cleaned, block)
    for placeholder in (
        "{{FROZEN_SEGMENTS_JSON}}",
        "{{TARGET_METADATA_JSON}}",
        "{{LOCAL_METADATA_JSON}}",
        "{{CLEANED_KEYFRAMES_JSON}}",
        "{{INTERNAL_BOUNDARIES_JSON}}",
        "{{BOUNDARIES_TO_CHECK_JSON}}",
    ):
        cleaned = cleaned.replace(placeholder, "")
    return cleaned.strip()


def summarize_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"messages": []}
    for message in messages:
        item: Dict[str, Any] = {"role": message.get("role")}
        content = message.get("content")
        if isinstance(content, list):
            item["content"] = [_summarize_part(idx, part) for idx, part in enumerate(content, start=1)]
        else:
            text = str(content or "")
            item["content"] = {
                "type": "text",
                "chars": len(text),
                "labels": _text_labels(text),
            }
        summary["messages"].append(item)
    return summary


def _summarize_part(idx: int, part: Dict[str, Any]) -> Dict[str, Any]:
    if part.get("type") == "image_url":
        return {"index": idx, "type": "image_url", "image_url": "<base64 image omitted>"}
    text = str(part.get("text", ""))
    return {"index": idx, "type": part.get("type"), "chars": len(text), "labels": _text_labels(text)}


def _text_labels(text: str) -> List[str]:
    checks = [
        ("COMMON_CONCEPTS_PROMPT", "<common_concepts>"),
        ("PASS1_SYSTEM_PROMPT", "你是机器人单夹爪轨迹结构分析器"),
        ("PASS2_SYSTEM_PROMPT", "你是机器人单夹爪轨迹关键帧边界选择器"),
        ("PASS3_SYSTEM_PROMPT", "你是机器人单夹爪轨迹缺失边界检查器"),
        ("PASS1_GLOBAL_IMAGE_INSTRUCTION", "<pass1_global_image_instruction>"),
        ("PASS1_LOCAL_IMAGE_INSTRUCTION", "<pass1_local_image_instruction>"),
        ("PASS2_LOCAL_IMAGE_INSTRUCTION", "<pass2_local_image_instruction>"),
        ("PASS3_GLOBAL_IMAGE_INSTRUCTION", "<pass3_global_image_instruction>"),
        ("PASS3_LOCAL_IMAGE_INSTRUCTION", "<pass3_local_image_instruction>"),
        ("CONTEXT_WITHOUT_CANDIDATES", "<context_without_candidates>"),
        ("FROZEN_SEGMENTS", "<frozen_segments>"),
        ("LOCAL_METADATA", "<local_metadata>"),
        ("PASS2_SEGMENTS", "<pass2_segments>"),
        ("CLEANED_KEYFRAMES", "<cleaned_keyframes>"),
        ("INTERNAL_BOUNDARIES", "<internal_boundaries>"),
        ("BOUNDARIES_TO_CHECK", "<boundaries_to_check>"),
        ("PASS1_USER_PROMPT", "请完成 Pass 1：Structure Analysis"),
        ("PASS2_USER_PROMPT", "请完成 Pass 2：Candidate Selection"),
        ("PASS3_USER_PROMPT", "请完成 Pass 3：Frozen Boundary Coverage Check"),
    ]
    return [name for name, marker in checks if marker in text]


def _payload(model_cfg: Any, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model_cfg.model,
        "messages": messages,
        "max_tokens": model_cfg.max_tokens,
        "temperature": model_cfg.temperature,
    }
    if model_cfg.top_p is not None:
        payload["top_p"] = model_cfg.top_p
    if model_cfg.top_k is not None:
        payload["top_k"] = model_cfg.top_k
    if model_cfg.extra_body:
        payload.update(model_cfg.extra_body)
    return payload


def render_user_prompt(review_slice: ReviewSlice) -> str:
    metadata_text, review_rules_text, output_schema_text = render_user_prompt_parts(review_slice)
    return "\n\n".join([TASK_INSTRUCTION, metadata_text, review_rules_text, output_schema_text])


def render_user_prompt_parts(review_slice: ReviewSlice) -> Tuple[str, str, str]:
    review_rules_text, output_schema_text = split_user_prompt_template(user_prompt)
    metadata_text = render_metadata_text(review_slice)
    output_schema_text = output_schema_text.replace(
        "{{OUTPUT_SCHEMA_JSON}}",
        json.dumps(output_schema_example(review_slice), ensure_ascii=False, indent=2),
    )
    return metadata_text, review_rules_text, output_schema_text


def render_metadata_text(review_slice: ReviewSlice) -> str:
    return (
        "<metadata>\n"
        f"{json.dumps(review_slice.metadata(), ensure_ascii=False, indent=2)}\n"
        "</metadata>"
    )


def split_user_prompt_template(prompt_template: str) -> Tuple[str, str]:
    without_task = _remove_xml_block(prompt_template, "task")
    without_metadata = _remove_xml_block(without_task, "metadata")
    output_start = without_metadata.find("<output_format>")
    if output_start < 0:
        output_start = without_metadata.find("<output_schema>")
    if output_start < 0:
        return without_metadata.strip(), (
            "<output_schema>\n{{OUTPUT_SCHEMA_JSON}}\n</output_schema>\n\n"
            "<final_output_instruction>\n"
            "严格按照上述 schema 输出 JSON。\n"
            "不要输出 Markdown 或 JSON 外文本。\n"
            "</final_output_instruction>"
        )
    review_rules_text = without_metadata[:output_start].strip()
    output_schema_text = without_metadata[output_start:].strip()
    return review_rules_text, output_schema_text


def _remove_xml_block(text: str, tag: str) -> str:
    start_marker = f"<{tag}>"
    end_marker = f"</{tag}>"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end < start:
        return text
    return (text[:start] + text[end + len(end_marker) :]).strip()


def output_schema_example(review_slice: ReviewSlice) -> Dict[str, Any]:
    target_ids = review_slice.target_display_ids or ["001"]
    interval_ids = [interval.display_interval_id for interval in review_slice.owned_intervals] or ["001_002"]
    return {
        "region": {
            "adjacent_regions": {
                "left": {"description": "左侧上下文整体稳定/上升/下降及波动情况；无左侧上下文时填 null。"},
                "right": {"description": "右侧上下文整体稳定/上升/下降及波动情况；无右侧上下文时填 null。"},
            },
            "description": "当前黄色 target region 的整体轨迹结构。",
            "keyframes": [
                {
                    "id": event_id,
                    "before": "该点之前的局部轨迹描述。",
                    "point": "该候选关键帧自身的形态特点。",
                    "after": "该点之后的局部轨迹描述。",
                }
                for event_id in target_ids
            ],
            "segments": [
                {
                    "id": "001",
                    "type": "稳定关闭-带波动",
                    "description": "该 segment 的最大连续轨迹结构描述。",
                    "events": target_ids[:2] if len(target_ids) >= 2 else target_ids,
                }
            ],
        },
        "additions": [
            {
                "interval": interval_ids[0],
                "type": "平台右端点",
                "approx_time_sec": None,
                "description": "仅当已有候选关键帧无法表达必要结构边界时填写；否则 additions 为空数组。",
            }
        ] if interval_ids else [],
        "needs_review": False,
    }
