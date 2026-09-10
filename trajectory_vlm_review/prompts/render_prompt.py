from __future__ import annotations

from ..core.models import ReviewSlice
from ..vlm.request_builder import render_user_prompt
from .vlm_review_prompt import system_prompt as system_prompt_template


def system_prompt() -> str:
    return system_prompt_template


def user_prompt(review_slice: ReviewSlice) -> str:
    return render_user_prompt(review_slice)
