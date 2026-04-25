"""Runtime patches for third-party ASR serving dependencies."""

from __future__ import annotations

from functools import wraps
from typing import Any


def _patch_transformers_mistral_regex() -> None:
    try:
        from transformers import AutoProcessor, AutoTokenizer
    except Exception:
        return

    def with_mistral_regex_fix(func: Any) -> Any:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("fix_mistral_regex", True)
            try:
                return func(*args, **kwargs)
            except TypeError as exc:
                if "fix_mistral_regex" not in str(exc):
                    raise
                kwargs.pop("fix_mistral_regex", None)
                return func(*args, **kwargs)

        return wrapped

    AutoTokenizer.from_pretrained = with_mistral_regex_fix(
        AutoTokenizer.from_pretrained
    )
    AutoProcessor.from_pretrained = with_mistral_regex_fix(
        AutoProcessor.from_pretrained
    )


_patch_transformers_mistral_regex()
