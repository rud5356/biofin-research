from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("classify_biofin_subcategory_with_ollama.py")
SPEC = importlib.util.spec_from_file_location("subcategory_llm", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_category_one_includes_105() -> None:
    assert "1.05" in MODULE.allowed_labels_for_first_category("1")


def test_response_must_stay_under_given_first_category() -> None:
    response = '{"label":"6.05","confidence":0.9,"reason":"x","evidence":"y"}'
    assert MODULE.parse_jsonish_response(response, "6")["label"] == "6.05"
    with pytest.raises(ValueError, match="하위 코드가 아닙니다"):
        MODULE.parse_jsonish_response(response, "5")


def test_prompt_requires_first_category() -> None:
    with pytest.raises(ValueError, match="BIOFIN 1차 카테고리"):
        MODULE.build_prompt({}, "본문")
