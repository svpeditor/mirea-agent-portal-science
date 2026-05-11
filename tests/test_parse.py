"""Unit-тесты на чистые helpers science-agent — без сети, без LLM."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent as science  # noqa: E402


def test_strip_reasoning_removes_think_block() -> None:
    raw = "<think>думаю</think>\n[{\"i\":0}]"
    assert science._strip_reasoning(raw) == "[{\"i\":0}]"


def test_strip_reasoning_removes_json_fence() -> None:
    raw = "```json\n[{\"a\":1}]\n```"
    assert science._strip_reasoning(raw) == "[{\"a\":1}]"


def test_parse_json_extracts_array_from_noise() -> None:
    raw = "Вот результат:\n[{\"title\":\"X\",\"score\":0.9}]\nнадеюсь подойдёт"
    parsed = science._parse_json(raw)
    assert isinstance(parsed, list)
    assert parsed[0]["title"] == "X"


def test_parse_json_handles_code_fence_and_think() -> None:
    raw = "<think>...</think>\n```json\n[{\"i\":3}]\n```"
    assert science._parse_json(raw) == [{"i": 3}]


def test_paper_bibkey_is_safe() -> None:
    p = science.Paper(
        title="Attention Is All You Need",
        authors=["Vaswani A.", "Shazeer N."],
        year=2017,
        venue="NeurIPS",
        arxiv_id="1706.03762",
        url=None,
        annotation="",
    )
    key = p.bibkey
    assert key.startswith("vaswani")
    assert "2017" in key
    # Ключ должен быть пригоден как идентификатор BibTeX (только [a-z0-9_])
    assert all(c.isalnum() or c == "_" for c in key)


def test_paper_bibkey_unknown_author() -> None:
    p = science.Paper(
        title="X", authors=[], year=None, venue="", arxiv_id=None, url=None, annotation=""
    )
    key = p.bibkey
    assert key.startswith("anon")
    assert "nd" in key  # n.d. = no date


def test_build_bibtex_minimal() -> None:
    p = science.Paper(
        title="Hello", authors=["Иванов И."], year=2024, venue="arXiv",
        arxiv_id="2401.01234", url="https://arxiv.org/abs/2401.01234",
        annotation="aaa",
    )
    bib = science._build_bibtex([p])
    assert "@article{" in bib
    assert "title = {Hello}" in bib
    assert "year = {2024}" in bib
    assert "eprint = {arxiv:2401.01234}" in bib
    assert "url = {https://arxiv.org/abs/2401.01234}" in bib


def test_build_bibtex_no_year() -> None:
    p = science.Paper(
        title="X", authors=["A B"], year=None, venue="", arxiv_id=None, url=None,
        annotation="",
    )
    assert "year = {n.d.}" in science._build_bibtex([p])


def test_build_report_includes_topic_and_warning() -> None:
    papers = [
        science.Paper(
            title="P1", authors=["A B"], year=2020, venue="ICLR",
            arxiv_id="2001.0001", url=None, annotation="ann1", score=0.9,
        )
    ]
    doc = science._build_report("nlp", papers, "deepseek/deepseek-r1")
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Тема: nlp" in text
    assert "deepseek/deepseek-r1" in text
    # Warning о галлюцинациях обязательно
    assert "Перепроверьте" in text or "перепроверьте" in text.lower()
