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


def test_strip_jats_removes_xml_tags() -> None:
    raw = "<jats:p>Реальный <jats:italic>abstract</jats:italic> текст</jats:p>"
    assert science._strip_jats(raw) == "Реальный  abstract  текст"


def test_paper_bibkey_is_safe() -> None:
    p = science.Paper(
        title="Attention Is All You Need", authors=["Vaswani A.", "Shazeer N."],
        year=2017, venue="NeurIPS", arxiv_id="1706.03762", url=None, annotation="",
    )
    key = p.bibkey
    assert key.startswith("vaswani")
    assert "2017" in key
    assert all(c.isalnum() or c == "_" for c in key)


def test_paper_bibkey_unknown_author() -> None:
    p = science.Paper(
        title="X", authors=[], year=None, venue="", arxiv_id=None, url=None, annotation=""
    )
    key = p.bibkey
    assert key.startswith("anon")
    assert "nd" in key


def test_best_link_prefers_arxiv_then_doi_then_url() -> None:
    assert science.Paper("T", [], None, "", "2401.001", "u", "").best_link == \
        "https://arxiv.org/abs/2401.001"
    assert science.Paper("T", [], None, "", None, "u", "", doi="10.1/x").best_link == \
        "https://doi.org/10.1/x"
    assert science.Paper("T", [], None, "", None, "http://j/x", "").best_link == \
        "http://j/x"
    assert science.Paper("T", [], None, "", None, None, "").best_link is None


def test_merge_dedupes_by_doi_and_unions_provenance() -> None:
    a = science.Paper("Цифровой след", ["И."], 2022, "Жур", None, None, "",
                       doi="10.1/x", provenance=["Crossref"])
    b = science.Paper("Цифровой след", ["И.", "П."], 2022, "Жур", "2201.001", None,
                       "длинный abstract", doi="10.1/x", citation_count=5,
                       provenance=["Semantic Scholar"])
    merged = science._merge([[a], [b]])
    assert len(merged) == 1
    m = merged[0]
    assert set(m.provenance) == {"Crossref", "Semantic Scholar"}
    assert m.arxiv_id == "2201.001"  # добили из дубликата
    assert m.citation_count == 5
    assert m.annotation == "длинный abstract"


def test_merge_dedupes_by_normalized_title_when_no_doi() -> None:
    a = science.Paper("Machine Learning!", [], 2020, "", None, None, "",
                       provenance=["arXiv"])
    b = science.Paper("machine  learning", [], 2020, "", None, None, "",
                       provenance=["Crossref"])
    merged = science._merge([[a], [b]])
    assert len(merged) == 1
    assert set(merged[0].provenance) == {"arXiv", "Crossref"}


def test_build_report_real_mode_has_no_hallucination_warning() -> None:
    """Реальный режим: статьи настоящие — НИКАКОГО общего дисклеймера про
    выдумки. Есть строка про реальные базы + кликабельная ссылка."""
    papers = [science.Paper(
        title="P1", authors=["A B"], year=2020, venue="ICLR",
        arxiv_id="2001.0001", url=None, annotation="ann1", score=0.9,
        provenance=["arXiv"],
    )]
    doc = science._build_report("nlp", papers, "deepseek/deepseek-r1", unverified=False)
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Тема: nlp" in text
    assert "deepseek/deepseek-r1" in text
    assert "реальные записи из баз" in text
    assert "https://arxiv.org/abs/2001.0001" in text
    assert "НЕ ПРОВЕРЕН" not in text  # никакой паники в реальном режиме


def test_build_report_llm_mode_has_explicit_unverified_warning() -> None:
    """Явный LLM-режим: обязателен крупный warning + пометка у каждой статьи."""
    papers = [science.Paper(
        title="Возможно выдуманная", authors=[], year=2021, venue="",
        arxiv_id=None, url=None, annotation="", score=0.5,
        provenance=["LLM (не проверено)"], unverified=True,
    )]
    doc = science._build_report("тема", papers, "m", unverified=True)
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "НЕ ПРОВЕРЕН" in text
    assert "Перепроверьте" in text
    assert "НЕ ПРОВЕРЕНО — предложено моделью" in text


def test_build_bibtex_minimal() -> None:
    p = science.Paper(
        title="Hello", authors=["Иванов И."], year=2024, venue="arXiv",
        arxiv_id="2401.01234", url="https://arxiv.org/abs/2401.01234", annotation="aaa",
    )
    bib = science._build_bibtex([p])
    assert "@article{" in bib
    assert "title = {Hello}" in bib
    assert "year = {2024}" in bib
    assert "eprint = {arxiv:2401.01234}" in bib
    assert "url = {https://arxiv.org/abs/2401.01234}" in bib


def test_build_bibtex_includes_doi() -> None:
    p = science.Paper(
        title="J", authors=["A B"], year=2023, venue="Журнал", arxiv_id=None,
        url=None, annotation="", doi="10.1234/abc",
    )
    assert "doi = {10.1234/abc}" in science._build_bibtex([p])


def test_build_bibtex_no_year() -> None:
    p = science.Paper(
        title="X", authors=["A B"], year=None, venue="", arxiv_id=None, url=None,
        annotation="",
    )
    assert "year = {n.d.}" in science._build_bibtex([p])
