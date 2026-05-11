"""science-agent — реальный поиск научных статей по теме.

Pipeline:
1. Если тема на русском — DeepSeek-R1 переводит её в английский query.
2. Тянем кандидатов из arXiv API (http://export.arxiv.org/api/query).
3. DeepSeek-R1 ранжирует кандидатов по релевантности теме + даёт краткую
   аннотацию для каждой топ-N статьи.
4. Формируем report.docx + sources.bib.

LLM: DeepSeek-R1 через OpenRouter, ephemeral-токен OPENROUTER_API_KEY
инжектится порталом.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
from dataclasses import dataclass, field

import feedparser
import httpx
from docx import Document

from portal_sdk import Agent

ARXIV_API = "http://export.arxiv.org/api/query"


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int | None
    url: str
    score: float = 0.0
    annotation: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def bibkey(self) -> str:
        # arxiv:2401.01234 → arxiv_2401_01234
        return re.sub(r"[^A-Za-z0-9]+", "_", self.arxiv_id).strip("_")


def _llm_call(messages: list[dict], model: str, api_key: str, base_url: str, *, max_tokens: int = 2000) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/svpeditor/mirea-agent-portal-science",
        "X-Title": "mirea-science-agent",
    }
    with httpx.Client(timeout=120) as client:
        r = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _strip_reasoning(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
    if m:
        s = m.group(1)
    return s.strip()


def _parse_json(s: str) -> object:
    s = _strip_reasoning(s)
    m = re.search(r"[\[\{].*[\]\}]", s, re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)


def _ru_to_en_query(topic_ru: str, model: str, api_key: str, base_url: str) -> str:
    out = _llm_call(
        [
            {"role": "system", "content": "Ты переводчик. Переводи научную тему RU→EN, отдавай только сам перевод одной строкой, без кавычек, без префиксов."},
            {"role": "user", "content": topic_ru},
        ],
        model=model, api_key=api_key, base_url=base_url, max_tokens=200,
    )
    out = _strip_reasoning(out).strip().strip('"').strip("'")
    out = out.split("\n", 1)[0].strip()
    return out or topic_ru


def _arxiv_search(query: str, max_results: int) -> list[Paper]:
    q = urllib.parse.quote(query)
    url = f"{ARXIV_API}?search_query=all:{q}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    with httpx.Client(timeout=60) as client:
        r = client.get(url)
        r.raise_for_status()
        feed = feedparser.parse(r.text)

    papers: list[Paper] = []
    for entry in feed.entries:
        arxiv_id = entry.get("id", "").rsplit("/", 1)[-1]
        if not arxiv_id:
            continue
        title = re.sub(r"\s+", " ", entry.get("title", "")).strip()
        abstract = re.sub(r"\s+", " ", entry.get("summary", "")).strip()
        authors = [a.get("name", "") for a in entry.get("authors", [])]
        year = None
        published = entry.get("published", "")
        m = re.match(r"(\d{4})", published)
        if m:
            year = int(m.group(1))
        papers.append(Paper(
            arxiv_id=f"arxiv:{arxiv_id}",
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            url=entry.get("link", ""),
        ))
    return papers


def _llm_rank_and_annotate(topic: str, papers: list[Paper], model: str, api_key: str, base_url: str) -> list[Paper]:
    """Один LLM-запрос: ранжирование + аннотации в JSON."""
    if not papers:
        return []
    items = [
        {
            "i": i,
            "title": p.title,
            "abstract": p.abstract[:600],
            "year": p.year,
        }
        for i, p in enumerate(papers)
    ]
    system = (
        "Ты — научный библиограф. Получишь тему исследования и список статей. "
        "Отранжируй по релевантности теме и для каждой дай краткую (1-2 предложения) "
        "аннотацию на русском. Отвечай строго в JSON, без markdown."
    )
    user = (
        f"Тема: {topic}\n\n"
        f"Список (json): {json.dumps(items, ensure_ascii=False)}\n\n"
        "Верни JSON-массив объектов в порядке убывания релевантности:\n"
        '[{"i": <индекс>, "score": <0..1>, "annotation": "<RU>"}, ...]\n'
        "Только массив, никакого текста вне него."
    )
    raw = _llm_call(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model, api_key=api_key, base_url=base_url, max_tokens=4000,
    )
    parsed = _parse_json(raw)
    if not isinstance(parsed, list):
        raise ValueError("LLM вернула не массив")

    ranked: list[Paper] = []
    seen: set[int] = set()
    for row in parsed:
        if not isinstance(row, dict):
            continue
        idx = row.get("i")
        if not isinstance(idx, int) or idx < 0 or idx >= len(papers) or idx in seen:
            continue
        p = papers[idx]
        p.score = float(row.get("score", 0))
        p.annotation = str(row.get("annotation", ""))
        ranked.append(p)
        seen.add(idx)
    # хвост: статьи, которые LLM не вернула, приклеиваем со score=0
    for i, p in enumerate(papers):
        if i not in seen:
            ranked.append(p)
    return ranked


def _build_bibtex(papers: list[Paper]) -> str:
    lines: list[str] = []
    for p in papers:
        authors = " and ".join(p.authors) if p.authors else "Unknown"
        lines.append(
            f"@article{{{p.bibkey},\n"
            f"  title = {{{p.title}}},\n"
            f"  author = {{{authors}}},\n"
            f"  year = {{{p.year or 'n.d.'}}},\n"
            f"  eprint = {{{p.arxiv_id}}},\n"
            f"  url = {{{p.url}}}\n"
            "}\n"
        )
    return "\n".join(lines)


def _build_report(topic: str, query: str, papers: list[Paper], model: str) -> Document:
    doc = Document()
    doc.add_heading("Поиск научных статей — отчёт", level=0)
    doc.add_paragraph(f"Тема: {topic}")
    if query != topic:
        doc.add_paragraph(f"Поисковый запрос (EN): {query}")
    doc.add_paragraph(f"Всего статей: {len(papers)}")
    doc.add_paragraph(f"Источник: arXiv (api.export.arxiv.org)")
    doc.add_paragraph(f"Модель ранжирования: {model}")

    doc.add_heading("Ранжированный список", level=1)
    for i, p in enumerate(papers, start=1):
        doc.add_heading(f"{i}. {p.title}", level=2)
        if p.authors:
            doc.add_paragraph(f"Авторы: {', '.join(p.authors[:6])}{' и др.' if len(p.authors) > 6 else ''}")
        meta_parts = []
        if p.year:
            meta_parts.append(str(p.year))
        meta_parts.append(p.arxiv_id)
        if p.score:
            meta_parts.append(f"релевантность {p.score:.2f}")
        doc.add_paragraph("  •  ".join(meta_parts))
        if p.annotation:
            doc.add_paragraph(f"Аннотация: {p.annotation}")
        if p.url:
            doc.add_paragraph(f"URL: {p.url}")

    return doc


def main() -> None:
    agent = Agent()
    params = agent.params
    topic: str = (params.get("topic") or "").strip()
    max_papers: int = max(5, min(50, int(params.get("max_papers", 20))))
    language: str = params.get("language", "en")

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    model = os.environ.get("LLM_MODEL", "deepseek/deepseek-r1").strip()

    if not api_key:
        agent.failed("OPENROUTER_API_KEY не передан в контейнер.")
        return
    if not topic:
        agent.failed("Не указана тема (topic).")
        return

    agent.log("info", f"science: topic={topic[:80]!r}, max_papers={max_papers}, language={language}")

    # 1. Перевод темы на английский для arXiv, если нужно.
    query = topic
    if language == "ru":
        agent.progress(0.05, "Переводим тему на английский")
        try:
            query = _ru_to_en_query(topic, model, api_key, base_url)
            agent.log("info", f"Переведённый запрос: {query!r}")
        except Exception as e:  # noqa: BLE001
            agent.log("warn", f"Перевод не удался ({e}), ищем по исходной теме")

    # 2. arXiv search.
    agent.progress(0.15, "Запрос в arXiv")
    try:
        papers = _arxiv_search(query, max_papers)
    except Exception as e:  # noqa: BLE001
        agent.failed(f"arXiv недоступен: {e}")
        return
    agent.log("info", f"arXiv вернул {len(papers)} кандидатов")
    if not papers:
        agent.failed("arXiv не нашёл статей по запросу. Попробуй переформулировать тему.")
        return

    for i, p in enumerate(papers):
        agent.item_done(p.arxiv_id, summary=p.title, data={"year": p.year})

    # 3. LLM ранжирование + аннотации.
    agent.progress(0.6, f"DeepSeek-R1 ранжирует {len(papers)} статей")
    try:
        ranked = _llm_rank_and_annotate(topic, papers, model, api_key, base_url)
    except Exception as e:  # noqa: BLE001
        agent.log("error", f"LLM ранжирование сорвалось: {e}; идём с arXiv-порядком")
        ranked = papers

    # 4. Финальные артефакты.
    agent.progress(0.9, "Формируем report.docx и sources.bib")
    out_dir = agent.output_dir
    report = _build_report(topic, query, ranked, model)
    report.save(out_dir / "report.docx")
    (out_dir / "sources.bib").write_text(_build_bibtex(ranked), encoding="utf-8")

    agent.progress(1.0, "Готово")
    agent.result(artifacts=[
        {"id": "report", "path": "report.docx"},
        {"id": "bibtex", "path": "sources.bib"},
    ])


if __name__ == "__main__":
    main()
