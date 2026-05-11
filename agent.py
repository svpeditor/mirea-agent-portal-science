"""science-agent — поиск научных статей по теме.

Два режима:
1. **arxiv** (default) — портал-api прокидывает запрос в export.arxiv.org
   через /api/sandbox/arxiv (allowlist-proxy), DeepSeek-R1 ранжирует
   результаты и пишет аннотации на русском. Реальные статьи, реальные
   arxiv_id.
2. **llm** — fallback на знания LLM (если sandbox-endpoint недоступен,
   или хочется быстро без сетевого запроса).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import httpx
from docx import Document

from portal_sdk import Agent


@dataclass
class Paper:
    title: str
    authors: list[str]
    year: int | None
    venue: str
    arxiv_id: str | None
    url: str | None
    annotation: str
    score: float = 0.0
    score_explanation: str = ""
    pdf_url: str | None = None
    doi: str | None = None
    citation_count: int = 0

    @property
    def bibkey(self) -> str:
        # Берём самое длинное «слово» в первом авторе как фамилию.
        # У "Vaswani A." это Vaswani, у "A. Vaswani" — тоже Vaswani.
        parts = (self.authors[0] if self.authors else "anon").split()
        if parts:
            surname = max(parts, key=len)
        else:
            surname = "anon"
        surname = re.sub(r"[^a-z0-9]", "", surname.lower()) or "anon"
        return f"{surname}{self.year or 'nd'}_{re.sub(r'[^A-Za-z0-9]', '', self.title)[:20].lower()}"


def _llm_call(messages: list[dict], model: str, api_key: str, base_url: str, *, max_tokens: int = 8000) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/svpeditor/mirea-agent-portal-science",
        "X-Title": "mirea-science-agent",
    }
    with httpx.Client(timeout=300) as client:
        r = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _strip_reasoning(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
    if m:
        s = m.group(1)
    return s.strip()


def _parse_json(s: str):
    s = _strip_reasoning(s)
    m = re.search(r"[\[\{].*[\]\}]", s, re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)


def _llm_papers(topic: str, n: int, language: str, model: str, api_key: str, base_url: str) -> list[Paper]:
    system = (
        "Ты — научный библиограф. По теме исследования предложишь публикации "
        "из своих знаний (arXiv, ведущие конференции). Аннотации пиши кратко и информативно. "
        "Отвечай строго JSON-массивом, без markdown, без пояснений."
    )
    ann_lang = "Аннотации — на русском." if language == "ru" else "Annotations — in English."
    user = f"""Тема: {topic}

Подбери {n} наиболее релевантных публикаций. {ann_lang}

Формат — JSON-массив объектов, отранжированный по релевантности:
[
  {{
    "title": "<точное название>",
    "authors": ["<Фамилия И.О.>", "..."],
    "year": <YYYY>,
    "venue": "<arXiv/conference/journal>",
    "arxiv_id": "<arxiv-id если знаешь, иначе null>",
    "url": "<https://... если знаешь, иначе null>",
    "annotation": "<1-2 предложения: о чём работа и почему релевантна теме>",
    "score": <0..1, релевантность теме>
  }}
]

Только массив, ничего больше."""
    raw = _llm_call(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model, api_key=api_key, base_url=base_url, max_tokens=8000,
    )
    parsed = _parse_json(raw)
    if not isinstance(parsed, list):
        raise ValueError("LLM вернула не массив")

    papers: list[Paper] = []
    for row in parsed:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        authors_raw = row.get("authors", [])
        authors = [str(a) for a in authors_raw if a] if isinstance(authors_raw, list) else []
        year = row.get("year")
        try:
            year = int(year) if year is not None else None
        except (TypeError, ValueError):
            year = None
        papers.append(Paper(
            title=title,
            authors=authors,
            year=year,
            venue=str(row.get("venue", "")),
            arxiv_id=row.get("arxiv_id") or None,
            url=row.get("url") or None,
            annotation=str(row.get("annotation", "")),
            score=float(row.get("score", 0.0) or 0.0),
        ))
    papers.sort(key=lambda p: -p.score)
    return papers


def _sandbox_root(base_url: str) -> str:
    """Корень portal-api для sandbox endpoints (base_url = LLM-прокси)."""
    base = base_url.rstrip("/")
    if base.endswith("/llm/v1"):
        base = base[: -len("/llm/v1")]
    return base


def _arxiv_pdf_url(arxiv_id: str | None) -> str | None:
    """arxiv:1706.03762v5 → https://arxiv.org/pdf/1706.03762v5.pdf"""
    if not arxiv_id:
        return None
    aid = arxiv_id.replace("arxiv:", "")
    return f"https://arxiv.org/pdf/{aid}.pdf"


def _arxiv_search(query: str, max_results: int, api_key: str, base_url: str) -> list[Paper]:
    """Дёргает /api/sandbox/arxiv на portal-api (или совместимом URL)."""
    url = f"{_sandbox_root(base_url)}/api/sandbox/arxiv"
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=60) as client:
        r = client.get(url, params={"search_query": query, "max_results": max_results}, headers=headers)
        r.raise_for_status()
        data = r.json()

    papers: list[Paper] = []
    for row in data.get("papers", []):
        aid = row.get("arxiv_id")
        papers.append(Paper(
            title=row.get("title", ""),
            authors=row.get("authors", []),
            year=row.get("year"),
            venue="arXiv",
            arxiv_id=aid,
            url=row.get("url"),
            annotation="",  # потом дополним LLM
            pdf_url=_arxiv_pdf_url(aid),
        ))
    return papers


def _enrich_via_s2(papers: list[Paper], topic: str, api_key: str, base_url: str) -> list[Paper]:
    """Достаём citation counts + DOI + abstracts из Semantic Scholar.

    Делаем ОДИН запрос по `topic`, сопоставляем по нормализованному названию.
    Если статья нашлась — заполняем citation_count, doi, удлиняем abstract
    в annotation если он пустой.
    """
    url = f"{_sandbox_root(base_url)}/api/sandbox/semantic-scholar"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(url, params={"query": topic, "limit": 50}, headers=headers)
            r.raise_for_status()
            s2 = r.json().get("papers", [])
    except Exception:  # noqa: BLE001
        return papers

    def _norm(t: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (t or "").lower())

    by_title = {_norm(p["title"]): p for p in s2 if p.get("title")}
    for p in papers:
        match = by_title.get(_norm(p.title))
        if match:
            p.citation_count = int(match.get("citation_count") or 0)
            p.doi = match.get("doi") or p.doi
            if not p.annotation and match.get("abstract"):
                p.annotation = match["abstract"][:600]
    return papers


def _llm_rank_and_annotate(
    topic: str,
    papers: list[Paper],
    language: str,
    model: str,
    api_key: str,
    base_url: str,
) -> list[Paper]:
    """LLM: score (0..1) + аннотация (~5 предложений) + объяснение балла."""
    if not papers:
        return papers
    ann_lang = "русском" if language == "ru" else "английском"
    items = [
        {
            "i": i,
            "title": p.title,
            "year": p.year,
            "authors": p.authors[:3],
            "abstract": (p.annotation or "")[:600],  # S2 abstract если есть
        }
        for i, p in enumerate(papers)
    ]
    system = (
        "Ты — научный библиограф. Получишь тему и список статей. "
        f"Для каждой статьи: (1) поставь score 0..1 (соответствие теме), "
        f"(2) напиши аннотацию на {ann_lang} ~5 предложений, "
        "(3) объясни в 1-2 предложениях почему именно такой score. "
        "Отвечай строго JSON-массивом, без markdown."
    )
    user = (
        f"Тема: {topic}\n\nСтатьи: {json.dumps(items, ensure_ascii=False)}\n\n"
        'Верни JSON: [{"i":<idx>,"score":<0..1>,"annotation":"...","score_explanation":"..."},...]. '
        "Сортировка по убыванию score."
    )
    raw = _llm_call(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model, api_key=api_key, base_url=base_url, max_tokens=6000,
    )
    parsed = _parse_json(raw)
    if not isinstance(parsed, list):
        return papers

    ranked: list[Paper] = []
    seen: set[int] = set()
    for row in parsed:
        if not isinstance(row, dict):
            continue
        idx = row.get("i")
        if not isinstance(idx, int) or idx < 0 or idx >= len(papers):
            continue
        if idx in seen:
            continue
        seen.add(idx)
        p = papers[idx]
        p.score = float(row.get("score", 0.0) or 0.0)
        new_ann = str(row.get("annotation", "")).strip()
        if new_ann:
            p.annotation = new_ann
        p.score_explanation = str(row.get("score_explanation", "")).strip()
        ranked.append(p)
    for i, p in enumerate(papers):
        if i not in seen:
            ranked.append(p)
    return ranked


def _build_bibtex(papers: list[Paper]) -> str:
    out: list[str] = []
    for p in papers:
        authors = " and ".join(p.authors) if p.authors else "Unknown"
        fields = [
            f"  title = {{{p.title}}}",
            f"  author = {{{authors}}}",
            f"  year = {{{p.year or 'n.d.'}}}",
        ]
        if p.venue:
            fields.append(f"  journal = {{{p.venue}}}")
        if p.arxiv_id:
            fields.append(f"  eprint = {{arxiv:{p.arxiv_id}}}")
        if p.url:
            fields.append(f"  url = {{{p.url}}}")
        out.append("@article{" + p.bibkey + ",\n" + ",\n".join(fields) + "\n}\n")
    return "\n".join(out)


def _build_report(topic: str, papers: list[Paper], model: str) -> Document:
    doc = Document()
    doc.add_heading("Поиск научных статей — отчёт", level=0)
    doc.add_paragraph(f"Тема: {topic}")
    doc.add_paragraph(f"Всего статей: {len(papers)}")
    doc.add_paragraph(f"Модель: {model}")
    doc.add_paragraph(
        "Подбор сделан LLM на основе её знаний. Перепроверьте цитирования "
        "перед использованием — LLM может ошибаться в датах и идентификаторах."
    )

    doc.add_heading("Ранжированный список", level=1)
    for i, p in enumerate(papers, start=1):
        doc.add_heading(f"{i}. {p.title}", level=2)
        if p.authors:
            doc.add_paragraph(f"Авторы: {', '.join(p.authors[:6])}{' и др.' if len(p.authors) > 6 else ''}")
        meta_parts = []
        if p.year:
            meta_parts.append(str(p.year))
        if p.venue:
            meta_parts.append(p.venue)
        if p.arxiv_id:
            meta_parts.append(f"arXiv:{p.arxiv_id}")
        if p.doi:
            meta_parts.append(f"DOI:{p.doi}")
        if p.citation_count:
            meta_parts.append(f"цитирований: {p.citation_count}")
        if p.score:
            meta_parts.append(f"релевантность {p.score:.2f}")
        if meta_parts:
            doc.add_paragraph("  •  ".join(meta_parts))
        if p.annotation:
            doc.add_paragraph(f"Аннотация: {p.annotation}")
        if p.score_explanation:
            doc.add_paragraph(f"Почему такой балл: {p.score_explanation}")
        if p.pdf_url:
            doc.add_paragraph(f"Скачать PDF: {p.pdf_url}")
        elif p.url:
            doc.add_paragraph(f"URL: {p.url}")
    return doc


def main() -> None:
    agent = Agent()
    params = agent.params
    topic: str = (params.get("topic") or "").strip()
    max_papers: int = max(5, min(30, int(params.get("max_papers", 15))))
    language: str = params.get("language", "en")
    source: str = (params.get("source") or "arxiv").lower()
    sort_by: str = (params.get("sort_by") or "relevance").lower()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    model = (
        params.get("model")
        or os.environ.get("LLM_MODEL")
        or "deepseek/deepseek-r1"
    ).strip()

    if not api_key:
        agent.failed("OPENROUTER_API_KEY не передан.")
        return
    if not topic:
        agent.failed("Не указана тема.")
        return

    agent.log(
        "info",
        f"science: topic={topic[:80]!r}, max_papers={max_papers}, "
        f"language={language}, model={model}, source={source}, sort={sort_by}",
    )

    papers: list[Paper] = []
    if source == "arxiv":
        agent.progress(0.1, "Запрос в arXiv через portal-api proxy")
        try:
            papers = _arxiv_search(topic, max_papers, api_key, base_url)
        except httpx.HTTPStatusError as e:
            agent.log("warn", f"arXiv proxy ответил {e.response.status_code}, переключаюсь на LLM-режим")
            source = "llm"
        except Exception as e:  # noqa: BLE001
            agent.log("warn", f"arXiv proxy недоступен ({e}), переключаюсь на LLM-режим")
            source = "llm"

        if papers:
            agent.log("info", f"arXiv вернул {len(papers)} статей, обогащаю через Semantic Scholar")
            agent.progress(0.35, "Semantic Scholar: citations + DOI + abstracts")
            try:
                papers = _enrich_via_s2(papers, topic, api_key, base_url)
            except Exception as e:  # noqa: BLE001
                agent.log("warn", f"S2 enrichment не удался ({e}), идём дальше")

            agent.progress(0.5, f"{model} оценивает релевантность и пишет аннотации")
            try:
                papers = _llm_rank_and_annotate(topic, papers, language, model, api_key, base_url)
            except Exception as e:  # noqa: BLE001
                agent.log("warn", f"LLM ранжирование сорвалось ({e}); идём с arXiv-порядком")

    if source == "llm" or not papers:
        if "deepseek" in model.lower() and "r1" in model.lower():
            agent.log("info", "DeepSeek-R1 — reasoning-модель, обычно отвечает 1-4 минуты. Терпение.")
        agent.progress(0.1, f"Запрос к {model} на подбор публикаций из знаний модели")
        try:
            papers = _llm_papers(topic, max_papers, language, model, api_key, base_url)
        except httpx.HTTPStatusError as e:
            agent.failed(f"LLM ответил {e.response.status_code}: {e.response.text[:200]}")
            return
        except Exception as e:  # noqa: BLE001
            agent.failed(f"LLM не вернула валидный JSON: {e}")
            return

    if not papers:
        agent.failed("Не удалось найти ни одной публикации — переформулируй тему.")
        return

    # Финальная сортировка
    if sort_by == "popularity":
        papers.sort(key=lambda p: (-p.citation_count, -p.score))
    else:
        papers.sort(key=lambda p: (-p.score, -p.citation_count))

    for p in papers:
        agent.item_done(
            p.arxiv_id or p.title[:40],
            summary=p.title,
            data={"year": p.year, "venue": p.venue, "score": p.score},
        )

    agent.progress(0.85, "Формируем report.docx и sources.bib")
    out_dir = agent.output_dir
    _build_report(topic, papers, model).save(out_dir / "report.docx")
    (out_dir / "sources.bib").write_text(_build_bibtex(papers), encoding="utf-8")

    agent.progress(1.0, "Готово")
    agent.result(artifacts=[
        {"id": "report", "path": "report.docx"},
        {"id": "bibtex", "path": "sources.bib"},
    ])


if __name__ == "__main__":
    main()
