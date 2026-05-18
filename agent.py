"""science-agent — поиск РЕАЛЬНЫХ научных статей по теме.

Принцип: агент НИКОГДА не выдумывает публикации. Он ищет в реальных
индексах через allowlist-прокси portal-api и работает только с тем, что
там реально нашлось:

- arXiv          (/api/sandbox/arxiv)            — препринты STEM
- Crossref       (/api/sandbox/crossref)         — DOI-журналы, в т.ч.
                                                   русско-/гуманитарные
- Semantic Scholar (/api/sandbox/semantic-scholar) — abstracts + цитируемость

Результаты трёх источников объединяются и дедуплицируются. LLM
используется ТОЛЬКО чтобы оценить релевантность реальных статей и
написать аннотации по их реальным abstract'ам — не для генерации списка.

Режим `llm` существует, но он ЯВНЫЙ: каждая запись помечается «НЕ
ПРОВЕРЕНО», и отчёт несёт крупное предупреждение. Если реальный поиск
ничего не дал — агент честно завершается с ошибкой, а НЕ подменяет
выдачу галлюцинацией.

Для реально найденных arXiv-статей агент скачивает сам PDF (через
/api/sandbox/arxiv-pdf) в output/pdfs/, чтобы файл сохранился у портала.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import httpx
from docx import Document

from portal_sdk import Agent

MAX_PDF_DOWNLOADS = 10  # сколько верхних статей качать файлом
# Совокупный бюджет PDF: заведомо ниже portal max_job_output_bytes (1 GiB),
# чтобы report.docx/sources.bib всегда сохранились, даже если PDF большие.
MAX_PDF_TOTAL_BYTES = 200 * 1024 * 1024


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
    provenance: list[str] = field(default_factory=list)  # ['arXiv','Crossref',...]
    unverified: bool = False  # True только в явном LLM-режиме
    pdf_filename: str | None = None  # имя файла в output/pdfs/ если скачали

    @property
    def bibkey(self) -> str:
        parts = (self.authors[0] if self.authors else "anon").split()
        surname = max(parts, key=len) if parts else "anon"
        surname = re.sub(r"[^a-z0-9]", "", surname.lower()) or "anon"
        return f"{surname}{self.year or 'nd'}_{re.sub(r'[^A-Za-z0-9]', '', self.title)[:20].lower()}"

    @property
    def best_link(self) -> str | None:
        aid = _norm_arxiv_id(self.arxiv_id)
        if aid:
            return f"https://arxiv.org/abs/{aid}"
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return self.url


def _llm_call(messages: list[dict], model: str, api_key: str, base_url: str, *, max_tokens: int = 8000) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens}
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


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", (t or "").lower())


def _strip_jats(s: str) -> str:
    """Crossref abstract приходит JATS-XML — выкидываем теги."""
    return re.sub(r"<[^>]+>", " ", s or "").strip()


def _sandbox_root(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/llm/v1"):
        base = base[: -len("/llm/v1")]
    return base


def _norm_arxiv_id(s: str | None) -> str | None:
    """Единственная точка нормализации arxiv_id: срезаем префикс `arxiv:`
    и пробелы. Используется везде (ingest/ссылки/скачивание)."""
    if not s:
        return None
    out = re.sub(r"(?i)^arxiv:", "", s).strip()
    return out or None


def _surname(authors: list[str]) -> str:
    parts = (authors[0] if authors else "").split()
    return max(parts, key=len).lower() if parts else ""


def _arxiv_pdf_url(arxiv_id: str | None) -> str | None:
    aid = _norm_arxiv_id(arxiv_id)
    return f"https://arxiv.org/pdf/{aid}.pdf" if aid else None


# --- Реальные источники через sandbox-прокси ---

def _get_json(client: httpx.Client, url: str, params: dict, api_key: str) -> dict:
    r = client.get(url, params=params, headers={"Authorization": f"Bearer {api_key}"})
    r.raise_for_status()
    return r.json()


def _search_arxiv(topic: str, n: int, api_key: str, root: str) -> list[Paper]:
    out: list[Paper] = []
    with httpx.Client(timeout=60) as c:
        data = _get_json(c, f"{root}/api/sandbox/arxiv",
                         {"search_query": topic, "max_results": n}, api_key)
    for row in data.get("papers", []):
        aid = _norm_arxiv_id(row.get("arxiv_id"))
        out.append(Paper(
            title=row.get("title", ""), authors=row.get("authors", []),
            year=row.get("year"), venue="arXiv", arxiv_id=aid,
            url=row.get("url"), annotation=_strip_jats(row.get("abstract", "")),
            pdf_url=_arxiv_pdf_url(aid), provenance=["arXiv"],
        ))
    return out


def _search_crossref(topic: str, n: int, api_key: str, root: str) -> list[Paper]:
    out: list[Paper] = []
    with httpx.Client(timeout=60) as c:
        data = _get_json(c, f"{root}/api/sandbox/crossref",
                         {"query": topic, "rows": n}, api_key)
    for row in data.get("works", []):
        out.append(Paper(
            title=row.get("title", ""), authors=row.get("authors", []),
            year=row.get("year"), venue=row.get("venue", ""), arxiv_id=None,
            url=row.get("url"), annotation=_strip_jats(row.get("abstract", "")),
            doi=row.get("doi"), citation_count=int(row.get("citation_count") or 0),
            provenance=["Crossref"],
        ))
    return out


def _search_s2(topic: str, n: int, api_key: str, root: str) -> list[Paper]:
    out: list[Paper] = []
    with httpx.Client(timeout=60) as c:
        data = _get_json(c, f"{root}/api/sandbox/semantic-scholar",
                         {"query": topic, "limit": n}, api_key)
    for row in data.get("papers", []):
        aid = _norm_arxiv_id(row.get("arxiv_id"))
        out.append(Paper(
            title=row.get("title", ""), authors=row.get("authors", []),
            year=row.get("year"), venue=row.get("venue", ""), arxiv_id=aid,
            url=row.get("url"), annotation=_strip_jats(row.get("abstract", "")),
            doi=row.get("doi"), citation_count=int(row.get("citation_count") or 0),
            pdf_url=_arxiv_pdf_url(aid), provenance=["Semantic Scholar"],
        ))
    return out


def _merge(groups: list[list[Paper]]) -> list[Paper]:
    """Дедуп. Сильное тождество — DOI: тогда сливаем И идентификаторы.
    Слабое — нормализованное название + год + фамилия первого автора:
    тогда сливаем ТОЛЬКО provenance и описательные поля, идентификаторы
    (arxiv_id/doi/url/pdf_url) НЕ заимствуем — иначе можно подвесить чужой
    PDF к статье (misattribution)."""
    by_key: dict[str, Paper] = {}
    for group in groups:
        for p in group:
            if not p.title.strip():
                continue
            doi = (p.doi or "").strip().lower()
            if doi:
                key, strong = f"doi:{doi}", True
            else:
                nt = _norm_title(p.title)
                if not nt:
                    continue
                key = f"t:{nt}|{p.year or ''}|{_surname(p.authors)}"
                strong = False
            if key not in by_key:
                by_key[key] = p
                continue
            ex = by_key[key]
            for src in p.provenance:
                if src not in ex.provenance:
                    ex.provenance.append(src)
            ex.year = ex.year or p.year
            ex.venue = ex.venue or p.venue
            ex.citation_count = max(ex.citation_count, p.citation_count)
            if len(p.annotation) > len(ex.annotation):
                ex.annotation = p.annotation
            if len(p.authors) > len(ex.authors):
                ex.authors = p.authors
            if strong:  # только при совпадении по DOI безопасно сливать ID
                ex.arxiv_id = ex.arxiv_id or p.arxiv_id
                ex.doi = ex.doi or p.doi
                ex.url = ex.url or p.url
                ex.pdf_url = ex.pdf_url or p.pdf_url
    return list(by_key.values())


def _llm_papers(topic: str, n: int, language: str, model: str, api_key: str, base_url: str) -> list[Paper]:
    """ЯВНЫЙ LLM-режим. Каждая запись помечается unverified=True."""
    ann_lang = "Аннотации — на русском." if language == "ru" else "Annotations — in English."
    system = (
        "Ты — научный библиограф. По теме предложи публикации из своих знаний. "
        "Отвечай строго JSON-массивом, без markdown."
    )
    user = (
        f"Тема: {topic}\n\nПодбери {n} релевантных публикаций. {ann_lang}\n"
        'JSON-массив: [{"title","authors":[..],"year","venue","arxiv_id"|null,'
        '"url"|null,"annotation","score":0..1}]. Только массив.'
    )
    parsed = _parse_json(_llm_call(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model, api_key=api_key, base_url=base_url, max_tokens=8000,
    ))
    if not isinstance(parsed, list):
        raise ValueError("LLM вернула не массив")
    papers: list[Paper] = []
    for row in parsed:
        if not isinstance(row, dict) or not str(row.get("title", "")).strip():
            continue
        year = row.get("year")
        try:
            year = int(year) if year is not None else None
        except (TypeError, ValueError):
            year = None
        papers.append(Paper(
            title=str(row["title"]).strip(),
            authors=[str(a) for a in row.get("authors", []) if a],
            year=year, venue=str(row.get("venue", "")),
            arxiv_id=row.get("arxiv_id") or None, url=row.get("url") or None,
            annotation=str(row.get("annotation", "")),
            score=float(row.get("score", 0.0) or 0.0),
            provenance=["LLM (не проверено)"], unverified=True,
        ))
    papers.sort(key=lambda p: -p.score)
    return papers


def _llm_rank_and_annotate(topic, papers, language, model, api_key, base_url) -> list[Paper]:
    """LLM оценивает РЕАЛЬНЫЕ статьи и пишет аннотации по их abstract'ам.
    Не добавляет и не удаляет статьи — только score/annotation/explanation."""
    if not papers:
        return papers
    ann_lang = "русском" if language == "ru" else "английском"
    items = [
        {"i": i, "title": p.title, "year": p.year, "authors": p.authors[:3],
         "abstract": (p.annotation or "")[:600]}
        for i, p in enumerate(papers)
    ]
    system = (
        "Ты — научный библиограф. Дан список РЕАЛЬНЫХ статей. Для каждой: "
        f"(1) score 0..1 релевантности теме, (2) аннотация на {ann_lang} ~5 "
        "предложений строго по приведённому abstract (не выдумывай фактов), "
        "(3) объяснение балла 1-2 предложения. Строго JSON-массив."
    )
    user = (
        f"Тема: {topic}\n\nСтатьи: {json.dumps(items, ensure_ascii=False)}\n\n"
        'Верни [{"i","score","annotation","score_explanation"}]. Без markdown.'
    )
    try:
        parsed = _parse_json(_llm_call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model, api_key=api_key, base_url=base_url, max_tokens=6000,
        ))
    except Exception:  # noqa: BLE001
        return papers  # LLM сорвалась — отдаём реальные статьи как есть
    if not isinstance(parsed, list):
        return papers
    seen: set[int] = set()
    for row in parsed:
        if not isinstance(row, dict):
            continue
        idx = row.get("i")
        if not isinstance(idx, int) or idx < 0 or idx >= len(papers) or idx in seen:
            continue
        seen.add(idx)
        p = papers[idx]
        p.score = float(row.get("score", 0.0) or 0.0)
        new_ann = str(row.get("annotation", "")).strip()
        if new_ann:
            p.annotation = new_ann
        p.score_explanation = str(row.get("score_explanation", "")).strip()
    return papers


def _download_pdfs(agent: Agent, papers: list[Paper], api_key: str, root: str) -> None:
    """Качаем PDF верхних arXiv-статей в output/pdfs/ через sandbox-прокси."""
    pdf_dir = agent.output_dir / "pdfs"
    done = 0
    total = 0
    for p in papers:
        if done >= MAX_PDF_DOWNLOADS or total >= MAX_PDF_TOTAL_BYTES:
            break
        aid = _norm_arxiv_id(p.arxiv_id)
        if not aid:
            continue
        try:
            with httpx.Client(timeout=90) as c:
                r = c.get(f"{root}/api/sandbox/arxiv-pdf",
                          params={"arxiv_id": aid},
                          headers={"Authorization": f"Bearer {api_key}"})
            if r.status_code != 200:
                agent.log("warn", f"PDF {aid}: прокси {r.status_code}, оставляю ссылку")
                continue
            if total + len(r.content) > MAX_PDF_TOTAL_BYTES:
                agent.log("warn", f"PDF {aid}: превышен общий бюджет, оставляю ссылку")
                break
            pdf_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{re.sub(r'[^A-Za-z0-9._-]', '_', aid)}.pdf"
            (pdf_dir / fname).write_bytes(r.content)
            p.pdf_filename = f"pdfs/{fname}"
            done += 1
            total += len(r.content)
        except Exception as e:  # noqa: BLE001
            agent.log("warn", f"PDF {aid}: не скачал ({e}), оставляю ссылку")
    if done:
        agent.log("info", f"Скачано PDF-файлов: {done} ({total // 1024} КБ)")


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
        if p.doi:
            fields.append(f"  doi = {{{p.doi}}}")
        if p.url:
            fields.append(f"  url = {{{p.url}}}")
        out.append("@article{" + p.bibkey + ",\n" + ",\n".join(fields) + "\n}\n")
    return "\n".join(out)


def _build_report(topic: str, papers: list[Paper], model: str, *, unverified: bool = False) -> Document:
    doc = Document()
    doc.add_heading("Поиск научных статей — отчёт", level=0)
    doc.add_paragraph(f"Тема: {topic}")
    doc.add_paragraph(f"Всего статей: {len(papers)}")
    doc.add_paragraph(f"Модель: {model}")

    if unverified:
        w = doc.add_paragraph()
        run = w.add_run(
            "ВНИМАНИЕ: список подобран языковой моделью из её знаний и НЕ "
            "ПРОВЕРЕН по реальным базам. Возможны несуществующие статьи и "
            "неверные идентификаторы. Перепроверьте каждый источник перед "
            "использованием."
        )
        run.bold = True
    else:
        srcs = sorted({s for p in papers for s in p.provenance})
        doc.add_paragraph(
            "Все статьи ниже — реальные записи из баз: "
            + (", ".join(srcs) if srcs else "—")
            + ". Ссылки кликабельны; для arXiv-статей PDF приложен файлом."
        )

    doc.add_heading("Ранжированный список", level=1)
    for i, p in enumerate(papers, start=1):
        doc.add_heading(f"{i}. {p.title}", level=2)
        if p.unverified:
            r = doc.add_paragraph().add_run("НЕ ПРОВЕРЕНО — предложено моделью")
            r.bold = True
        if p.authors:
            doc.add_paragraph(
                f"Авторы: {', '.join(p.authors[:6])}"
                f"{' и др.' if len(p.authors) > 6 else ''}"
            )
        meta = []
        if p.year:
            meta.append(str(p.year))
        if p.venue:
            meta.append(p.venue)
        if p.arxiv_id:
            meta.append(f"arXiv:{p.arxiv_id}")
        if p.doi:
            meta.append(f"DOI:{p.doi}")
        if p.citation_count:
            meta.append(f"цитирований: {p.citation_count}")
        if p.score:
            meta.append(f"релевантность {p.score:.2f}")
        if p.provenance:
            meta.append("источник: " + ", ".join(p.provenance))
        if meta:
            doc.add_paragraph("  •  ".join(meta))
        if p.annotation:
            doc.add_paragraph(f"Аннотация: {p.annotation}")
        if p.score_explanation:
            doc.add_paragraph(f"Почему такой балл: {p.score_explanation}")
        link = p.best_link
        if link:
            doc.add_paragraph(f"Ссылка: {link}")
        if p.pdf_filename:
            doc.add_paragraph(f"Файл статьи: {p.pdf_filename}")
        elif p.arxiv_id and p.pdf_url:
            doc.add_paragraph(f"PDF: {p.pdf_url}")
    return doc


def main() -> None:
    agent = Agent()
    params = agent.params
    topic: str = (params.get("topic") or "").strip()
    max_papers: int = max(5, min(30, int(params.get("max_papers", 15))))
    language: str = params.get("language", "ru")
    raw_source: str = (params.get("source") or "real").lower()
    # back-compat: старое значение 'arxiv' = реальный поиск
    source = "llm" if raw_source == "llm" else "real"
    sort_by: str = (params.get("sort_by") or "relevance").lower()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    model = (params.get("model") or os.environ.get("LLM_MODEL") or "deepseek/deepseek-r1").strip()
    root = _sandbox_root(base_url)

    if not api_key:
        agent.failed("OPENROUTER_API_KEY не передан.")
        return
    if not topic:
        agent.failed("Не указана тема.")
        return

    agent.log("info", f"science: topic={topic[:80]!r}, max={max_papers}, "
                       f"lang={language}, model={model}, source={source}, sort={sort_by}")

    papers: list[Paper] = []
    unverified = False

    if source == "real":
        per_src = max_papers
        groups: list[list[Paper]] = []
        errors: list[str] = []
        for name, fn in (("arXiv", _search_arxiv),
                         ("Crossref", _search_crossref),
                         ("Semantic Scholar", _search_s2)):
            agent.progress(0.15, f"Поиск: {name}")
            try:
                res = fn(topic, per_src, api_key, root)
                agent.log("info", f"{name}: найдено {len(res)}")
                groups.append(res)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {e}")
                agent.log("warn", f"{name} недоступен: {e}")

        papers = _merge(groups)
        if not papers:
            # ЧЕСТНЫЙ отказ — НЕ подменяем галлюцинацией. Различаем
            # «инфраструктура недоступна» и «реально 0 результатов»,
            # иначе препод снова решит что агент «не умеет в русские темы».
            if not groups and errors:
                agent.failed(
                    "Источники поиска недоступны (ни один не ответил), это "
                    "сбой инфраструктуры, а не отсутствие статей по теме. "
                    "Повторите запуск позже. Детали: " + "; ".join(errors)
                )
            else:
                detail = "; ".join(errors) if errors else "все источники вернули 0 результатов"
                agent.failed(
                    "По теме не найдено реальных публикаций в arXiv / Crossref "
                    "/ Semantic Scholar. Уточните формулировку (попробуйте "
                    "ключевые слова по-английски) либо явно выберите источник "
                    f"«Знания LLM», понимая что он не проверен. Детали: {detail}"
                )
            return
        agent.progress(0.5, f"{model}: оценка релевантности и аннотации")
        papers = _llm_rank_and_annotate(topic, papers, language, model, api_key, base_url)
    else:
        unverified = True
        agent.log("warn", "ЯВНЫЙ LLM-режим: статьи НЕ проверяются по реальным базам.")
        agent.progress(0.2, f"{model}: подбор из знаний модели (не проверено)")
        try:
            papers = _llm_papers(topic, max_papers, language, model, api_key, base_url)
        except httpx.HTTPStatusError as e:
            agent.failed(f"LLM ответил {e.response.status_code}: {e.response.text[:200]}")
            return
        except Exception as e:  # noqa: BLE001
            agent.failed(f"LLM не вернула валидный JSON: {e}")
            return
        if not papers:
            agent.failed("LLM не предложила ни одной публикации — переформулируй тему.")
            return

    if sort_by == "popularity":
        papers.sort(key=lambda p: (-p.citation_count, -p.score))
    else:
        papers.sort(key=lambda p: (-p.score, -p.citation_count))
    papers = papers[:max_papers]

    if source == "real":
        agent.progress(0.8, "Скачиваю PDF реально найденных arXiv-статей")
        _download_pdfs(agent, papers, api_key, root)

    for p in papers:
        agent.item_done(p.arxiv_id or p.doi or p.title[:40], summary=p.title,
                        data={"year": p.year, "venue": p.venue, "score": p.score,
                              "unverified": p.unverified})

    agent.progress(0.92, "Формирую report.docx и sources.bib")
    out_dir = agent.output_dir
    _build_report(topic, papers, model, unverified=unverified).save(out_dir / "report.docx")
    (out_dir / "sources.bib").write_text(_build_bibtex(papers), encoding="utf-8")

    agent.progress(1.0, "Готово")
    agent.result(artifacts=[
        {"id": "report", "path": "report.docx"},
        {"id": "bibtex", "path": "sources.bib"},
    ])


if __name__ == "__main__":
    main()
