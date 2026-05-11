# mirea-agent-portal-science

Агент для платформы [mirea-agent-portal](https://github.com/svpeditor/mirea-agent-portal).

## Что делает

Принимает тему исследования, ищет статьи в **arXiv**, ранжирует и аннотирует их через **DeepSeek-R1** (OpenRouter), отдаёт:

- `report.docx` — отчёт с ранжированным списком статей, аннотациями на русском
- `sources.bib` — BibTeX-файл для подключения в LaTeX

## Pipeline

1. Если тема на русском → DeepSeek-R1 переводит в EN-query для arXiv.
2. `http://export.arxiv.org/api/query` → до 50 кандидатов.
3. DeepSeek-R1 ранжирует по релевантности + пишет 1-2 предложения аннотации на каждую.
4. `python-docx` рисует отчёт, на лету генерим BibTeX.

## Параметры

| Поле | Тип | Описание |
|------|-----|----------|
| `topic` | textarea | Тема (RU или EN) |
| `max_papers` | number | Сколько статей искать (5..50, default 20) |
| `language` | radio | `ru` — переводим в EN; `en` — ищем напрямую |

## LLM

`OPENROUTER_API_KEY` инжектится порталом (ephemeral-токен). Модель по умолчанию `deepseek/deepseek-r1`, переопределяется через env `LLM_MODEL`.

## Локально

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-v1-..."
export INPUT_DIR=/tmp/in OUTPUT_DIR=/tmp/out
mkdir -p $INPUT_DIR $OUTPUT_DIR
echo '{"topic":"machine learning for medical imaging","max_papers":15,"language":"en"}' > $INPUT_DIR/params.json
python agent.py
```

## Подключение к порталу

```bash
curl -X POST https://your-portal/api/admin/agents \
  -H 'Cookie: session=...' \
  -d '{"git_url":"https://github.com/svpeditor/mirea-agent-portal-science","git_ref":"main"}'
```
