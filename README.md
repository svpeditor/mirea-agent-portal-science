# mirea-agent-portal-science

Агент для платформы [mirea-agent-portal](https://github.com/svpeditor/mirea-agent-portal).

## Что делает

Принимает тему исследования, спрашивает у **DeepSeek-R1** (OpenRouter, через LLM-прокси портала) список релевантных публикаций со ссылками и краткими аннотациями. Возвращает:

- `report.docx` — отчёт с ранжированным списком статей
- `sources.bib` — BibTeX для подключения в LaTeX

## Почему LLM-only, а не arXiv API

Агенты на платформе крутятся в изолированной docker-сети (`internal: true`). Им доступен только LLM-прокси portal-api, публичный интернет (включая `export.arxiv.org`) — закрыт.

Поэтому источник публикаций — знания самой модели DeepSeek-R1. Аннотации и идентификаторы перепроверяй перед использованием: LLM иногда галлюцинирует.

Production-вариант (вне scope wave0): прокинуть arXiv через отдельный allowlist-endpoint portal-api.

## Параметры

| Поле | Тип | Описание |
|------|-----|----------|
| `topic` | textarea | Тема (RU или EN) |
| `max_papers` | number | Сколько публикаций (5..30, default 15) |
| `language` | radio | Язык аннотаций: `ru` / `en` |

## LLM

`OPENROUTER_API_KEY` — ephemeral, инжектится порталом. `OPENROUTER_BASE_URL` указывает на LLM-прокси portal-api. Модель по умолчанию `deepseek/deepseek-r1`.

## Локально (вне портала)

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-v1-..."
export INPUT_DIR=/tmp/in OUTPUT_DIR=/tmp/out
mkdir -p $INPUT_DIR $OUTPUT_DIR
echo '{"topic":"machine learning for medical imaging","max_papers":10,"language":"en"}' > $INPUT_DIR/params.json
python agent.py
```

## Публикация в портал

```bash
curl -X POST https://your-portal/api/admin/agents \
  -H 'Cookie: session=...' \
  -d '{"git_url":"https://github.com/svpeditor/mirea-agent-portal-science","git_ref":"main"}'
```
