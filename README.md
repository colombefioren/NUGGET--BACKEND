<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo-dark.png">
  <img src="docs/logo.png" alt="Nugget" width="260">
</picture>

### the api that digs the nugget out of your docs

Multilingual hybrid-search RAG: local embeddings, BM25 + vectors, streamed cited answers.

![Python](https://img.shields.io/badge/Python_3.12-18141e?style=for-the-badge&logo=python&logoColor=FFBF2E)
![FastAPI](https://img.shields.io/badge/FastAPI-18141e?style=for-the-badge&logo=fastapi&logoColor=A6F0CE)
![LangChain](https://img.shields.io/badge/LangChain-18141e?style=for-the-badge&logo=langchain&logoColor=CCBAFF)
![Chroma](https://img.shields.io/badge/ChromaDB-18141e?style=for-the-badge&logo=databricks&logoColor=FF48A0)
![ONNX](https://img.shields.io/badge/FastEmbed_·_ONNX-18141e?style=for-the-badge&logo=onnx&logoColor=A6DCFF)
![uv](https://img.shields.io/badge/uv-18141e?style=for-the-badge&logo=astral&logoColor=FFBF2E)

[← Frontend](https://github.com/colombefioren/NUGGET--FRONTEND) · **Backend**

</div>

## ✦ How it digs

```mermaid
flowchart LR
  F[PDF · DOCX · HTML · MD · CSV · JSON] --> S[chunk per page] --> E[multilingual embeddings<br/>local ONNX] --> C[(Chroma)]
  Q[question] --> R[rewrite follow-up] --> H{vectors + BM25<br/>RRF fusion}
  C --> H --> L[any OpenAI-compatible LLM] -- SSE + citations --> A[answer]
```

- **Hybrid retrieval**: dense vectors for meaning, BM25 (accent folding, CJK bigrams) for exact terms, merged with reciprocal rank fusion
- **Cross-lingual**: ask in Japanese about a French PDF and get the answer in Japanese
- **Grounded**: every claim carries an `[n]` citation. Follow-ups are rewritten into standalone search queries first
- **Private indexing**: embeddings are computed on your machine, and re-uploaded files are deduplicated by content hash

## ✦ Run it

```bash
cp .env.example .env     # set NUGGET_API_BASE, NUGGET_CHAT_MODEL, NUGGET_API_KEY
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

| Env | |
| --- | --- |
| `NUGGET_API_BASE` · `NUGGET_CHAT_MODEL` | **required**, any OpenAI-compatible endpoint + model |
| `NUGGET_API_KEY` | needed to generate answers (retrieval works without it) |
| `NUGGET_EMBEDDING_MODEL` | default `paraphrase-multilingual-MiniLM-L12-v2` |
| `NUGGET_TOP_K` · `NUGGET_CHUNK_SIZE` · `NUGGET_MAX_UPLOAD_MB` | `6` · `1000` · `25` |

## ✦ Endpoints

| | |
| --- | --- |
| `POST /query/stream` | SSE: `sources` → `token`… → `done` |
| `POST /query` | `{ question, history?, doc_ids?, locale? }` → answer + sources + timings |
| `POST /ingest/file` · `/ingest/text` | index a file or pasted text |
| `GET /documents` · `DELETE /documents/{id}` | library |
| `GET /documents/{id}/chunks` · `GET /health` | passages · status |

```bash
uv run pytest && uv run ruff check .   # offline: fake embedder + fake LLM
```

<div align="center"><sub>made with 🟡 by <a href="https://github.com/colombefioren">colombefioren</a></sub></div>
