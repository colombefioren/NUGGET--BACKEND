# Portable image for Hugging Face Spaces (Docker SDK, port 7860), Render, or any
# platform that injects $PORT. No system packages needed: embeddings are a remote
# API call (see app/embeddings.py), not an in-process model.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first so they're cached across rebuilds that only touch app code.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

# HF Spaces (Docker SDK) expects the container to listen on 7860 by default;
# most other hosts inject their own $PORT and this falls back to that.
ENV PORT=7860
EXPOSE 7860

# Spaces containers run as a non-root user; the writable dir must exist and be owned by it.
RUN useradd -m -u 1000 app \
    && mkdir -p /app/data/chroma \
    && chown -R app:app /app
USER app

# --no-sync: the venv was already built above; skip re-checking the lockfile
# on every boot, which otherwise re-downloads dev-only deps (ruff, pytest) each start.
CMD ["sh", "-c", "uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
