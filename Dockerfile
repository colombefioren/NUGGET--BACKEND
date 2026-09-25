# Portable image for Hugging Face Spaces (Docker SDK, port 7860) or any
# platform that injects $PORT (Cloud Run, Render, Fly, ...).
FROM python:3.12-slim

# libgomp1 is required by onnxruntime (used by FastEmbed) at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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

# Spaces containers run as a non-root user; writable dirs must exist and be owned by it.
RUN useradd -m -u 1000 app \
    && mkdir -p /app/data/chroma /app/data/models \
    && chown -R app:app /app
USER app

# --no-sync: the venv was already built above; skip re-checking the lockfile
# on every boot, which otherwise re-downloads dev-only deps (ruff, pytest) each start.
CMD ["sh", "-c", "uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
