# DarwinTrade — web UI + HTTP API.
#
# Build:  docker build -t darwintrade .
# Run:    docker run --rm -p 8000:8000 --env-file .env darwintrade
#
# The image ships no credentials. Supply LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
# and any market-data keys at run time via --env-file or -e.
FROM python:3.12-slim

# Build toolchain is needed by cvxpy/scipy wheels on some platforms; dropped
# from the final image would require a multi-stage build, which is not worth the
# complexity for a research tool.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so edits to source do not invalidate the install layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt "fastapi>=0.115,<1.0" "uvicorn>=0.30,<1.0"

COPY pyproject.toml README.md LICENSE ./
COPY darwintrade ./darwintrade
COPY backtest ./backtest
COPY skills ./skills
RUN pip install --no-cache-dir --no-deps -e .

# Cache and session state live here. Mount a volume to keep them across runs;
# without one, memory resets when the container is removed.
ENV DARWINTRADE_STORAGE_DIR=/data
RUN mkdir -p /data/cache /data/live/sessions

# Runs as a non-root user; /data must be writable by it.
RUN useradd --create-home --uid 10001 darwin \
    && chown -R darwin:darwin /data /app
USER darwin

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# Binds 0.0.0.0 because the port is published to the host. The API has no
# authentication — do not publish it to an untrusted network without putting
# auth and rate limiting in front of it.
CMD ["python", "-m", "darwintrade.live.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
