FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.12 /uv /uvx /bin/

# Which torch build to install: "cpu" (default) or "gpu" (CUDA 12.6, run with --gpus all)
ARG TORCH=cpu

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
COPY  pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-default-groups --group ${TORCH}
COPY . .
# --no-sync keeps uv run from swapping the installed torch group back to the default
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
