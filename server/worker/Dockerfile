FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.12 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
COPY  pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY . .
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]