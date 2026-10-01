FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    AGENTMESH_STATE_DIR=/var/lib/agentmesh/state

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create isolated state directory
RUN mkdir -p /var/lib/agentmesh/state

COPY pyproject.toml .
RUN pip install --no-cache-dir hatchling pydantic anyio pytest pytest-asyncio

COPY . .

CMD ["python", "examples/chaos_recovery_demo.py"]
