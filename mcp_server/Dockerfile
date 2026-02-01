# ============================================
# Graphiti MCP Server - Custom Build
# ============================================
# Standalone MCP server (ohne FalkorDB - nutzt externen Container)

FROM python:3.11-slim

WORKDIR /app/mcp

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Configure uv for optimal Docker usage
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MCP_SERVER_HOST="0.0.0.0" \
    PYTHONUNBUFFERED=1

# Accept graphiti-core version as build argument
ARG GRAPHITI_CORE_VERSION=0.23.1

# Copy project files for dependency installation
COPY pyproject.toml uv.lock ./

# Remove local path override and set specific graphiti-core version
RUN sed -i '/\[tool\.uv\.sources\]/,/graphiti-core/d' pyproject.toml && \
    if [ -n "${GRAPHITI_CORE_VERSION}" ]; then \
      sed -i "s/graphiti-core\[falkordb\]>=[0-9]\+\.[0-9]\+\.[0-9]\+$/graphiti-core[falkordb]==${GRAPHITI_CORE_VERSION}/" pyproject.toml; \
    fi && \
    rm -f uv.lock && \
    uv lock

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-group dev

# Store graphiti-core version
RUN echo "${GRAPHITI_CORE_VERSION}" > /app/mcp/.graphiti-core-version

# Copy application code
COPY main.py ./
COPY src/ ./src/
COPY config/ ./config/

# Create log directory
RUN mkdir -p /var/log/graphiti

# Add version labels
ARG MCP_SERVER_VERSION=1.0.1-custom
ARG BUILD_DATE
LABEL org.opencontainers.image.title="Graphiti MCP Server (Custom)" \
      org.opencontainers.image.description="Graphiti MCP server with entity type fields support" \
      org.opencontainers.image.version="${MCP_SERVER_VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      graphiti.core.version="${GRAPHITI_CORE_VERSION}"

# Expose MCP server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run MCP server
CMD ["/root/.local/bin/uv", "run", "--no-sync", "main.py"]
