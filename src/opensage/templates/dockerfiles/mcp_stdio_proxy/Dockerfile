FROM python:3.12-slim

# Node.js (for npx-based MCP servers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# uv (for uvx-based MCP servers)
RUN pip install --no-cache-dir uv

# mcp-proxy (stdio→SSE bridge)
RUN pip install --no-cache-dir mcp-proxy

ENTRYPOINT ["mcp-proxy"]
