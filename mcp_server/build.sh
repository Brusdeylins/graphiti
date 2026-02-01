#!/bin/bash
# ============================================
# Graphiti MCP Server - Local Build Script
# ============================================
# Builds MCP server with LOCAL graphiti-core source
# (includes all fork-specific features and changes)

set -e

# Resolve actual directory (follow symlink)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DIR="$(cd -P "$SCRIPT_DIR" && pwd)"

# Find graphiti-milofax root (parent of mcp_server)
if [[ "$REAL_DIR" == */mcp_server ]]; then
    GRAPHITI_ROOT="$(dirname "$REAL_DIR")"
else
    # Fallback: assume we're in graphiti-mcp symlink
    GRAPHITI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/graphiti-milofax"
fi

# Configuration
IMAGE_NAME="${IMAGE_NAME:-graphiti-mcp-custom}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Building Graphiti MCP Server (Local)${NC}"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Graphiti Root: ${GRAPHITI_ROOT}"
echo ""

# Verify graphiti_core exists
if [[ ! -d "$GRAPHITI_ROOT/graphiti_core" ]]; then
    echo -e "${RED}Error: graphiti_core not found at $GRAPHITI_ROOT/graphiti_core${NC}"
    exit 1
fi

# Verify Dockerfile exists
DOCKERFILE="$GRAPHITI_ROOT/mcp_server/docker/Dockerfile.standalone-local"
if [[ ! -f "$DOCKERFILE" ]]; then
    echo -e "${RED}Error: Dockerfile not found at $DOCKERFILE${NC}"
    exit 1
fi

# Show what will be included
echo "Including local graphiti_core with CRUD methods:"
grep -c "async def create_\|async def get_\|async def update_\|async def delete_" \
    "$GRAPHITI_ROOT/graphiti_core/graphiti.py" | xargs -I{} echo "  {} CRUD methods found"
echo ""

# Build from graphiti root with local Dockerfile
cd "$GRAPHITI_ROOT"
docker build \
    -f mcp_server/docker/Dockerfile.standalone-local \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_NAME}:local-$(date +%Y%m%d)" \
    .

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To deploy:"
echo "  cd /v/graphiti && stack rebuild"
