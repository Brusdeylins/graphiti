#!/bin/bash
# ============================================
# Graphiti MCP Server - Build Script
# ============================================

set -e

# Configuration
IMAGE_NAME="graphiti-mcp-custom"
IMAGE_TAG="latest"
GRAPHITI_CORE_VERSION="${GRAPHITI_CORE_VERSION:-0.23.1}"
MCP_SERVER_VERSION="1.0.1-custom"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building Graphiti MCP Server (Custom)${NC}"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Graphiti Core: ${GRAPHITI_CORE_VERSION}"
echo "  MCP Server: ${MCP_SERVER_VERSION}"
echo ""

# Build the image
docker build \
    --build-arg GRAPHITI_CORE_VERSION="${GRAPHITI_CORE_VERSION}" \
    --build-arg MCP_SERVER_VERSION="${MCP_SERVER_VERSION}" \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_NAME}:${MCP_SERVER_VERSION}" \
    .

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "To use this image, update your docker-compose.yml:"
echo "  image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Or run directly:"
echo "  docker run -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"
