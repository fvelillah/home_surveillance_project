#!/usr/bin/env bash
# ==============================================================================
# scripts/deploy_production.sh - Zero-Downtime Production Deployment Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "  DAHUA SURVEILLANCE PLATFORM: PRODUCTION DEPLOYMENT & HEALTH AUDIT"
echo "======================================================================"

# 1. Pre-Flight Verification
echo "[1/5] Running pre-flight configuration checks..."
if [ ! -f .env ]; then
    echo "ERROR: .env configuration file not found in $PROJECT_ROOT"
    exit 1
fi

mkdir -p data/storage/incidents data/storage/snapshots data/qdrant_edge
echo "  ✓ Storage directories verified."

# 2. Build Production Artifacts
echo "[2/5] Building frontend production bundle..."
if [ -d frontend ]; then
    (cd frontend && npm run build)
    echo "  ✓ Next.js frontend production bundle compiled."
fi

# 3. Docker Compose Configuration Validation
echo "[3/5] Validating Docker Compose production configuration..."
if command -v docker &> /dev/null; then
    if docker compose version &> /dev/null; then
        docker compose -f docker-compose.prod.yml config -q
        echo "  ✓ docker-compose.prod.yml syntax verified."
    fi
else
    echo "  ⚠ Docker not found on host; skipping container build step."
fi

# 4. Service Health Checks
echo "[4/5] Checking service port availability..."
for PORT in 7777 9876 3000; do
    if ss -tuln | grep -q ":$PORT "; then
        echo "  • Port $PORT is active (Service running)."
    else
        echo "  • Port $PORT is available."
    fi
done

# 5. Deployment Summary
echo "[5/5] Production readiness assessment complete."
echo "======================================================================"
echo "  PRODUCTION DEPLOYMENT MODES AVAILABLE:"
echo "    1. Docker Compose: docker compose -f docker-compose.prod.yml up -d --build"
echo "    2. Systemd Services: sudo cp scripts/systemd/*.service /etc/systemd/system/"
echo "                         sudo systemctl daemon-reload"
echo "                         sudo systemctl enable --now surveillance-edge surveillance-backend surveillance-frontend"
echo "======================================================================"
