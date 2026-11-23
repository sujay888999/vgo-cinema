#!/bin/bash
# ==============================================================================
# VGO Cinema - Server Deployment Script
# Run this on the Google Cloud VM after uploading the project files.
# ==============================================================================
set -e

PROJECT_DIR="${1:-/opt/vgo-cinema}"

echo "============================================"
echo "  VGO Cinema - Server Deployment"
echo "============================================"

# --- 1. Install Docker ---
echo "[1/5] Installing Docker..."
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to log out and back in for group access."
else
    echo "Docker already installed: $(docker --version)"
    # Ensure compose plugin is available
    if ! docker compose version &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq docker-compose-plugin
    fi
fi

# --- 2. Prepare project directory ---
echo "[2/5] Preparing project directory: $PROJECT_DIR"
sudo mkdir -p "$PROJECT_DIR"
sudo chown -R "$USER:$USER" "$PROJECT_DIR"

# --- 3. Configure .env ---
echo "[3/5] Configuring .env..."
if [ ! -f "$PROJECT_DIR/backend/.env" ]; then
    echo "WARNING: backend/.env not found. Copy from backend/.env.example"
    echo "         and fill in your API keys before deploying."
    exit 1
fi

# Update CORS for server deployment
sed -i.bak "s|^CORS_ALLOW_ORIGINS=.*|CORS_ALLOW_ORIGINS=http://$(hostname -I 2>/dev/null | awk '{print $1}'):3000,http://localhost:3000,http://127.0.0.1:3000|" "$PROJECT_DIR/backend/.env" 2>/dev/null || true

# Ensure Redis is enabled for server deployment
sed -i.bak "s|^ENABLE_REDIS=.*|ENABLE_REDIS=true|" "$PROJECT_DIR/backend/.env" 2>/dev/null || true

# --- 4. Create output directory ---
echo "[4/5] Creating output directories..."
sudo mkdir -p /data/outputs
sudo chown -R "$USER:$USER" /data/outputs

# --- 5. Build and start services ---
echo "[5/5] Building and starting services..."
cd "$PROJECT_DIR"

# Build images first
docker compose build --no-cache

# Start services in detached mode
docker compose up -d

# Wait for backend to be healthy
echo "Waiting for backend to become healthy..."
for i in $(seq 1 30); do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/health 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo "Backend is healthy!"
        break
    fi
    echo "  waiting... ($i/30)"
    sleep 3
done

# Show service status
echo ""
echo "============================================"
echo "  Deployment Complete!"
echo "============================================"
docker compose ps

VM_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "Frontend:  http://$VM_IP:3000"
echo "Backend:   http://$VM_IP:8081/api"
echo "Health:    http://$VM_IP:8081/api/health"
echo ""
echo "Logs: docker compose logs -f"
echo "Stop: docker compose down"
