#!/bin/bash
# ==============================================================================
# VGO Cinema - Upload & Deploy to Google Cloud VM
# Run this on your local machine after creating the VM.
# ==============================================================================
set -e

# --- Configuration ---
VM_IP="${1:-}"
VM_USER="${2:-$USER}"
PROJECT_DIR="${3:-/opt/vgo-cinema}"
SSH_KEY="${SSH_KEY:-~/.ssh/google_cloud_key}"

if [ -z "$VM_IP" ]; then
    echo "Usage: $0 <VM_IP> [VM_USER] [PROJECT_DIR]"
    echo "  VM_IP       - The public IP of your Google Cloud VM"
    echo "  VM_USER     - SSH user (default: current user)"
    echo "  PROJECT_DIR - Deploy directory on VM (default: /opt/vgo-cinema)"
    exit 1
fi

echo "============================================"
echo "  VGO Cinema - Upload & Deploy"
echo "============================================"
echo "VM: $VM_IP (user: $VM_USER)"
echo "Deploy dir: $PROJECT_DIR"
echo ""

# --- 1. Verify SSH connectivity ---
echo "[1/4] Testing SSH connectivity..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=10 "$VM_USER@$VM_IP" "echo 'SSH OK'"

# --- 2. Upload project files ---
echo "[2/4] Uploading project files..."
# Exclude node_modules, __pycache__, .git, data, etc.
rsync -avz -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new" \
    --exclude='node_modules/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='data/' \
    --exclude='.env' \
    --exclude='dist/' \
    ./ "$VM_USER@$VM_IP:$PROJECT_DIR/"

# --- 3. Upload .env separately (in case it has secrets) ---
echo "[3/4] Uploading .env configuration..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
    ./backend/.env "$VM_USER@$VM_IP:$PROJECT_DIR/backend/.env"

# --- 4. Run deployment script on VM ---
echo "[4/4] Running deployment on VM..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
    "$VM_USER@$VM_IP" "cd $PROJECT_DIR && chmod +x deploy/deploy.sh && bash deploy/deploy.sh '$PROJECT_DIR'"

echo ""
echo "============================================"
echo "  Deployment initiated!"
echo "============================================"
