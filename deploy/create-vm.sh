#!/bin/bash
# ==============================================================================
# VGO Cinema - Create Google Cloud VM (Free Tier)
# Run this after gcloud is installed and authenticated.
# ==============================================================================
set -e

ZONE="${1:-us-west1-b}"
INSTANCE_NAME="${2:-vgo-cinema}"
MACHINE_TYPE="${3:-e2-micro}"  # Always-free tier

echo "============================================"
echo "  Creating Google Cloud VM"
echo "============================================"
echo "Instance: $INSTANCE_NAME"
echo "Machine:  $MACHINE_TYPE"
echo "Zone:     $ZONE"
echo ""

# --- 1. Create the VM instance ---
echo "[1/4] Creating VM instance..."
gcloud compute instances create "$INSTANCE_NAME" \
    --machine-type="$MACHINE_TYPE" \
    --zone="$ZONE" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=30GB \
    --boot-disk-type=pd-standard \
    --tags=http-server,https-server \
    --no-address \
    --metadata=enable-docker=true

# --- 2. Get the external IP ---
echo ""
echo "[2/4] Getting VM external IP..."
VM_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "")

if [ -z "$VM_IP" ]; then
    # Wait a moment for the IP to be assigned
    sleep 5
    VM_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "")
fi

echo "VM IP: $VM_IP"

# --- 3. Create a new SSH key pair (or use existing) ---
echo ""
echo "[3/4] Setting up SSH access..."
SSH_KEY_NAME="vgo-cinema-key"
SSH_DIR="$HOME/.ssh"
SSH_KEY="$SSH_DIR/${SSH_KEY_NAME}"
SSH_PUB="$SSH_KEY.pub"

mkdir -p "$SSH_DIR"
if [ ! -f "$SSH_KEY" ]; then
    ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "vgo-cinema-deploy"
    echo "Generated new SSH key: $SSH_KEY"
fi

# Add public key to the VM
cat "$SSH_PUB" | gcloud compute os-login ssh-keys add \
    --zone="$ZONE" \
    --instance="$INSTANCE_NAME" \
    --public-key-file="$SSH_PUB" 2>/dev/null || \
gcloud compute instances add-metadata "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --metadata=ssh-keys="$(cat "$SSH_PUB")"

# --- 4. Test SSH ---
echo ""
echo "[4/4] Testing SSH connection..."
ssh -o StrictHostKeyChecking=accept-new \
    -i "$SSH_KEY" \
    "ubuntu@$VM_IP" "echo 'SSH connection successful'"

echo ""
echo "============================================"
echo "  VM Created Successfully!"
echo "============================================"
echo "Instance:  $INSTANCE_NAME"
echo "IP:        $VM_IP"
echo "Zone:      $ZONE"
echo "SSH Key:   $SSH_KEY"
echo ""
echo "SSH command:"
echo "  ssh -i $SSH_KEY ubuntu@$VM_IP"
echo ""
echo "Next step:"
echo "  bash deploy/upload-and-deploy.sh $VM_IP ubuntu /opt/vgo-cinema"
echo ""
echo "IMPORTANT: Set SSH_KEY env var to the key path:"
echo "  SSH_KEY=$SSH_KEY bash deploy/upload-and-deploy.sh $VM_IP ubuntu /opt/vgo-cinema"
