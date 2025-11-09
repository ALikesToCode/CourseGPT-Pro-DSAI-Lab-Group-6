#!/bin/bash
# Google Cloud Compute Engine deployment script (with GPU support)
# This creates a VM instance with GPU for running the router agent

set -e

PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
ZONE=${GCP_ZONE:-"us-central1-a"}
INSTANCE_NAME="router-agent-gpu"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
IMAGE_NAME="gcr.io/${PROJECT_ID}/router-agent:latest"
BOOT_DISK_SIZE="100GB"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Setting up Compute Engine VM with GPU for Router Agent${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Please install it: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

# Set project
gcloud config set project ${PROJECT_ID}

# Check if instance already exists
if gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} &>/dev/null; then
    echo -e "${YELLOW}⚠️  Instance ${INSTANCE_NAME} already exists.${NC}"
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}🗑️  Deleting existing instance...${NC}"
        gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE} --quiet
    else
        echo -e "${GREEN}✅ Using existing instance.${NC}"
        INSTANCE_IP=$(gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
        echo -e "${GREEN}🌐 Instance IP: ${INSTANCE_IP}${NC}"
        echo -e "${YELLOW}   Access via: http://${INSTANCE_IP}:7860${NC}"
        exit 0
    fi
fi

# Create startup script
cat > /tmp/startup-script.sh << 'EOF'
#!/bin/bash
set -e

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Pull and run the container
docker pull gcr.io/PROJECT_ID/router-agent:latest
docker run -d \
    --name router-agent \
    --gpus all \
    -p 7860:7860 \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e GRADIO_SERVER_NAME=0.0.0.0 \
    -e GRADIO_SERVER_PORT=7860 \
    gcr.io/PROJECT_ID/router-agent:latest

# Install firewall rule (if needed)
gcloud compute firewall-rules create allow-router-agent \
    --allow tcp:7860 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow Router Agent Gradio UI" \
    --quiet || true
EOF

# Replace PROJECT_ID in startup script
sed -i "s/PROJECT_ID/${PROJECT_ID}/g" /tmp/startup-script.sh

echo -e "${GREEN}🖥️  Creating VM instance with GPU...${NC}"
gcloud compute instances create ${INSTANCE_NAME} \
    --zone=${ZONE} \
    --machine-type=${MACHINE_TYPE} \
    --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --image-family=cos-stable \
    --image-project=cos-cloud \
    --boot-disk-size=${BOOT_DISK_SIZE} \
    --boot-disk-type=pd-standard \
    --metadata-from-file startup-script=/tmp/startup-script.sh \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata="HF_TOKEN=${HF_TOKEN:-your-token-here}" \
    --tags=http-server,https-server

echo -e "${GREEN}✅ Instance created!${NC}"
echo -e "${YELLOW}⏳ Waiting for instance to start (this may take a few minutes)...${NC}"

# Wait for instance to be ready
sleep 30

# Get instance IP
INSTANCE_IP=$(gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo -e "${GREEN}🌐 Instance IP: ${INSTANCE_IP}${NC}"
echo -e "${YELLOW}⏳ Waiting for application to start (check logs with: gcloud compute instances get-serial-port-output ${INSTANCE_NAME} --zone=${ZONE})${NC}"
echo -e "${GREEN}📝 Access the application at: http://${INSTANCE_IP}:7860${NC}"

# Cleanup
rm -f /tmp/startup-script.sh

