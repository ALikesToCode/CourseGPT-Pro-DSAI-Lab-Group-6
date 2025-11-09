#!/bin/bash
# Google Cloud Platform deployment script
# Usage: ./deploy-gcp.sh [cloud-run|compute-engine]

set -e

PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="router-agent"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Deploying Router Agent to Google Cloud Platform${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Please install it: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker.${NC}"
    exit 1
fi

# Authenticate if needed
echo -e "${YELLOW}📋 Checking authentication...${NC}"
gcloud auth configure-docker --quiet || true

# Set project
echo -e "${YELLOW}📋 Setting project to ${PROJECT_ID}...${NC}"
gcloud config set project ${PROJECT_ID}

DEPLOYMENT_TYPE=${1:-"cloud-run"}

if [ "$DEPLOYMENT_TYPE" == "cloud-run" ]; then
    echo -e "${GREEN}📦 Building Docker image...${NC}"
    docker build -t ${IMAGE_NAME}:latest .
    
    echo -e "${GREEN}📤 Pushing image to Container Registry...${NC}"
    docker push ${IMAGE_NAME}:latest
    
    echo -e "${GREEN}🚀 Deploying to Cloud Run...${NC}"
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_NAME}:latest \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 7860 \
        --memory 8Gi \
        --cpu 4 \
        --timeout 3600 \
        --max-instances 10 \
        --set-env-vars "GRADIO_SERVER_NAME=0.0.0.0,GRADIO_SERVER_PORT=7860" \
        --quiet
    
    echo -e "${GREEN}✅ Deployment complete!${NC}"
    SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')
    echo -e "${GREEN}🌐 Service URL: ${SERVICE_URL}${NC}"
    
elif [ "$DEPLOYMENT_TYPE" == "compute-engine" ]; then
    echo -e "${GREEN}📦 Building Docker image...${NC}"
    docker build -t ${IMAGE_NAME}:latest .
    
    echo -e "${GREEN}📤 Pushing image to Container Registry...${NC}"
    docker push ${IMAGE_NAME}:latest
    
    echo -e "${YELLOW}⚠️  Compute Engine deployment requires manual VM setup.${NC}"
    echo -e "${YELLOW}   See deploy-compute-engine.sh for GPU instance setup.${NC}"
    
else
    echo -e "${RED}❌ Unknown deployment type: ${DEPLOYMENT_TYPE}${NC}"
    echo -e "${YELLOW}Usage: ./deploy-gcp.sh [cloud-run|compute-engine]${NC}"
    exit 1
fi

