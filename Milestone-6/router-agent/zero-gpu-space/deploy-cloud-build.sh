#!/bin/bash
# Deploy using Cloud Build (no local Docker required)
# This uses Google Cloud Build service to build and deploy

set -e

PROJECT_ID=${GCP_PROJECT_ID:-"light-quest-475608-k7"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="router-agent"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Deploying Router Agent to Cloud Run using Cloud Build${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Please install it: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

# Set project
echo -e "${YELLOW}📋 Setting project to ${PROJECT_ID}...${NC}"
gcloud config set project ${PROJECT_ID}

# Check if Cloud Build API is enabled
echo -e "${YELLOW}📋 Checking Cloud Build API...${NC}"
if ! gcloud services list --enabled --filter="name:cloudbuild.googleapis.com" --format="value(name)" | grep -q cloudbuild; then
    echo -e "${YELLOW}⚠️  Cloud Build API not enabled. Attempting to enable...${NC}"
    gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com || {
        echo -e "${RED}❌ Failed to enable APIs. Please enable them manually:${NC}"
        echo -e "${YELLOW}   https://console.cloud.google.com/apis/library/cloudbuild.googleapis.com?project=${PROJECT_ID}${NC}"
        exit 1
    }
fi

# Submit build to Cloud Build
echo -e "${GREEN}📦 Submitting build to Cloud Build...${NC}"
COMMIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")

gcloud builds submit --config=cloudbuild.yaml \
    --substitutions=_SERVICE_NAME=${SERVICE_NAME},_REGION=${REGION},COMMIT_SHA=${COMMIT_SHA} \
    . || {
    echo -e "${YELLOW}⚠️  Cloud Build failed. Trying alternative approach...${NC}"
    
    # Alternative: Build and deploy separately
    echo -e "${GREEN}📦 Building image with Cloud Build...${NC}"
    gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest . || {
        echo -e "${RED}❌ Build failed. Please check:${NC}"
        echo -e "${YELLOW}   1. Cloud Build API is enabled${NC}"
        echo -e "${YELLOW}   2. You have permissions (roles/cloudbuild.builds.editor)${NC}"
        echo -e "${YELLOW}   3. Billing is enabled for the project${NC}"
        exit 1
    }
    
    echo -e "${GREEN}🚀 Deploying to Cloud Run...${NC}"
    gcloud run deploy ${SERVICE_NAME} \
        --image gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 7860 \
        --memory 8Gi \
        --cpu 4 \
        --timeout 3600 \
        --max-instances 10 \
        --set-env-vars "GRADIO_SERVER_NAME=0.0.0.0,GRADIO_SERVER_PORT=7860" \
        --quiet || {
        echo -e "${RED}❌ Deployment failed. Please check:${NC}"
        echo -e "${YELLOW}   1. Cloud Run API is enabled${NC}"
        echo -e "${YELLOW}   2. You have permissions (roles/run.admin)${NC}"
        exit 1
    }
}

echo -e "${GREEN}✅ Deployment complete!${NC}"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)' 2>/dev/null || echo "Check Cloud Console")
echo -e "${GREEN}🌐 Service URL: ${SERVICE_URL}${NC}"

