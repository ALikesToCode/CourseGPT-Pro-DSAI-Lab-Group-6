#!/bin/bash
# Script to help set up GCP permissions and create a new project if needed

set -e

PROJECT_ID=${GCP_PROJECT_ID:-"light-quest-475608-k7"}

echo "🔧 GCP Setup Helper"
echo "=================="
echo ""
echo "Current project: ${PROJECT_ID}"
echo ""

# Option 1: Create a new project
read -p "Create a new project? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter new project ID (lowercase, hyphens only): " NEW_PROJECT_ID
    read -p "Enter project name: " PROJECT_NAME
    
    echo "Creating project..."
    gcloud projects create ${NEW_PROJECT_ID} --name="${PROJECT_NAME}" || {
        echo "❌ Failed to create project. It may already exist or you don't have permission."
        exit 1
    }
    
    echo "Setting as active project..."
    gcloud config set project ${NEW_PROJECT_ID}
    
    echo "Enabling billing (you'll need to do this manually)..."
    echo "Visit: https://console.cloud.google.com/billing/linkedaccount?project=${NEW_PROJECT_ID}"
    read -p "Press Enter after enabling billing..."
    
    PROJECT_ID=${NEW_PROJECT_ID}
fi

# Enable required APIs
echo ""
echo "Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    --project=${PROJECT_ID} || {
    echo "⚠️  Failed to enable APIs. You may need:"
    echo "   1. Billing enabled: https://console.cloud.google.com/billing?project=${PROJECT_ID}"
    echo "   2. Permissions: roles/serviceusage.serviceUsageConsumer"
    echo "   3. Owner role on the project"
    exit 1
}

echo ""
echo "✅ Setup complete!"
echo "   Project ID: ${PROJECT_ID}"
echo ""
echo "Next steps:"
echo "   1. Set HF_TOKEN secret (if using private models)"
echo "   2. Run: ./deploy-cloud-build.sh"
echo ""

