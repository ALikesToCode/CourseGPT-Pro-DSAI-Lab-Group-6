# Quick Deploy to Google Cloud Run

## Current Status

**Issue**: Permission denied on project `light-quest-475608-k7`

## Quick Fix Options

### Option 1: Fix Permissions (Recommended)

1. **Grant yourself permissions** on the existing project:
   - Visit: https://console.cloud.google.com/iam-admin/iam/project?project=light-quest-475608-k7
   - Add role: `Editor` or `Owner` to your account (`jameswilsonlearnsrocode@gmail.com`)
   - Wait 2-3 minutes for propagation

2. **Enable APIs**:
   ```bash
   gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com --project=light-quest-475608-k7
   ```

3. **Deploy**:
   ```bash
   cd Milestone-6/router-agent/zero-gpu-space
   ./deploy-cloud-build.sh
   ```

### Option 2: Create New Project

```bash
# Run setup helper
./setup-gcp-permissions.sh

# Or manually:
gcloud projects create router-agent-deploy --name="Router Agent Deployment"
gcloud config set project router-agent-deploy
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com

# Then deploy
./deploy-cloud-build.sh
```

### Option 3: Use Different Account/Project

If you have access to another project:

```bash
export GCP_PROJECT_ID="your-other-project-id"
gcloud config set project ${GCP_PROJECT_ID}
./deploy-cloud-build.sh
```

## Manual Deployment Steps

If scripts don't work, deploy manually:

```bash
# 1. Set project
gcloud config set project YOUR_PROJECT_ID

# 2. Enable APIs (if not already enabled)
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com

# 3. Submit build
cd Milestone-6/router-agent/zero-gpu-space
gcloud builds submit --config=cloudbuild.yaml .

# 4. Or build and deploy separately
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/router-agent:latest .
gcloud run deploy router-agent \
    --image gcr.io/YOUR_PROJECT_ID/router-agent:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 7860 \
    --memory 8Gi \
    --cpu 4
```

## Required Permissions

Your account needs these roles:
- `roles/cloudbuild.builds.editor` - To build images
- `roles/run.admin` - To deploy to Cloud Run
- `roles/serviceusage.serviceUsageConsumer` - To enable APIs
- `roles/storage.admin` - To push to Container Registry

Or use `roles/editor` or `roles/owner` for full access.

## Troubleshooting

**Permission Denied**: 
- Check IAM: https://console.cloud.google.com/iam-admin/iam
- Ensure billing is enabled
- Wait 2-3 minutes after granting permissions

**API Not Enabled**:
- Enable manually: https://console.cloud.google.com/apis/library
- Or use: `gcloud services enable <api-name>`

**Build Fails**:
- Check logs: `gcloud builds list` then `gcloud builds log BUILD_ID`
- Ensure Dockerfile is correct
- Check requirements.txt for compatibility

