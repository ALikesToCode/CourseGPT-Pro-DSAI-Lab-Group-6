# Manual Deployment Steps (Cloud Console)

Since there are permission propagation issues, here's how to deploy manually via Cloud Console:

## Step 1: Enable APIs via Cloud Console

1. **Cloud Build API**: https://console.cloud.google.com/apis/library/cloudbuild.googleapis.com?project=spherical-gate-477614-q7
   - Click "Enable"

2. **Cloud Run API**: https://console.cloud.google.com/apis/library/run.googleapis.com?project=spherical-gate-477614-q7
   - Click "Enable"

3. **Container Registry API**: https://console.cloud.google.com/apis/library/containerregistry.googleapis.com?project=spherical-gate-477614-q7
   - Click "Enable"

## Step 2: Create Cloud Build Trigger (Optional)

Or use Cloud Shell:

```bash
# In Cloud Shell
cd Milestone-6/router-agent/zero-gpu-space
gcloud builds submit --tag gcr.io/spherical-gate-477614-q7/router-agent:latest .
```

## Step 3: Deploy to Cloud Run

After image is built:

```bash
gcloud run deploy router-agent \
    --image gcr.io/spherical-gate-477614-q7/router-agent:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 7860 \
    --memory 8Gi \
    --cpu 4 \
    --timeout 3600 \
    --set-env-vars "GRADIO_SERVER_NAME=0.0.0.0,GRADIO_SERVER_PORT=7860"
```

## Alternative: Use Cloud Shell

Cloud Shell has proper permissions. Open:
https://shell.cloud.google.com/?project=spherical-gate-477614-q7

Then run:
```bash
git clone https://github.com/ALikesToCode/CourseGPT-Pro-DSAI-Lab-Group-6.git
cd CourseGPT-Pro-DSAI-Lab-Group-6/Milestone-6/router-agent/zero-gpu-space
./deploy-cloud-build.sh
```

