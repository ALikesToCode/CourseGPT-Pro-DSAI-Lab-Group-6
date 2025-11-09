# Google Cloud Platform Deployment Guide

This guide covers deploying the Router Agent application to Google Cloud Platform with GPU support.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **gcloud CLI** installed and configured
   ```bash
   curl https://sdk.cloud.google.com | bash
   gcloud init
   ```
3. **Docker** installed locally
4. **HF_TOKEN** environment variable set (for accessing private models)

## Deployment Options

### Option 1: Cloud Run (Serverless, CPU only)

**Pros:**
- Serverless, pay-per-use
- Auto-scaling
- No VM management

**Cons:**
- No GPU support (CPU inference only)
- Cold starts
- Limited to 8GB memory

**Steps:**

```bash
# Set your project ID
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"

# Make script executable
chmod +x deploy-gcp.sh

# Deploy to Cloud Run
./deploy-gcp.sh cloud-run
```

**Cost:** ~$0.10-0.50/hour when active (depends on traffic)

### Option 2: Compute Engine with GPU (Recommended for Production)

**Pros:**
- Full GPU support (T4, V100, A100)
- Persistent instance
- Better for long-running workloads
- Lower latency (no cold starts)

**Cons:**
- Requires VM management
- Higher cost for always-on instances

**Steps:**

```bash
# Set your project ID and zone
export GCP_PROJECT_ID="your-project-id"
export GCP_ZONE="us-central1-a"
export HF_TOKEN="your-huggingface-token"

# Make script executable
chmod +x deploy-compute-engine.sh

# Deploy to Compute Engine
./deploy-compute-engine.sh
```

**GPU Options:**
- **T4** (nvidia-tesla-t4): ~$0.35/hour - Good for 27B-32B models with quantization
- **V100** (nvidia-tesla-v100): ~$2.50/hour - Better performance
- **A100** (nvidia-a100): ~$3.50/hour - Best performance for large models

**Cost:** GPU instance + storage (~$0.35-3.50/hour depending on GPU type)

## Manual Deployment Steps

### 1. Build and Push Docker Image

```bash
# Authenticate Docker
gcloud auth configure-docker

# Set project
gcloud config set project YOUR_PROJECT_ID

# Build image
docker build -t gcr.io/YOUR_PROJECT_ID/router-agent:latest .

# Push to Container Registry
docker push gcr.io/YOUR_PROJECT_ID/router-agent:latest
```

### 2. Deploy to Cloud Run (CPU)

```bash
gcloud run deploy router-agent \
    --image gcr.io/YOUR_PROJECT_ID/router-agent:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 7860 \
    --memory 8Gi \
    --cpu 4 \
    --timeout 3600 \
    --set-env-vars "HF_TOKEN=your-token,GRADIO_SERVER_NAME=0.0.0.0,GRADIO_SERVER_PORT=7860"
```

### 3. Deploy to Compute Engine (GPU)

```bash
# Create VM with GPU
gcloud compute instances create router-agent-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --image-family=cos-stable \
    --image-project=cos-cloud \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform

# SSH into instance
gcloud compute ssh router-agent-gpu --zone=us-central1-a

# On the VM, install Docker and NVIDIA runtime
# Then pull and run the container
docker pull gcr.io/YOUR_PROJECT_ID/router-agent:latest
docker run -d \
    --name router-agent \
    --gpus all \
    -p 7860:7860 \
    -e HF_TOKEN="your-token" \
    gcr.io/YOUR_PROJECT_ID/router-agent:latest
```

## Environment Variables

Set these in Cloud Run or as VM metadata:

- `HF_TOKEN`: Hugging Face access token (required for private models)
- `GRADIO_SERVER_NAME`: Server hostname (default: 0.0.0.0)
- `GRADIO_SERVER_PORT`: Server port (default: 7860)
- `ROUTER_PREFETCH_MODELS`: Comma-separated list of models to preload
- `ROUTER_WARM_REMAINING`: Set to "1" to warm remaining models

## Monitoring and Logs

### Cloud Run Logs
```bash
gcloud run services logs read router-agent --region us-central1
```

### Compute Engine Logs
```bash
gcloud compute instances get-serial-port-output router-agent-gpu --zone us-central1-a
```

## Cost Optimization

1. **Cloud Run**: Use only when needed, auto-scales to zero
2. **Compute Engine**: 
   - Use preemptible instances for 80% cost savings (with risk of termination)
   - Stop instance when not in use: `gcloud compute instances stop router-agent-gpu --zone us-central1-a`
   - Use smaller GPU types (T4) for development, larger (A100) for production

## Troubleshooting

### GPU Not Available
- Check GPU quota: `gcloud compute project-info describe --project YOUR_PROJECT_ID`
- Request quota increase if needed
- Verify GPU drivers are installed on Compute Engine VM

### Out of Memory
- Increase Cloud Run memory: `--memory 16Gi`
- Use larger VM instance type
- Enable model quantization (AWQ/BitsAndBytes)

### Cold Starts (Cloud Run)
- Use Cloud Run min-instances to keep warm
- Pre-warm models on startup
- Consider Compute Engine for always-on workloads

## Security

1. **Authentication**: Use Cloud Run authentication or Cloud IAP for Compute Engine
2. **Secrets**: Store HF_TOKEN in Secret Manager
3. **Firewall**: Restrict access to specific IP ranges
4. **HTTPS**: Use Cloud Load Balancer with SSL certificate

## Next Steps

1. Set up Cloud Load Balancer for HTTPS
2. Configure monitoring and alerts
3. Set up CI/CD with Cloud Build
4. Use Cloud Storage for model caching
5. Implement auto-scaling policies

