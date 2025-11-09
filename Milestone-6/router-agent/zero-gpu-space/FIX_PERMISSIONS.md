# Fix GCP Permissions for Deployment

## Current Issue

Your account (`jameswilsonlearnsrocode@gmail.com`) needs permissions on project `spherical-gate-477614-q7`.

## Required Permissions

You need one of these roles on the project:
- **Editor** (recommended) - Full access except billing/admin
- **Owner** - Full access including billing
- Or these specific roles:
  - `roles/cloudbuild.builds.editor`
  - `roles/run.admin`
  - `roles/serviceusage.serviceUsageConsumer`
  - `roles/storage.admin`

## Steps to Fix

### Step 1: Grant Permissions

1. Visit: https://console.cloud.google.com/iam-admin/iam/project?project=spherical-gate-477614-q7
2. Click **"Grant Access"** or **"Add Principal"**
3. Enter your email: `jameswilsonlearnsrocode@gmail.com`
4. Select role: **Editor** (or **Owner**)
5. Click **Save**
6. **Wait 2-3 minutes** for permissions to propagate

### Step 2: Enable APIs

After permissions are granted, run:

```bash
gcloud config set project spherical-gate-477614-q7
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com
```

### Step 3: Deploy

```bash
cd Milestone-6/router-agent/zero-gpu-space
export GCP_PROJECT_ID="spherical-gate-477614-q7"
./deploy-cloud-build.sh
```

## Alternative: Use Service Account

If you can't get user permissions, create a service account:

```bash
# Create service account
gcloud iam service-accounts create router-agent-deploy \
    --display-name="Router Agent Deployment" \
    --project=spherical-gate-477614-q7

# Grant permissions
gcloud projects add-iam-policy-binding spherical-gate-477614-q7 \
    --member="serviceAccount:router-agent-deploy@spherical-gate-477614-q7.iam.gserviceaccount.com" \
    --role="roles/editor"

# Use service account
gcloud auth activate-service-account router-agent-deploy@spherical-gate-477614-q7.iam.gserviceaccount.com \
    --key-file=path/to/key.json
```

## Quick Test

Test if permissions are working:

```bash
gcloud projects describe spherical-gate-477614-q7
```

If this works, you're ready to deploy!

