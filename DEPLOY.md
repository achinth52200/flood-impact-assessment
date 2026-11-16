# Deployment Guide

## Quick tunnel (no account, temporary)
```bash
python deploy_tunnel.py          # requires NGROK_AUTHTOKEN env var
# OR use cloudflared (no auth):
cloudflared tunnel --url http://localhost:8501
```

---

## Streamlit Community Cloud (free, permanent)

1. Push repo to GitHub:
```bash
git remote add origin https://github.com/YOUR_USERNAME/flood-impact-assessment.git
git push -u origin main
```

2. Go to https://share.streamlit.io → **New app**
3. Select your repo, branch `main`, file `app.py`
4. Add secret in Settings → Secrets:
```toml
HF_REPO = "YOUR_HF_USERNAME/flood-model"
```
5. Click **Deploy**

---

## Hugging Face Spaces (free, GPU available)

1. Create a Space at https://huggingface.co/new-space
   - SDK: **Streamlit**
   - Hardware: CPU Basic (free) or T4 GPU

2. Upload checkpoint to HF Hub:
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="models/checkpoints/best_model.pth",
    path_in_repo="best_model.pth",
    repo_id="YOUR_USERNAME/flood-model",
    repo_type="model",
)
```

3. Add `HF_REPO=YOUR_USERNAME/flood-model` to Space secrets.

4. Push code:
```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/flood-assessment
git push space main
```

---

## Docker / Render / Railway

```bash
# Build
docker build -t flood-app .

# Run locally
docker run -p 8501:8501 flood-app

# Deploy to Render: connect GitHub repo, set Dockerfile as build method
# Deploy to Railway: railway up
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `HF_REPO` | HuggingFace repo ID with best_model.pth |
| `NGROK_AUTHTOKEN` | ngrok auth token for tunnel |
