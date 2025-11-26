# Trader-AI Railway Deployment Guide

## Quick Start

### 1. Push to GitHub
```bash
git add Dockerfile railway.toml .dockerignore .env.railway.example
git add src/dashboard/run_server_simple.py src/dashboard/constants.py
git add src/dashboard/frontend/vite.config.ts
git commit -m "feat: Add Railway deployment configuration"
git push origin main
```

### 2. Deploy to Railway

1. Go to [railway.app](https://railway.app)
2. Click "New Project" > "Deploy from GitHub repo"
3. Select your trader-ai repository
4. Railway auto-detects Dockerfile and builds

### 3. Configure Environment Variables

In Railway dashboard, add these variables:

**Required:**
- `TRADING_MODE` = `paper`
- `ALPACA_API_KEY` = your key
- `ALPACA_SECRET_KEY` = your secret
- `JWT_SECRET_KEY` = (generate: `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
- `DATABASE_ENCRYPTION_KEY` = (generate: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`)

**Auto-set by Railway:**
- `PORT` (do not override)
- `RAILWAY_PUBLIC_DOMAIN` (for CORS)

### 4. Verify Deployment

Once deployed, check:
- Health: `https://your-app.railway.app/health`
- API: `https://your-app.railway.app/api/health`
- Dashboard: `https://your-app.railway.app/`

## Architecture

```
                    Railway Container
+--------------------------------------------------+
|                                                  |
|   +------------------+   +-------------------+   |
|   |  React Frontend  |   |   FastAPI Backend |   |
|   |  (Built Static)  |-->|   (uvicorn)       |   |
|   +------------------+   +-------------------+   |
|          dist/                  |               |
|                                 v               |
|                    +------------------------+   |
|                    | WebSocket + REST API   |   |
|                    +------------------------+   |
|                                                  |
+--------------------------------------------------+
            PORT (Railway-assigned)
```

## Files Modified/Created

| File | Change |
|------|--------|
| `Dockerfile` | NEW - Multi-stage build (Node + Python) |
| `railway.toml` | NEW - Railway configuration |
| `.dockerignore` | NEW - Optimize build context |
| `.env.railway.example` | NEW - Environment template |
| `src/dashboard/run_server_simple.py` | Modified - PORT env, 0.0.0.0, static serving |
| `src/dashboard/constants.py` | Modified - Railway CORS support |
| `src/dashboard/frontend/vite.config.ts` | Modified - Production build settings |

## Troubleshooting

### Build Fails
- Check `npm ci` runs successfully for frontend
- Verify `requirements.txt` has all dependencies

### Health Check Fails
- Ensure `/health` endpoint responds
- Check logs: `railway logs`

### CORS Errors
- Add your domain to `CORS_ALLOW_ORIGIN` env var
- Verify `RAILWAY_PUBLIC_DOMAIN` is set

### WebSocket Not Connecting
- Railway supports WebSocket on same port
- Check browser console for connection errors

## Local Testing

```bash
# Build Docker image locally
docker build -t trader-ai .

# Run container
docker run -p 8000:8000 \
  -e TRADING_MODE=paper \
  -e ALPACA_API_KEY=test \
  -e ALPACA_SECRET_KEY=test \
  trader-ai

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/health
```
