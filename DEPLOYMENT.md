# SepsisGuard v3.0 — Technical Deployment & Monitoring Specification

This guide documents environment configuration, Docker containerization, production WSGI server deployment, health endpoint monitoring, and technical vs clinical deployment boundaries.

---

## ⚠️ Important Boundary: Technical vs Clinical Deployment

> **CRITICAL NOTICE**:
> Deploying this application technical stack (via Docker or WSGI server) provides a **technical demonstration system**.
> It **DOES NOT** constitute clinical deployment for active patient diagnosis or care.
> Prospective multi-center clinical trials, IRB approval, HIPAA audit, and SaMD regulatory clearance are required prior to any clinical deployment.

---

## 1. Environment Configuration

Copy `.env.example` to `.env` in the project root:

```bash
cp .env.example .env
```

### Environment Variables Matrix

| Variable | Required | Default / Description |
| :--- | :---: | :--- |
| `FLASK_SECRET_KEY` | **Yes** | Secret key for session encryption. Must be set in production. |
| `API_KEY` | **Yes** | Shared key for `X-API-Key` REST header and Socket.IO auth. |
| `GEMINI_API_KEY` | Optional | Google Gemini API key for narrative synthesis. Uses local fallback if empty. |
| `FRONTEND_ORIGIN` | **Yes** | Explicit CORS allow-list (`http://localhost:5000,http://127.0.0.1:5000`). |
| `PREDICT_RATE_LIMIT`| Optional | Rate limit threshold (`60 per minute`). |

---

## 2. Docker Container Deployment

### Build & Run Container

```bash
# Build production container
docker build -t sepsisguard:v3.0 .

# Run container with environment file
docker run -d \
  --name sepsisguard_app \
  -p 5000:5000 \
  --env-file .env \
  sepsisguard:v3.0
```

### Docker Compose

```bash
# Launch backend service with health checks
docker-compose up -d --build

# View container logs
docker-compose logs -f
```

---

## 3. Health & Status Monitoring

The `/health` endpoint exposes real-time operational state for load balancers and container orchestrators:

```bash
curl http://localhost:5000/health
```

### Example Health Response

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v2_2026-08-12",
  "last_successful_inference": "2026-08-12T18:30:00.000000",
  "active_patients": 6,
  "service": "SepsisGuard AI v3.0 Telemetry Engine"
}
```

- **HTTP 200 OK**: Model loaded and ready.
- **HTTP 503 Service Unavailable**: Model failed to load (`status: degraded`).

---

## 4. Production WSGI Server

In production environments, execute SepsisGuard using `gunicorn` with eventlet worker support or gevent for Flask-SocketIO compatibility:

```bash
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 backend.app:app
```
