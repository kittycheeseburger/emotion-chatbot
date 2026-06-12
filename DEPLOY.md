# Deployment

This project is configured for a single Render Web Service.

## Render Blueprint

1. Push this repository to GitHub.
2. In Render, create a new Blueprint from the repository.
3. Render reads `render.yaml` and creates the `emotion-chatbot` service.

The service build does two things:

```bash
pip install -r backend/requirements.deploy.txt
cd frontend
npm ci
npm run build
```

The service starts FastAPI, and FastAPI serves both:

- `/` and frontend routes from `frontend/dist`
- `/api/*` from the backend API

## Production Mode

The Render config sets:

```text
CHAT_REPLY_MODE=rag
APP_ENV=production
```

This keeps the deployment lightweight and avoids uploading the local 3GB model
directory. If you want online LLM replies, add these Render environment
variables and change `CHAT_REPLY_MODE`:

```text
CHAT_REPLY_MODE=glm
GLM_API_KEY=your_key
GLM_API_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=glm-4.7-flash
```

For OpenAI mode:

```text
CHAT_REPLY_MODE=openai
OPENAI_API_KEY=your_key
OPENAI_API_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```
