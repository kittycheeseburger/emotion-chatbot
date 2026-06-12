FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.deploy.txt ./backend/requirements.deploy.txt
RUN pip install --no-cache-dir -r backend/requirements.deploy.txt

COPY backend/ ./backend/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

WORKDIR /app/backend

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
