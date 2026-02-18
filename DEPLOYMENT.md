# Nagar Rakshak Deployment Guide (Single-Port)

This deployment is configured for English and Hindi only.

- Publicly exposed port: `7777`
- Internal-only services: `pipeline-english`, `pipeline-hindi`, `ollama`
- Reverse proxy: Nginx handles both static UI and WebSocket routing on `:7777`

## Architecture

User browser connects only to `http://<host>:7777`.

Nginx routes:

- `/` -> serves `web_ui/realtime_multi.html`
- `/ws/en` -> `pipeline-english:8767`
- `/ws/hi` -> `pipeline-hindi:8768`

No other service ports are published on the host.

## Start

```bash
docker compose up -d --build
```

Open:

- `http://localhost:7777/realtime_multi.html`
- or `http://<public-ip>:7777/realtime_multi.html`

## HTTPS public URL (for microphone access)

Microphone capture in browsers needs `https://` (or `localhost`).

Start quick HTTPS tunnel:

```bash
docker compose up -d cloudflared
docker compose logs -f cloudflared
```

Use the `https://...trycloudflare.com` URL printed in logs.

## Stop

```bash
docker compose down
```

## Verify only one open Docker port

```bash
docker compose ps
docker ps --format "table {{.Names}}\t{{.Ports}}"
```

Expected result: only `konkani-reverse-proxy` shows `0.0.0.0:7777->7777/tcp`.

## Optional: start legacy services

Legacy Konkani and older STT services are still present in `docker-compose.yml` under profile `legacy` and are disabled by default.

```bash
docker compose --profile legacy up -d
```

Use this only if you intentionally need old routes/services.
