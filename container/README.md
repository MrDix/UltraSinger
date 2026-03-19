# Containerized UltraSinger

Run UltraSinger in a Docker or Podman container — no local Python install required.

## Quick start (Docker Compose)

```bash
# Build the image (once)
docker compose -f container/compose-cpu.yml build

# Convert a YouTube video (CPU)
docker compose -f container/compose-cpu.yml run --rm ultrasinger \
    uv run python /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/

# With NVIDIA GPU
docker compose -f container/compose-gpu.yml run --rm ultrasinger \
    uv run python /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/
```

## Detailed guides

- [Docker](docker.md) — Docker and Docker Compose setup
- [Podman](podman.md) — Podman setup (rootless, WSL2)

## Why run as a container?

- **Environment Consistency:** Same environment across different machines
- **Isolation:** No conflicts with other Python packages or system dependencies
- **Simplified Deployment:** Image contains all dependencies (FFmpeg, PyTorch, models)
- **Security:** Application is isolated from the host system
