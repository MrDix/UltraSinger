# Containerized UltraSinger

Run UltraSinger in a Docker or Podman container — no local Python install required.

## Quick start (Docker Compose)

```bash
# Create the input and output folders (once, in the project root)
mkdir input
mkdir output

# Build the image (once)
docker compose -f container/compose-cpu.yml build

# Convert a video URL (CPU)
docker compose -f container/compose-cpu.yml run --rm ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/

# With NVIDIA GPU
docker compose -f container/compose-gpu.yml run --rm ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/
```

## Detailed guides

- [Docker](docker.md) — Docker and Docker Compose setup
- [Podman](podman.md) — Podman setup (rootless, WSL2)

## What the image contains

- FFmpeg, Python 3.12, PyTorch with CUDA support and the packages of the conversion pipeline (CLI).
- Not included: the GUI, the optional game-score engine (`scoring` extra; scores fall back to the simple estimate) and the PO-token plugin and provider for full-quality video downloads (`potoken` extra), so video downloads may be limited to a reduced format set (typically 360p) or blocked. Converting local files from `input/` is not affected.
- The AI models are not part of the image: they are downloaded on first use and kept in the `model-cache` volume (Compose) or the cache folder you mount (see the guides).

## Why run as a container?

- **Environment Consistency:** Same environment across different machines
- **Isolation:** No conflicts with other Python packages or system dependencies
- **Simplified Deployment:** One build sets up FFmpeg, PyTorch and the Python packages
- **Security:** Application is isolated from the host system
