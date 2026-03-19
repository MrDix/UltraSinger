# Containerized UltraSinger (Podman)

## Prerequisites

1. Podman installed (for Windows WSL2 machine provider is recommended)
   → [Installation instructions](https://podman-desktop.io/docs/installation)
2. (optional) For GPU acceleration: NVIDIA Container Toolkit installed
   → [GPU container access](https://podman-desktop.io/docs/podman/gpu)

## Setup

All commands are run from the **project root** directory.

### Build the image

```bash
podman build -t ultrasinger .
```

Or pull a pre-built image:

```bash
podman pull ghcr.io/rakuri255/ultrasinger:latest
```

## Usage

### One-off conversion (CLI)

**Bash (Linux / macOS / WSL2):**

```bash
# CPU
podman run --rm -it --name ultrasinger \
    -v ./input:/app/UltraSinger/input \
    -v ./output:/app/UltraSinger/output \
    ultrasinger \
    uv run python /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/

# GPU (requires NVIDIA Container Toolkit)
podman run --rm -it --name ultrasinger \
    --device nvidia.com/gpu=all \
    -v ./input:/app/UltraSinger/input \
    -v ./output:/app/UltraSinger/output \
    ultrasinger \
    uv run python /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/
```

**PowerShell (Windows):**

```powershell
# CPU
podman run --rm -it --name ultrasinger `
    -v ./input:/app/UltraSinger/input `
    -v ./output:/app/UltraSinger/output `
    ultrasinger `
    uv run python /app/UltraSinger/src/UltraSinger.py `
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/

# GPU (requires NVIDIA Container Toolkit)
podman run --rm -it --name ultrasinger `
    --device nvidia.com/gpu=all `
    -v ./input:/app/UltraSinger/input `
    -v ./output:/app/UltraSinger/output `
    ultrasinger `
    uv run python /app/UltraSinger/src/UltraSinger.py `
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/
```

### Long-running service

Start the container in the background for repeated conversions:

```bash
# Start detached
podman run -d --name ultrasinger \
    -v ./input:/app/UltraSinger/input \
    -v ./output:/app/UltraSinger/output \
    ultrasinger

# Run a conversion inside the running container
podman exec -it ultrasinger \
    uv run python /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/

# View logs
podman logs -f ultrasinger

# Stop and remove
podman stop ultrasinger && podman rm ultrasinger
```

### Convert a local file

Place the file in the `input/` folder, then:

```bash
podman run --rm -it --name ultrasinger \
    -v ./input:/app/UltraSinger/input \
    -v ./output:/app/UltraSinger/output \
    ultrasinger \
    uv run python /app/UltraSinger/src/UltraSinger.py \
    -i /app/UltraSinger/input/song.mp3 -o /app/UltraSinger/output/
```

## Volumes

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `input/` | `/app/UltraSinger/input` | Local audio/video files to convert |
| `output/` | `/app/UltraSinger/output` | Generated UltraStar txt + audio files |
| `cookies.txt` | `/app/UltraSinger/cookies.txt` | Video platform cookies (optional, read-only) |

## Video platform cookies

For age-restricted or authenticated video downloads, export your cookies with a browser
extension like [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
and save the file as `cookies.txt` in the project root.

```bash
podman run --rm -it --name ultrasinger \
    -v ./input:/app/UltraSinger/input \
    -v ./output:/app/UltraSinger/output \
    -v ./cookies.txt:/app/UltraSinger/cookies.txt:ro \
    ultrasinger \
    uv run python /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/ \
    --cookiefile /app/UltraSinger/cookies.txt
```

## Model cache

To avoid re-downloading models on every container start, mount a cache directory:

```bash
# Bash
-v $HOME/.cache:/app/UltraSinger/.cache

# PowerShell
-v $env:USERPROFILE\.cache:/app/UltraSinger/.cache
```
