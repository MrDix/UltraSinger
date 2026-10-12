# Containerized UltraSinger (Podman)

## Prerequisites

1. Podman installed (for Windows WSL2 machine provider is recommended)
   → [Installation instructions](https://podman-desktop.io/docs/installation)
2. (optional) For GPU acceleration: NVIDIA Container Toolkit installed
   → [GPU container access](https://podman-desktop.io/docs/podman/gpu)

## Setup

All commands are run from the **project root** directory.

### Create the input and output folders

```bash
mkdir input
mkdir output
```

Create them before the first run: `podman run -v` would otherwise create them as root, and the
container could not write to them.

The container runs as user ID 1000. With rootless Podman on Linux, add `--userns=keep-id` to the
`podman run` commands below: it runs the container with your own user ID, so it can write to
`output/`.

### Build the image

```bash
podman build -t ultrasinger .
```

## Usage

### One-off conversion (CLI)

**Bash (Linux / macOS / WSL2):**

```bash
# CPU
podman run --rm -it --name ultrasinger \
    -v ./input:/app/UltraSinger/input \
    -v ./output:/app/UltraSinger/output \
    -v ultrasinger-models:/home/ultrasinger/.cache \
    ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/

# GPU (requires NVIDIA Container Toolkit)
podman run --rm -it --name ultrasinger \
    --device nvidia.com/gpu=all \
    -v ./input:/app/UltraSinger/input \
    -v ./output:/app/UltraSinger/output \
    -v ultrasinger-models:/home/ultrasinger/.cache \
    ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/
```

**PowerShell (Windows):**

```powershell
# CPU
podman run --rm -it --name ultrasinger `
    -v ./input:/app/UltraSinger/input `
    -v ./output:/app/UltraSinger/output `
    -v ultrasinger-models:/home/ultrasinger/.cache `
    ultrasinger `
    python3.12 /app/UltraSinger/src/UltraSinger.py `
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/

# GPU (requires NVIDIA Container Toolkit)
podman run --rm -it --name ultrasinger `
    --device nvidia.com/gpu=all `
    -v ./input:/app/UltraSinger/input `
    -v ./output:/app/UltraSinger/output `
    -v ultrasinger-models:/home/ultrasinger/.cache `
    ultrasinger `
    python3.12 /app/UltraSinger/src/UltraSinger.py `
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/
```

### Long-running service

Start the container in the background for repeated conversions:

```bash
# Start detached
podman run -d -it --name ultrasinger \
    -v ./input:/app/UltraSinger/input \
    -v ./output:/app/UltraSinger/output \
    -v ultrasinger-models:/home/ultrasinger/.cache \
    ultrasinger

# Run a conversion inside the running container
podman exec -it ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
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
    -v ultrasinger-models:/home/ultrasinger/.cache \
    ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i /app/UltraSinger/input/song.mp3 -o /app/UltraSinger/output/
```

## Volumes

| Host | Container path | Purpose |
|-----------|---------------|---------|
| `input/` | `/app/UltraSinger/input` | Local audio/video files to convert (and `cookies.txt`, if you use one) |
| `output/` | `/app/UltraSinger/output` | Generated UltraStar txt + audio files |
| volume `ultrasinger-models` | `/home/ultrasinger/.cache` | Downloaded AI models, kept between runs |

## Video platform cookies

For age-restricted or authenticated video downloads, export your cookies with a browser
extension like [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
and save the file as `cookies.txt` in the `input/` folder. Then add
`--cookiefile /app/UltraSinger/input/cookies.txt` to the command:

```bash
podman run --rm -it --name ultrasinger \
    -v ./input:/app/UltraSinger/input \
    -v ./output:/app/UltraSinger/output \
    -v ultrasinger-models:/home/ultrasinger/.cache \
    ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/ \
    --cookiefile /app/UltraSinger/input/cookies.txt
```

The downloader writes refreshed cookies back to this file, so it must be writable (do not mount it
read-only).

## Model cache

The `ultrasinger-models` volume in the commands above keeps the downloaded models between runs, so
they are not downloaded again for every container. `podman volume rm ultrasinger-models` deletes
them.
