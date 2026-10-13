# Containerized UltraSinger (Docker)

## Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/) installed
- (optional) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU acceleration

## Setup

All commands are run from the **project root** directory.

### 1. Create the input and output folders

```bash
mkdir input
mkdir output
```

The compose files mount these folders and stop with an error if they are missing (instead of letting
Docker create them as root, which the container could not write to).

The container runs as user ID 1000. On Linux it must be able to write to `output/`: that is the case
if your own user ID is 1000 (check with `id -u`). Otherwise add `--user "$(id -u):$(id -g)"` after
`run` in the commands below.

### 2. Build the image

```bash
docker compose -f container/compose-cpu.yml build
```

> Both compose files share the same `Dockerfile`, so you only need to build once.

### 3. Choose CPU or GPU

| Setup | Compose file | Requires |
|-------|-------------|----------|
| CPU | `container/compose-cpu.yml` | Docker only |
| GPU | `container/compose-gpu.yml` | Docker + NVIDIA Container Toolkit |

## Usage

### One-off conversion (CLI)

Use `docker compose run --rm` to convert a single song and remove the container afterwards.

**Convert a video URL:**

```bash
# CPU
docker compose -f container/compose-cpu.yml run --rm ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/

# GPU
docker compose -f container/compose-gpu.yml run --rm ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/
```

**Convert a local file:**

Place the file in the `input/` folder, then:

```bash
docker compose -f container/compose-cpu.yml run --rm ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i /app/UltraSinger/input/song.mp3 -o /app/UltraSinger/output/
```

**Interactive shell** (for exploring or debugging):

```bash
docker compose -f container/compose-cpu.yml run --rm ultrasinger bash
```

### Long-running service

Use `docker compose up -d` to start the container in the background. This keeps
the container available for multiple conversions without rebuilding each time.

```bash
# Start in background
docker compose -f container/compose-gpu.yml up -d

# Run a conversion inside the running container
docker compose -f container/compose-gpu.yml exec ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/

# View logs
docker compose -f container/compose-gpu.yml logs -f

# Stop and remove
docker compose -f container/compose-gpu.yml down
```

## Volumes

Host paths are relative to the project root directory.

| Host | Container path | Purpose |
|-----------|---------------|---------|
| `input/` | `/app/UltraSinger/input` | Local audio/video files to convert (and `cookies.txt`, if you use one) |
| `output/` | `/app/UltraSinger/output` | Generated UltraStar txt + audio files |
| volume `model-cache` | `/home/ultrasinger/.cache` | Downloaded AI models, kept between runs |

The models are downloaded on the first conversion (several GB) and reused from the `model-cache`
volume afterwards. `docker volume rm container_model-cache` deletes them.

## Video platform cookies

For age-restricted or authenticated video downloads, export your cookies with a browser
extension like [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
and save the file as `cookies.txt` in the `input/` folder.

Then add `--cookiefile /app/UltraSinger/input/cookies.txt` to the command:

```bash
docker compose -f container/compose-cpu.yml run --rm ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/ \
    --cookiefile /app/UltraSinger/input/cookies.txt
```

The downloader writes refreshed cookies back to this file, so it must be writable (do not mount it
read-only).

## Building without Compose

If you prefer to use `docker build` and `docker run` directly (create `input/` and `output/` first;
`docker run -v` would otherwise create them as root):

```bash
# Build
docker build -t ultrasinger .

# Run (CPU)
docker run --rm -it \
    -v ./input:/app/UltraSinger/input \
    -v ./output:/app/UltraSinger/output \
    -v ultrasinger-models:/home/ultrasinger/.cache \
    ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i /app/UltraSinger/input/song.mp3 -o /app/UltraSinger/output/

# Run (GPU)
docker run --rm -it --gpus all \
    -v ./input:/app/UltraSinger/input \
    -v ./output:/app/UltraSinger/output \
    -v ultrasinger-models:/home/ultrasinger/.cache \
    ultrasinger \
    python3.12 /app/UltraSinger/src/UltraSinger.py \
    -i /app/UltraSinger/input/song.mp3 -o /app/UltraSinger/output/
```
