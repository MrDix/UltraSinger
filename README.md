[![Discord](https://img.shields.io/discord/1048892118732656731?logo=discord)](https://discord.gg/rYz9wsxYYK)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MrDix/UltraSinger/blob/main/colab/UltraSinger.ipynb)
![Status](https://img.shields.io/badge/status-development-yellow)

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/MrDix/UltraSinger/main.yml)
[![GitHub](https://img.shields.io/github/license/MrDix/UltraSinger)](https://github.com/MrDix/UltraSinger/blob/main/LICENSE)
[![Check Requirements](https://github.com/MrDix/UltraSinger/actions/workflows/main.yml/badge.svg)](https://github.com/MrDix/UltraSinger/actions/workflows/main.yml)
[![Tests (fork)](https://github.com/MrDix/UltraSinger/actions/workflows/tests.yml/badge.svg)](https://github.com/MrDix/UltraSinger/actions/workflows/tests.yml)

<p align="center" dir="auto">
<img src="https://repository-images.githubusercontent.com/594208922/4befe3da-a448-4cbc-b6ef-93899119071b" style="height: 300px;width: auto;" alt="UltraSinger Logo">
</p>

# UltraSinger

> ⚠️ _This project is still under development!_

UltraSinger is a tool to automatically create UltraStar.txt, midi and notes from music.
It automatically pitches UltraStar files, adding text and tapping to UltraStar files and creates separate UltraStar karaoke files.
It also can re-pitch current UltraStar files and calculates the possible in-game score.

Multiple AI models are used to extract text from the voice and to determine the pitch.

Please mention UltraSinger in your UltraStar.txt file if you use it. It helps others find this tool, and it helps this tool get improved and maintained.
You should only use it on Creative Commons licensed songs.

## ❤️ Support
There are many ways to support this project. Starring ⭐️ the repo is just one 🙏

You can also support this work on <a href="https://github.com/sponsors/rakuri255">GitHub sponsors</a> or <a href="https://patreon.com/Rakuri">Patreon</a> or <a href="https://www.buymeacoffee.com/rakuri255" target="_blank">Buy Me a Coffee</a>.

This will help me a lot to keep this project alive and improve it.

<a href="https://www.buymeacoffee.com/rakuri255" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" style="height: 60px !important;width: 217px !important;" ></a>
<a href="https://patreon.com/Rakuri"><img src="https://raw.githubusercontent.com/rakuri255/UltraSinger/main/assets/patreon.png" alt="Become a Patron" style="height: 60px !important;width: 217px !important;"/> </a>
<a href="https://github.com/sponsors/rakuri255"><img src="https://raw.githubusercontent.com/rakuri255/UltraSinger/main/assets/mona-heart-featured.webp" alt="GitHub Sponsor" style="height: 60px !important;width: auto;"/> </a>

## Table of Contents

- [UltraSinger](#ultrasinger)
  - [❤️ Support](#️-support)
  - [Table of Contents](#table-of-contents)
  - [💻 How to use this source code](#-how-to-use-this-source-code)
    - [Installation](#installation)
    - [Run](#run)
  - [🌐 Corporate proxy / firewall](#-corporate-proxy--firewall)
  - [📖 How to use the App](#-how-to-use-the-app)
    - [🎶 Input](#-input)
      - [Audio (full automatic)](#audio-full-automatic)
        - [Local file](#local-file)
        - [Video URL](#video-url)
      - [UltraStar (re-pitch)](#ultrastar-re-pitch)
      - [Song folder](#song-folder)
    - [🗣 Transcriber](#-transcriber)
      - [Whisper](#whisper)
        - [Whisper languages](#whisper-languages)
      - [✍️ Hyphenation](#️-hyphenation)
    - [👂 Pitcher](#-pitcher)
    - [👄 Separation](#-separation)
    - [🥁 BPM Override](#-bpm-override)
    - [🎵 Octave Shift](#-octave-shift)
    - [Sheet Music](#sheet-music)
    - [Format Version](#format-version)
    - [Line Break Timing](#line-break-timing)
    - [Cache Folder](#cache-folder)
    - [🧪 Experimental Features](#-experimental-features)
      - [Library Triage Tool](#library-triage-tool)
      - [Chart Benchmark Tool](#chart-benchmark-tool)
      - [Model-Based Note Segmentation](#model-based-note-segmentation---segmentation_model)
      - [LLM Lyric Correction](#llm-lyric-correction---llm_correct)
      - [Remote Speech-to-Text](#remote-speech-to-text---remote_stt)
      - [Syllable-Level Note Splitting](#syllable-level-note-splitting---syllable_split)
      - [Vocal Gap Fill](#vocal-gap-fill---vocal_gap_fill)
      - [Golden Notes](#golden-notes---golden_notes)
    - [🏆 Ultrastar Score Calculation](#-ultrastar-score-calculation)
    - [📟 Use GPU](#-use-gpu)
      - [Considerations for Windows users](#considerations-for-windows-users)
      - [Crashes due to low VRAM](#crashes-due-to-low-vram)
    - [🖥️ GUI](#️-run-gui)
    - [📦 Containerized](#containerized-docker-or-podman)

## 💻 How to use this source code

### Installation

**Prerequisites**

* [git](https://git-scm.com/downloads) — to get and update the source code.
* [ffmpeg](https://www.ffmpeg.org/download.html) — must be on your `PATH`.
* [Node.js](https://nodejs.org) 22 or newer — *recommended:* needed for full-quality video downloads (the install script builds the local PO-token provider with it; without Node.js, downloads fall back to reduced quality — you can install it later and re-run the install script).
* Python 3.12 or 3.13 ([Download](https://www.python.org/downloads/)) — *optional:* if neither is installed (e.g. you only have a newer version), the install scripts automatically download a portable, self-contained Python 3.12 via uv (into uv's per-user directory, or into an app-local `.uv-python/` folder if the global store is unusable); your system Python is never touched.
* Linux and macOS: a C compiler, because some dependencies are built from source; on Linux also the headers of the system Python 3.12/3.13, if one is used. Debian/Ubuntu: `sudo apt install build-essential python3-dev`; Fedora: `sudo dnf install gcc python3-devel`; macOS: `xcode-select --install`. The installer checks for them before it starts.
* Behind a corporate proxy? See [Corporate proxy / firewall](#-corporate-proxy--firewall) — the installer detects `HTTP(S)_PROXY` automatically.

**Install**

1. Get the source code:
   ```bash
   git clone https://github.com/MrDix/UltraSinger.git
   cd UltraSinger
   ```
2. Run the installer — `install\auto_install.bat` (Windows) or `install/auto_install.sh` (Linux/macOS). It takes care of everything:
   * detects your NVIDIA GPU via `nvidia-smi` and picks the CUDA or CPU build automatically (force with `--cuda` / `--cpu`); a GPU the CUDA build cannot run (compute capability below 7.0 on Linux or below 6.1 on Windows, or a driver without CUDA 12 support) gets the CPU build,
   * installs all dependencies including the GUI, scoring engine and PO-token plugin,
   * installs the newest `yt-dlp` release — video platforms change their internals often, so the version pinned in `uv.lock` soon fails to download; it is only used when the upgrade is not possible (e.g. offline),
   * builds the local PO-token provider when Node.js is available,
   * detects proxy environment variables and enables OS-store TLS for uv automatically,
   * and prints tailored advice when your GPU has little VRAM (< 8 GB) or none was found — including how a free cloud key can replace the slowest local step (`--remote_stt`).

   An existing installation keeps working if the install fails (e.g. offline): the previous environment is only replaced once the new one is complete.

   You can also run a specific sub-script directly: `install/CUDA/…` (NVIDIA GPU, Windows/Linux) or `install/CPU/…` (no NVIDIA GPU, or macOS — Apple Silicon has no CUDA).

**Update an existing installation**

Run `install\update.bat` (Windows) or `install/update.sh` (Linux/macOS). **This is the only command you need to update** — you never have to re-run `auto_install`. It pulls the latest changes, upgrades `yt-dlp` to its newest release (keeping the pinned version if that is not possible, e.g. offline), syncs the Python packages into `.venv` (so newer dependency versions land too), refreshes the PO-token provider, and re-checks ffmpeg — and it transparently handles the CUDA case, where the installer protects `pyproject.toml`/`uv.lock` from git resets (a plain `git pull` would refuse to update those files with "Your local changes ... would be overwritten"). The CUDA variant of the PyTorch index you use (e.g. `cu126` instead of the default `cu128`) is kept; other local changes to `pyproject.toml` are replaced, and the update says where it kept a copy of your file. Local changes to `uv.lock` from the in-app **Update yt-dlp** button are replaced the same way, since the newest `yt-dlp` is locked again anyway. If the Python packages cannot be updated after the new code was pulled (e.g. no internet), the update says so: fix the cause and run it again. `auto_install` is only for the first-time install.

**Keeping everything on one drive (e.g. all development on `D:`)**

By default uv stores its package cache and, when it needs one, a downloaded Python interpreter under your user profile on `C:`. These are the two directories UltraSinger causes to grow (the package cache is several GB). If you keep your projects on another drive — for example because `C:` is wiped when the machine is replaced — you can move them there too. In a normal Command Prompt (these persist to your user environment; open a **new** terminal afterwards):

```bat
setx UV_CACHE_DIR "D:\dev\uv\cache"
setx UV_PYTHON_INSTALL_DIR "D:\dev\uv\python"
```

On Linux/macOS set the same variables in your shell profile. Nothing here is "your work" — it is cache and a runtime that uv rebuilds automatically — but relocating it keeps `C:` free of development data. A side benefit: when the cache and your project are on the same drive, uv can hardlink packages into the virtual environment instead of copying them (faster syncs, no "Failed to hardlink … falling back to full copy" warning).

### Run (CLI)

* In root folder just run `run_on_windows.bat`, `run_on_linux.sh` or `run_on_mac.command` to start the app.
* Or invoke it directly: `uv run python src/UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]`. See [How to use](#-how-to-use-the-app) for more information.

### 🖥️ Run (GUI)

UltraSinger includes an optional graphical interface with an embedded video browser,
full settings panel, and real-time conversion log.

#### GUI dependencies

The install scripts already set up everything the GUI needs ([PySide6](https://doc.qt.io/qtforpython-6/) / Qt 6 including WebEngine for the embedded browser) — no extra step required.

> Only if you installed manually and need to (re-)sync yourself, always include **all** extras — `uv sync` is exact and would *remove* extras you leave out (breaking scoring and full-quality downloads):
> ```bash
> uv sync --extra gui --extra scoring --extra potoken
> ```

#### Start the GUI

```bash
uv run python src/gui_main.py
```

Or use the platform-specific launcher scripts:

- **Windows:** `run_gui_on_windows.bat`
- **Linux:** `./run_gui_on_linux.sh`
- **macOS:** `./run_gui_on_mac.command`

#### Desktop shortcut (Windows)

The installer offers to create Desktop and Start Menu shortcuts for the GUI
(with the UltraSinger icon). You can also create them any time by running:

```bat
install\create_desktop_shortcut.bat
```

The shortcuts are taskbar-pinnable: right-click the shortcut — or the running
UltraSinger taskbar icon — and choose *Pin to taskbar*. The running window
groups onto the pinned icon.

#### Features

- **Video Browser** — Browse video platforms, log in to your account, and send videos directly to conversion. Cookies are captured automatically for authenticated downloads. A **quality badge** next to the Queue button shows the best available download resolution, audio codec, duration, and LRCLIB lyrics availability — so you can evaluate video quality even when the embedded browser cannot play certain codecs.
- **Settings Panel** — All CLI parameters available as form controls: Whisper model, language, post-processing options, experimental features, LLM lyric correction, and more.
- **Conversion Queue** — Real-time color-coded log output with stage detection (Separating Vocals → Transcribing → Pitching → …), elapsed timer, and cancel support. After conversion, each song shows an **info line** with:
  - **Language badge** — detected language code with color-coded background: green = synced lyrics (best quality), orange = plain lyrics or fallback, red = transcribed only
  - **Lyrics source** — "Synced lyrics", "Plain lyrics", or "Transcribed"
  - **Info button** — view the settings used for this conversion
  - **Folder button** — open the output folder in your system file manager (hover to see the path)
  - **Re-queue button** — re-queue the song with different settings (opens per-song settings dialog; useful for correcting wrong language detection)
- **Retry** — A failed or cancelled song stays in the list: **↻** lets it wait again, in its place and with its per-song settings, and **Start All** converts it (a running queue converts it too).
- **Warnings & errors only** — The switch below the log shows only its warnings and errors (the red and orange lines), also those printed before it was switched on, and each Python traceback whole, with the message of its exception. Switched off, the log shows all lines again, also those printed in the meantime.
  - A log that follows the newest lines goes on following them.
  - In a log scrolled up, the line at the top stays at the top (switched on: the first warning or error after it), so that a warning you scrolled to is shown among the lines around it when you switch off.
  - Switched back without scrolling, the log is where it was.
- **Queue kept between sessions** — The queue is saved while you work and comes back the next time you start the GUI, also after an update. It never starts by itself: press **Start All**.
  - Waiting songs, the song that was running and songs stopped with **Cancel All** come back as waiting songs. Failed songs come back marked as failed: **↻** lets one wait again, and **×** removes it from the list. Finished songs are left out.
  - The menu button (**☰**) next to **Clear** has **Export Queue…**, which saves the songs that are not finished to a file, and **Import Queue…**, which adds the songs of such a file as waiting songs. A song that already waits in the queue with the same settings is skipped.
  - Per-song settings are kept, API keys and tokens never. An imported file may come from someone else, so it does not set per-song settings that name a program, folder, web address or model, and takes the other settings only with values the per-song settings dialog offers; your own settings apply there.
  - The queue is stored in `~/.ultrasinger/queue.json`. A queue file is read up to its 1000th song. If the GUI cannot read the file, or has to leave songs or settings in it out, it first keeps a copy as `queue.json.<date>-<time>.bak`; a file it cannot read starts an empty queue. If even the copy fails, the GUI tells you and saves the queue only once the copy works: it tries again with every change and when it closes.
- **Preferences** — Default output folder, LLM/Groq API configuration, cookie management.
- **Model training** — **Train...** next to the Segmentation Model setting trains a [segmentation model](#model-based-note-segmentation---segmentation_model) on your own song library: pick the library, a work folder and the model file and follow the progress; the finished model is set for your conversions automatically.

> **Note:** The GUI is a native desktop application (Qt). It requires a display and cannot run inside a Docker container. Use the [CLI](#run-cli) for containerized workflows.

#### Full-quality video downloads (PO token)

The video platform now delivers its player streams via **SABR**, where the media data and the required *Proof-of-Origin* (PO) token are sent in binary POST bodies that an embedded browser cannot read. Without a PO token, `yt-dlp` is limited to a reduced format set (typically 360p) or blocked with HTTP 403.

To restore full-quality downloads UltraSinger uses the maintained [`bgutil-ytdlp-pot-provider`](https://github.com/Brainicism/bgutil-ytdlp-pot-provider) yt-dlp plugin (installed by the install scripts via the `potoken` extra). The plugin fetches PO tokens from a small local **provider server**.

- **Node.js (default, no Docker):** if [Node.js](https://nodejs.org) 22 or newer is installed, the install scripts build the provider automatically into `.potoken/` and the **GUI starts it on launch and stops it on exit** — nothing else to do. If Node.js was missing during install, install it and re-run the install script. The console tab reports the provider status (`[PO-Token] …`). The provider is always built from one fixed release (the one that matches the plugin version in `uv.lock`, checked by its commit, with the exact dependency versions of its lockfile); `install/update` moves an existing provider to it. The plugin only works with a provider of its own major version, so the GUI never starts an outdated build (it tries to rebuild it, else the console tab says what to do) and reports a provider of another version that is already running.
- **Docker (fallback):** if no Node.js provider is set up but Docker is available, the GUI auto-starts the container `brainicism/bgutil-ytdlp-pot-provider` on port 4416 instead. It uses the version of the installed plugin and is reachable only from this computer. You can also run it yourself. Replace `<version>` with the plugin's version (`uv pip show bgutil-ytdlp-pot-provider`): the plugin refuses a provider of another major version.
  ```bash
  docker run -d --rm -p 127.0.0.1:4416:4416 brainicism/bgutil-ytdlp-pot-provider:<version>
  ```

Once a provider responds on `http://127.0.0.1:4416`, `yt-dlp` uses it transparently for every download (GUI and CLI). If no provider is available the app still works, but video downloads fall back to the limited formats.

A download that breaks off is retried up to 10 times, with pauses of 1, 2, 4 … 30 s in between (about 3 minutes in all); the console shows each retry and pause. The platform's servers sometimes refuse one format of a video for hours while its player still plays the video. When a format (or one fragment of a fragmented format) still does not come after the retries, the download starts again with the next best format (another video or another audio format, up to 3 times) instead of leaving a gap in the video; it never falls back to the audio alone. When the format came over IPv6, which some networks (often Wi-Fi) carry worse than IPv4, the same format is first tried once more over IPv4, from where it stopped, and the next formats come over IPv4 as well. An error of this computer, such as a file that cannot be written, ends the download at once.

Provider behaviour can be tuned in the GUI config (`~/.ultrasinger`): `potoken_auto_start` (check on launch), `potoken_auto_start_node` / `potoken_auto_start_docker` (allow the respective auto-launch), `potoken_base_url` (custom server URL). Save `config.json` as UTF-8. If the GUI cannot read it, it starts with the default settings and keeps a copy of the file as `config.json.<date>-<time>.bak`; a single setting with a wrong value falls back to its default (see the log).

## 🌐 Corporate proxy / firewall

UltraSinger works behind a corporate proxy. There is nothing UltraSinger-specific to configure — it honors the same environment variables every standard tool does.

**CLI:** set `HTTP_PROXY` / `HTTPS_PROXY` (and `NO_PROXY` for exceptions, e.g. internal hosts) in the environment before running UltraSinger. Any casing works (`http_proxy` / `HTTP_PROXY`).

**GUI:** the Settings tab has a **Network Proxy** card with a `Mode` selector:
- **System / environment (default)** — uses your OS proxy settings or the `HTTP_PROXY`/`HTTPS_PROXY` environment variables, unchanged.
- **Manual** — enter a proxy URL (used for both `http://` and `https://` traffic) and optional no-proxy exceptions (comma-separated hosts/domains).
- **No proxy** — ignores any proxy configured elsewhere, even if the OS/shell has one set.

> ⚠️ **PAC files are not supported by the conversion pipeline.** Automatic proxy configuration (a `.pac` script) is only understood by the embedded browser tab (Chromium handles it natively) — yt-dlp, downloads, and LLM/remote-STT calls in the conversion pipeline cannot evaluate a PAC file. If your network uses one, switch the Network Proxy card to **Manual** and enter the proxy host directly.

**TLS-intercepting proxies** (which re-sign HTTPS traffic with an internal CA) are supported automatically: UltraSinger uses [`truststore`](https://pypi.org/project/truststore/) to read certificates from the operating system's trust store instead of Python's bundled `certifi` bundle — no extra configuration needed, as long as your IT department has installed the CA in Windows/macOS/Linux.

**Installer:** if `auto_install.sh`/`auto_install.bat` detects an `HTTP_PROXY`/`HTTPS_PROXY` environment variable and `UV_SYSTEM_CERTS` isn't already set, it automatically sets `UV_SYSTEM_CERTS=1` before running the CUDA/CPU sub-script, so `uv sync`/`uv lock` trust the OS certificate store too (opt out with `UV_SYSTEM_CERTS=0`). If the install still fails (e.g. a proxy configured only via the Windows registry, with no environment variables set), set `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` and `UV_SYSTEM_CERTS=1` yourself and re-run. `git`/`npm` (used by the optional PO-token provider setup) honor the same variables.

**Provider behind an HTTP proxy:** older provider releases had an axios bug (missing `proxy: false`) that made the provider send absolute-form GETs to the proxy instead of CONNECT tunnels — enterprise proxies answer those with 502 and token generation fails even though the same URLs work via `curl`. The provider release the install scripts build includes the fix; `install/update` moves an older provider to it.

**DNS/content filters (AdGuard Home, Pi-hole, filtering proxies):** the PO-token provider must reach **three** hosts to generate tokens: `jnn-pa.googleapis.com` (BotGuard API), `www.youtube.com` (challenge), and `www.google.com` (the BotGuard interpreter script under `/js/th/…` — confirmed blocked by some corporate proxies with a 502). The symptom of any of these being blocked is a *running* provider whose token requests fail with HTTP 500, and yt-dlp warning `Error fetching PO Token from "bgutil:http" provider`. Verify each with `curl -x $HTTPS_PROXY -sI https://<host>/ | grep HTTP` (any HTTP status, including 404, means reachable; 502 or no output means blocked — whitelist that host in your filter/proxy). The server's own log is written to `.potoken/provider.log`; note that this log only exists when the app itself started the server — a provider left running from an earlier session logs to wherever it was started.

**Local PO-token provider:** the loopback connection to the local PO-token provider (`http://127.0.0.1:4416`) is automatically excluded from the proxy, so a corporate proxy never breaks full-quality downloads.

## 📖 How to use the App

_Not all options working now!_
```commandline
    UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]

    [opt]
    -h      This help text.
    -i      Ultrastar.txt
            audio/video like .mp3, .mp4, .wav, video platform link
    -o      Output folder

    [mode]
    ## if INPUT is audio ##
    default (Full Automatic Mode) - Creates all, depending on command line options
    --interactive - Interactive Mode. All options are asked at runtime for easier configuration

    # Single file creation selection is in progress, you currently getting all!
    (-u      Create ultrastar txt file) # In Progress
    (-m      Create midi file) # In Progress
    (-s      Create sheet file) # In Progress

    ## if INPUT is ultrastar.txt ##
    default  Creates all

    [separation]
    --separator           Vocal separation backend: demucs|audio_separator >> ((default) is audio_separator)
                          audio_separator runs deterministic Roformer-based models (same result every
                          run), Mel-Band-Roformer by default.
    --audio_separator_model  Model for audio-separator. Preset names or filenames from
                          https://github.com/nomadkaraoke/python-audio-separator >> ((default) is model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt)
    --demucs              Demucs model name htdemucs|htdemucs_ft|htdemucs_6s|hdemucs_mmi|mdx|mdx_extra|mdx_q|mdx_extra_q >> ((default) is htdemucs)

    [transcription]
    --transcriber           auto|qwen3|whisper >> ((default) is auto)
                            auto: Qwen3-ASR 1.7B on a GPU with at least 5 GB free, for the 30 languages
                            it knows, else Whisper; qwen3: the same, with a note when Whisper has to
                            take over; whisper: always Whisper. Qwen3-ASR recognizes sung lyrics better
                            and is downloaded once (about 4.4 GB).
    --whisper               Multilingual model > tiny|base|small|medium|large-v1|large-v2|large-v3  >> ((default) is large-v2)
                            English-only model > tiny.en|base.en|small.en|medium.en
                            Choosing a Whisper model chooses Whisper, unless --transcriber is given too.
    --whisper_align_model   Custom wav2vec2 forced-alignment model from HuggingFace
                            (e.g. --whisper_align_model gigant/romanian-wav2vec2)
    --language              Language of the singing (e.g. en, de) for the transcription, the alignment
                            and the hyphenation. Without it the language is detected: by the
                            transcription when it runs (this replaces any earlier guess), else quickly
                            via Whisper tiny (~2-3s). Only when found synced lyrics let the
                            transcription be skipped (without --remote_stt), the language metadata of
                            a video URL is used instead.
                            WARNING: setting this for non-matching songs will degrade
                            alignment quality (e.g. --language en for German songs).
    --whisper_batch_size    Segments processed in parallel. 'auto' (default) scales to GPU VRAM: 16 (~8GB and larger) / 8 (5-7GB) / 4 (smaller). Lower manually if still low on GPU mem: slower, but transcription is UNCHANGED (the safe lever)
    --whisper_compute_type  Change to "int8" to save more GPU mem at a small accuracy cost; use only if lowering the batch size is not enough >> ((default) is "float16" for cuda devices, "int8" for cpu)
    --keep_numbers          Numbers will be transcribed as numerics instead of as words
    --vad_onset             VAD (Voice Activity Detection) speech activation threshold (0.0-1.0). Lower
                            values capture more vocal segments including soft/breathy singing.
                            >> ((default) is 0.35, WhisperX default: 0.5)
    --vad_offset            VAD speech deactivation threshold (0.0-1.0). Lower values keep segments active
                            longer during vocal dips. >> ((default) is 0.20, WhisperX default: 0.363)
    --no_speech_threshold   No-speech probability threshold (0.0-1.0). Lower values prevent Whisper from
                            classifying singing as silence. >> ((default) is 0.4, WhisperX default: 0.6)
    --ignore_audio          Re-pitch mode: keep the lyrics and timing of an ultrastar.txt input (-i) and
                            only recompute the pitch. Set automatically for a .txt input; with an audio
                            or video input it stops with an error.

    [pitch detection]
    --pitcher               Pitch detection backend: swiftf0|fcpe >> ((default) is swiftf0)
                            swiftf0: ONNX-based, CPU-only, fast and lightweight.
                            fcpe: GPU-accelerated (torchfcpe), more stable pitch contours with fewer
                            outlier jumps. Better for difficult vocals (metal, screamo). Best
                            performance on CUDA, falls back to CPU if unavailable.

    [post-processing]
    --bpm                   Override auto-detected BPM with a manual value (e.g., --bpm 120)
    --octave                Shift all notes by N octaves after pitch detection (e.g., --octave 1 for up, --octave -1 for down)
    --disable_hyphenation   Disable word hyphenation. Hyphenation is enabled by default.
    --disable_separation    Disable track separation. Track separation is enabled by default.
    --disable_onset_correction  Disable onset-based timing correction. Enabled by default.
    --disable_vocal_center  Disable vocal-centre octave correction. Enabled by default.
    --disable_quantization  Disable key quantization. Enabled by default, removes pitch slides and out-of-key notes.
    --syllable_split        Keep syllable-level note splits at pitch changes (experimental). Disabled by default.
    --vocal_gap_fill        Fill un-transcribed vocal gaps with placeholder notes (experimental). Disabled by default.
    --pitch_change_split    Split notes at pitch change boundaries within a syllable. Detects sustained pitch
                            changes (melismas, runs) and creates separate notes for each pitch region. Uses
                            vibrato-aware detection (region median comparison) to avoid false splits. Enabled
                            by default.
    --no_pitch_change_split Disable pitch-change splitting (revert to one note per word).
    --pitch_notes           Generate notes from pitch contour instead of word timing (experimental). Best for
                            melismatic songs with runs and slides. Notes are segmented by pitch stability, then
                            split at word boundaries so each word gets its own note. When lyrics lookup is active,
                            reference lyrics fill remaining placeholder notes. Disabled by default.
    --disable_lyrics_lookup Disable LRCLIB lyrics lookup. Enabled by default, fetches verified reference
                            lyrics to correct Whisper transcription errors. No API key needed.
    --disable_reference_lyrics  Disable the reference-lyrics-first pipeline. When LRCLIB provides synced
                            (timestamped) lyrics, they are used with wav2vec2 forced alignment to obtain
                            precise word-level timing — dramatically improving lyrics coverage and timing
                            accuracy. Without synced lyrics (or when their alignment fails), plain lyrics
                            are aligned the same way, with line times from the transcription's words, then
                            a --remote_stt text. When disabled, or when none of them can be aligned, the
                            notes come from the transcription.
                            Enabled by default.

    [freestyle detection]
    --detect_freestyle              Detect vocal passages that cannot be reliably pitched and mark them as freestyle
                                    notes (displayed but not scored). Covers growls, screams, rap, spoken word,
                                    harsh vocals, and any non-melodic vocal style.
                                    Primary: HPSS (Harmonic-Percussive Source Separation) harmonicity
                                    analysis (genre/gender-independent, measures harmonic vs. percussive
                                    energy). Fallback (when HPSS is unavailable): SwiftF0 confidence +
                                    pitch stability.
    --freestyle_harmonicity         HPSS harmonic ratio threshold — segments below this are unpitchable >> ((default) is 0.40)
    --freestyle_energy              RMS energy threshold — segments below this are treated as silence >> ((default) is 0.01)
    --freestyle_confidence          SwiftF0 median confidence threshold (fallback) >> ((default) is 0.35)
    --freestyle_pitch_stdev         Pitch standard deviation threshold in semitones (fallback) >> ((default) is 4.0)
    --freestyle_spectral_flatness   Spectral flatness threshold (fallback) >> ((default) is 0.25)
    --no_freestyle_spectral         Disable spectral flatness analysis (fallback)

    [refinement]
    --disable_refine            Disable the reverse-scoring refinement pass. Refinement is enabled by default
                                and uses the game's C++ ptAKF (the pitch-detection algorithm the karaoke
                                games themselves use to score singing) to find and fix poorly-scoring notes.
    --refine_from_vocal         (legacy) Explicitly enable refinement (now the default)
    --disable_refine_pitch      Disable pitch refinement (enabled by default when refine is on; a note keeps
                                its octave, only its pitch class is corrected; notes from a segmentation
                                model keep a well-tracked pitch unless a second pitch track confirms the
                                correction)
    --disable_refine_timing     Disable timing refinement (enabled by default when refine is on; always skipped
                                for notes from a segmentation model)
    --disable_refine_gap        Disable the GAP sweep, which shifts all notes together by up to 70 ms to where
                                the game scores them best (enabled by default when refine is on; always
                                skipped for notes from a segmentation model)
    --refine_hit_ratio          Notes below this hit ratio are pitch-corrected (0.0-1.0) >> ((default) is 0.4)
    --refine_timing_threshold   Milliseconds threshold before correcting timing >> ((default) is 30)
    --ptakf_refit               (legacy) Explicitly enable the ptAKF chart refit (now the default)
    --disable_ptakf_refit       Disable the ptAKF chart refit (rebuilds note boundaries and pitches from
                                the game's own pitch detector; enabled by default)
    --ptakf_refit_min_note_ms   Merge refit notes shorter than this when score-neutral >> ((default) is 100)
    --ptakf_refit_fill          (legacy) Explicitly enable refit fill (now the default)
    --disable_ptakf_refit_fill  Disable charting sung regions outside all notes (ad-libs, vocalises,
                                melisma tails; enabled by default when the refit is on)
    --ptakf_refit_fill_min_ms   Minimum uncharted voiced run length before it is filled >> ((default) is 300)
    --disable_score             Skip the score calculation (internal + game score) at the end of the
                                conversion. Scoring is enabled by default

    [golden notes]
    --golden_notes              Mark 10 held notes per song as golden "*" bonus notes, worth double score
                                in-game, like hand-made charts do (experimental). Disabled by default,
                                since it changes the score distribution. Long notes the singing stays on
                                pitch with are preferred (a golden note nobody can hit is worse than none),
                                with a bonus for the highest note of a phrase. Only normal notes held for
                                at least 200ms are eligible; freestyle and rap notes are never marked. At
                                most 15% of all scorable notes become golden (short songs), and at most 3
                                in each tenth of the song, so they do not pile up in one section.


    [llm lyric correction]
    --llm_correct           Enable LLM-based lyric post-correction (disabled by default)
    --llm_api_base_url      LLM API base URL >> ((default) is https://api.openai.com/v1)
    --llm_api_key           LLM API key (or set LLM_API_KEY env var)
    --llm_model             LLM model name >> ((default) is gpt-4o-mini)
    --llm_no_retry          Disable automatic retry on rate limit (HTTP 429). Retry is enabled by default.
    --llm_retry_wait        Seconds to wait between retries >> ((default) is 60)
    --llm_retry_max         Maximum retries per text chunk >> ((default) is 3)

    [remote speech-to-text]
    --remote_stt                 Enable remote (cloud) speech-to-text as a Whisper alternative
                                  for GPU-less machines (disabled by default). Its rough word
                                  times only place the lines -- the notes are timed locally.
    --remote_stt_api_base_url    OpenAI-compatible API base URL >> ((default) is
                                  https://api.groq.com/openai/v1)
    --remote_stt_api_key         API key (or set ULTRASINGER_REMOTE_STT_API_KEY env var)
    --remote_stt_model           Remote STT model name >> ((default) is whisper-large-v3)
    --remote_stt_timeout         Seconds to wait for the remote STT response before falling back to
                                  local Whisper >> ((default) is 120)
    --remote_stt_no_retry        Disable automatic retry on rate limit (HTTP 429). Retry is enabled by default.
    --remote_stt_retry_wait      Seconds to wait between retries >> ((default) is 60)
    --remote_stt_retry_max       Maximum retries >> ((default) is 3)

    [output]
    --format_version        0.3.0|1.0.0|1.1.0|1.2.0 >> ((default) is 1.2.0)
    --linebreak_timing      yass|usdx|after_last_note: when the view changes to the next line >> ((default) is yass)
                            yass: in a pause of 2 s or more the sung line stays 1-2 s after its last note,
                            in shorter pauses the next line comes 2-4 beats before it starts (10 beats after
                            the line in pauses of more than 16 beats);
                            usdx: 2 beats after the line (1 beat in a pause of 2 beats, at the next line's
                            start in shorter pauses); after_last_note: right after the line's last note
    --disable_karaoke       Disable creation of karaoke txt file. Karaoke is enabled by default.
    --create_audio_chunks   Enable creation of audio chunks. Disabled by default.
    --disable_midi          Disable MIDI file creation. MIDI is created by default.
    --no_metadata_tags      Skip writing ID3/Vorbis tags (title/artist/year/genre/cover) to output
                            audio. Tags are written by default.
    --keep_audio_in_video   Keep full audio (vocals+instrumental) in the output video. Disabled by default.
    --write_settings_info   Write ultrasinger_parameter.info with all settings and scores to output dir.
    --plot                  Enable creation of plots. Disabled by default.
    --keep_cache            Keep the cache folder (intermediate files) in the song folder. Removed by default.
    --cache_folder          Folder for the cache folders of the conversions, never the song folder
                            >> ((default) is %LOCALAPPDATA%\UltraSinger\cache, on Linux/macOS ~/.cache/ultrasinger/cache)

    [denoise]
    --denoise_nr            Noise reduction in dB (0.01-97). Lower preserves more vocal detail. >> ((default) is 20)
    --denoise_nf            Noise floor in dB (-80 to -20) >> ((default) is -80)
    --disable_denoise_track_noise  Disable adaptive noise floor tracking >> ((default) tracking is enabled)

    [yt-dlp / metadata]
    --cookiefile            File name where cookies should be read from and dumped to.
    --video_url             Video URL for metadata lookup when -i is a local file.
                            When -i is a video URL, UltraSinger extracts artist/title
                            directly from the video platform. When -i is a local file,
                            UltraSinger parses the filename ("Artist - Title.ext") and
                            searches MusicBrainz for metadata and cover art. This works
                            well when the filename is descriptive but fails for generic
                            names like "video.avi".
                            --video_url provides a fallback: the audio is loaded from the
                            local file (-i), but artist/title are fetched from the video
                            URL instead of relying on the filename (the cover comes from
                            the lookup by them).
                            The GUI uses this automatically when downloading via the
                            embedded browser (pre-downloaded audio + video platform metadata).
                            Not needed when -i is already a video URL.
                            (--youtube_url is accepted as a deprecated alias.)
    --yt_po_token           GVS Proof-of-Origin token for yt-dlp (web.gvs). Without it the
                            video platform limits downloads to a reduced format set or blocks
                            them (HTTP 403). The GUI captures this token automatically from its
                            embedded browser while a video plays and passes it through together
                            with the cookies, restoring full-quality downloads. For manual CLI
                            use see yt-dlp's PO token guide
                            (https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide).
                            Can also be given as ULTRASINGER_YT_PO_TOKEN environment
                            variable, which keeps it out of the process list.

    [device]
    --force_cpu             Force all steps to be processed on CPU.
    --force_whisper_cpu     Only the transcription will be forced to cpu (so Whisper transcribes:
                            Qwen3-ASR needs the GPU)

    [paths]
    --musescore_path        Path to MuseScore executable
    --ffmpeg                Path to ffmpeg and ffprobe executable
```

For standard use, you only need to use [opt]. All other options are optional.

### 🎶 Input

### Mode
default (Full Automatic Mode) - Creates all, depending on command line options
--interactive - Interactive Mode. All options are asked at runtime for easier configuration
```commandline
--interactive
```
#### Audio / Video (full automatic)

##### Local file

```commandline
-i "input/music.mp3"
```

##### Video URL

```commandline
-i https://www.youtube.com/watch?v=YwNs1Z0qRY0
```

Note that if you run into a yt-dlp error such as `Sign in to confirm you’re not a bot. This helps protect our community` ([yt-dlp issue](https://github.com/yt-dlp/yt-dlp/issues/10128)) you can follow these steps:

* generate a cookies.txt file with [yt-dlp](https://github.com/yt-dlp/yt-dlp/wiki/Installation) `yt-dlp --cookies cookies.txt --cookies-from-browser firefox`
* then pass the cookies.txt to UltraSinger `--cookiefile cookies.txt`

#### UltraStar (re-pitch)

This re-pitches the notes of an UltraStar txt file from its audio and writes the chart with the new pitches into a new song folder, as `Artist - Title.txt`. Normal and golden notes get new pitches; freestyle and rap notes keep theirs. Like a converted song, the new chart gets a [game score](#-ultrastar-score-calculation). The cover, background and video files the chart names are copied along; when the cover is missing, a cover found online is saved and written into the new chart.

Older hand-made charts are read as well: in CP1252 or in the encoding their `#ENCODING` tag names, and with relative beats (`#RELATIVE:YES`). The new file is always written in UTF-8 and with absolute beats.

```commandline
-i "input/ultrastar.txt"
```

#### Song folder

Each song gets a folder of its own in the output folder, named `Artist - Title` (`Artist - Title (1)` ... if that folder exists), and its files are named alike. When the download of a song from the video platform fails, its new folder is removed again, so that the next try does not leave `(1)`, `(2)` ... folders of partial downloads behind; a folder that holds other files stays. A name longer than 60 characters is shortened and ends with a short code made from the full name, so the paths of the song's files and of its cache folder stay within the Windows path length limit.

### 🗣 Transcriber

By default (`--transcriber auto`) the lyrics are transcribed by **Qwen3-ASR** when a GPU with enough free memory is available and the song is in one of its languages, otherwise by **Whisper**. `--transcriber qwen3` asks for Qwen3-ASR (with a note when Whisper has to take over), `--transcriber whisper` or a Whisper model (`--whisper large-v2`) for Whisper.

A word of the transcription that holds nothing to sing is no note, as in lyrics (see *Reading the lines* under [Reference-Lyrics-First Pipeline](#-reference-lyrics-first-pipeline)): Whisper writes a music note (`♪`) for an instrumental part, which is left out, as are dashes; closing punctuation that stands alone, like the `?` after a space in French, joins the word before it. This also holds for a transcription reused from the cache.

#### Qwen3-ASR

Qwen3-ASR 1.7B is an audio language model that is trained on singing as well. On 100 separated vocal tracks with hand-made charts it made 26.9 % word errors with the song's language given (27.7 % when it found the language itself), against 30.0 % for Whisper large-v2, mostly because it invents fewer words.

- It needs a GPU with at least 5 GB of free memory; with less, without a GPU, or with `--force_whisper_cpu`, Whisper transcribes. The variants that would fit less memory were less accurate than Whisper on the same tracks (the 0.6B model 31.9 %, the 1.7B model quantized to 4 bit 33.4 %), so they are not used.
- The vocal track is transcribed in pieces of about half a minute, cut at quiet points (given a whole song at once, the model leaves out repeated parts). The language comes from `--language`, else from a quick detection when that is sure, else the model finds it; the word timings come from the same forced alignment as Whisper's text.
- It knows 30 languages, among them English, German, French, Spanish, Italian, Portuguese, Dutch, Swedish, Danish, Finnish, Polish, Czech, Russian, Chinese, Japanese and Korean; for a song in another language (for example Norwegian or Ukrainian), Whisper transcribes.
- When Qwen3-ASR fails (for example when the first download of the model fails), Whisper transcribes, with a note.
- The model (about 4.4 GB) is downloaded on the first use.

```commandline
-i XYZ --transcriber qwen3
```

#### Whisper

Keep in mind that while a larger model is more accurate, it also takes longer to transcribe.

For the first test run, use the `tiny`, to be accurate use the `large-v2` model.

```commandline
-i XYZ --whisper large-v2
```

##### Whisper languages

Currently provided default language models are `en, fr, de, es, it, ja, zh, nl, uk, pt`.
If the language is not in this list, you need to find a phoneme-based ASR model from
[🤗 huggingface model hub](https://huggingface.co). It will download automatically.

Example for romanian:
```commandline
-i XYZ --whisper_align_model "gigant/romanian-wav2vec2"
```

Without `--language`, Whisper finds the language of the song before it transcribes, in the first 30 s of the vocal track. An instrumental intro is near silence there, and for that Whisper may name a rare language (Norwegian Nynorsk for three songs in English, on a benchmark of 100 vocal tracks). So when the language found is not one of the languages listed above, it is detected again in the first 30 s of singing, and the song is transcribed in the language found there; on the benchmark, this gave English for the three songs and changed no other song.

#### ✍️ Hyphenation

Is on by default. Can also be deactivated if hyphenation does not produce
anything useful. Note that the word is simply split,
without paying attention to whether the separated word really
starts at the place or is heard. To disable:

```commandline
-i XYZ --disable_hyphenation
```

### 👂 Pitcher

UltraSinger supports two pitch detection backends:

- **SwiftF0** (default): ONNX-based, CPU-only, fast and lightweight.
  Detects pitch frequencies between 46.875 Hz (G1) and 2093.75 Hz (C7).
  UltraSinger uses 60 Hz and 400 Hz as detection range.

- **FCPE** (`--pitcher fcpe`): GPU-accelerated via [torchfcpe](https://github.com/CNChTu/FCPE).
  Produces more stable pitch contours with fewer outlier jumps (40-50% fewer pitch jumps >5 ST in benchmarks).
  Better for difficult vocals (metal, screamo, harsh vocals).
  Best performance on CUDA, falls back to CPU if unavailable.

```commandline
-i XYZ --pitcher fcpe
```

### 👄 Separation

The vocals are separated from the audio before they are passed to the models. If problems occur with this,
you have the option to disable this function; in which case the original audio file is used instead.

```commandline
-i XYZ --disable_separation
```

### 🎵 Key Quantization

By default, UltraSinger quantizes all detected notes to the detected musical key of the song. This helps to:
* Remove pitch slides and vocal transitions between notes
* Correct out-of-key notes from pitch detection errors

If you want to keep the raw pitch detection without quantization:

```commandline
-i XYZ --disable_quantization
```

For most songs, quantization produces better results.

### 🥁 BPM Override

UltraSinger automatically detects the BPM of the song. If the auto-detection gives an incorrect result, you can manually override it with the `--bpm` option. The value must be a positive number.

```commandline
-i XYZ --bpm 120
```

This skips the automatic BPM detection and uses the provided value instead. Decimal values are also supported. You can find the correct BPM for most songs on sites like [songbpm.com](https://songbpm.com/) or in the song's metadata.

### 🎵 Octave Shift

UltraSinger includes automatic octave correction, but in rare cases the pitch detector may consistently detect the wrong octave for an entire song. If all notes sound correct in relative pitch but are shifted by one or more octaves, use the `--octave` option to manually correct them.

```commandline
-i XYZ --octave 1
```

The value is an integer: positive values shift up, negative values shift down. For example, `--octave 1` shifts all notes up by one octave, `--octave -1` shifts down. The shift is applied after automatic octave correction, so it acts as a final override.

#### Isolated octave-spike snap (`--octave_snap`, GUI: Settings → Post-Processing → "Octave Spike Snap")

Separately from a whole-song shift, the pitch tracker occasionally lifts or drops a *single* note by an octave while its neighbours stay put — a lone note that jumps up and back down, jarring to sing and read. `--octave_snap` folds only those clear, isolated spikes back onto the melody: it acts on a note that sticks out above or below both of its immediate neighbours, sits in a stable local context, and is about an octave away. Genuine leaps, gradual movement and legitimately wide-range songs are left untouched, and because octave is scoring-irrelevant (the game folds octaves) it never changes the game score. It is a display/singability polish and does **not** fix a whole passage that is consistently an octave off (that is a separate, harder pitch-tracking problem). Disabled by default.

#### Octave consistency (`--octave_consistency` / `--disable_octave_consistency`, GUI: Settings → Post-Processing → "Octave Consistency")

The stronger octave repair: pitch trackers scatter individual notes *and short runs* into the wrong octave — measured against professional reference charts, generated charts contained about **10× as many jarring octave-size jumps** between adjacent notes (507 vs 47 across 8 songs), which makes passages extremely confusing to sing. `--octave_consistency` keeps every note's pitch class and re-chooses only its octave via dynamic programming over the whole song, balancing melodic smoothness against fidelity to what the tracker detected. The balance is self-limiting: short wrong-octave scatter (roughly 1–3 notes) is folded onto the melody line, while a genuine octave passage of about five notes or more is cheaper to keep — so real octave jumps and wide-range songs survive (validated on the same 8 reference songs: jumps dropped to professional-chart level while octave-sensitive pitch agreement with the references stayed within 1 percentage point). The game score is unaffected because scoring folds octaves. By default it is applied to the notes of a [segmentation model](#model-based-note-segmentation---segmentation_model) only: there the octave outlier pass moves whole phrases (notes joined by small steps) instead of every single note that is an octave or more away from the middle of the song — which pulled a correctly sung run into the low or high end of a wide-range song up or down an octave halfway — and only a phrase that an octave-size jump leads into or out of, so a part sung an octave higher or lower after a rest keeps its octave; this pass then removes the scattered wrong-octave notes. Measured on 99 benchmark songs: jumps of 10+ semitones that the reference chart does not have went down from 1.16 to 0.36 per 100 notes, while the share of notes in the same octave as the reference stayed within 0.2 points. `--octave_consistency` (GUI: "All notes") applies the pass to word-based notes as well, and `--disable_octave_consistency` (GUI: "Off") to no notes at all — it can flatten a sung leap of a note or two.

### Sheet Music

For Sheet Music generation you need to have `MuseScore` installed on your system.
Or provide the path to the `MuseScore` executable.

```commandline
-i XYZ --musescore_path "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
```

### Format Version

This defines the format version of the UltraStar.txt file. For more info see [Official UltraStar format specification](https://usdx.eu/format/).

You can choose between different format versions. The default is `1.2.0`.
* `0.3.0` is the first format version. Use this if you have an old UltraStar program and problems with the newer format.
* `1.0.0` should be supported by the most UltraStar programs. Use this if you have problems with the newest format version
* `1.1.0` is the current format version.
* `1.2.0` is the upcoming format version. It is not finished yet.
* `2.0.0` is the next format version. It is not finished yet.

```commandline
-i XYZ --format_version 1.2.0
```

### Line Break Timing

`--linebreak_timing` (GUI: **Line Break Timing**) sets when the view changes from a line to the next one. Which words make up a line does not change.

- `yass` (default), as the YASS editor does it: in a pause of 2 s or more, the sung line stays 1 s after its last note (2 s from a pause of 4 s on), so you can still see how well you hit its notes. In shorter pauses the next line appears 2 to 4 beats before it starts, or 10 beats after the line when the pause is longer than 16 beats.
- `usdx`, as the UltraStar Deluxe editor does it: 2 beats after the line (1 beat in a pause of 2 beats, at the next line's start in shorter pauses).
- `after_last_note`: right after the line's last note, so the next line is shown as early as possible.

```commandline
-i XYZ --linebreak_timing after_last_note
```

### Cache Folder

While a song is converted, its intermediate files (separated vocals and instrumental, denoised and mono audio, transcription and pitch data; several hundred MB) are kept in a cache folder of their own, outside the song folder. A song folder in a folder that a file-sync client watches thus does not get these files uploaded while the conversion still uses them.

- The cache folders are made in `%LOCALAPPDATA%\UltraSinger\cache` on Windows and in `~/.cache/ultrasinger/cache` on Linux and macOS. The log names the folder of each conversion.
- When the conversion is done, its cache folder is deleted. With `--keep_cache` (GUI: **Keep Cache**) it is moved into the song folder as `cache` instead (`cache (1)` ... if the song folder has one already).
- `--cache_folder <folder>` (GUI: **Cache Folder**) makes the cache folders somewhere else, for example on a drive with more free space. Choose a local folder that no file-sync client watches.
- A conversion that is cancelled or crashes leaves its cache folder behind. Such folders are removed by a later conversion once nothing in them has changed for a day. Only folders that UltraSinger made are removed: their name ends with `.ultrasinger` and they contain a `.ultrasinger-cache` marker file (a kept cache has none).

```commandline
-i XYZ --cache_folder D:\UltraSinger\cache
```

### Settings Info File

When enabled, UltraSinger writes a `ultrasinger_parameter.info` file to the output directory containing all conversion settings and a detailed pipeline trace. The `[Pipeline]` section shows exactly how the song was processed:

- Which pipeline was used (Reference-Lyrics-First vs standard transcription)
- How the language was determined (manual / video platform metadata / Whisper fast-detect / full transcription)
- Whether the transcriber received an explicit language hint or used auto-detect
- Language correction history (if fast-detect was wrong and the full transcription corrected it)
- Whether the reference pipeline was recovered after language correction
- LRCLIB lyrics availability (synced / plain / none)
- LLM correction results and score breakdown
- Whether a segmentation model was set and actually applied (`--segmentation_model`)

```commandline
-i XYZ --write_settings_info
```

### 🎤 Reference-Lyrics-First Pipeline

When LRCLIB provides **synced (timestamped) lyrics** for a song, UltraSinger uses them with wav2vec2 forced alignment instead of relying on Whisper transcription. This dramatically improves lyrics accuracy and timing:

- **100% lyrics coverage** — every word from the verified reference lyrics is included
- **Precise word-level timing** — wav2vec2 CTC alignment snaps words to the actual audio
- **~2 minutes faster** — the expensive Whisper transcription step is skipped entirely
- **Automatic language detection** — language is resolved in this order:
  1. `--language` CLI flag (highest priority, manual override)
  2. Video platform metadata (yt-dlp extracts the video's language automatically; not used with `--remote_stt`)
  3. Whisper tiny fast-detect (~2-3 seconds)
  4. Full transcription, when the pipeline falls back to it: its detection then replaces 2. and 3.

The pipeline is **enabled by default** and falls back to the standard transcription (Qwen3-ASR or Whisper, see [Transcriber](#-transcriber)) when LRCLIB has no lyrics for the song or they cannot be aligned. Use `--disable_reference_lyrics` to force the transcription path.

**Finding the lyrics:** UltraSinger searches LRCLIB by artist and title. Titles taken from a video often carry additions like `(Official Video)`, `- US Version`, `- Live Performance`, `[HD]` or `feat. …`; when nothing is found with them, the search is repeated without them. Other parts of a title, like `(Part 2)`, are kept, since they can tell two songs apart. When still nothing is found and the title has a description of the video after a dash (`Title - Singer Name live in the studio`: at least three words, without a part or volume number and without another dash, which would rather make a medley), the search is repeated with the title before the dash; the song database is searched the same way. Last, LRCLIB is searched with the title before the dash whatever follows it (`Title - Chapter 2`), but then only a version at most 5 s off the length of the audio is taken: `Title - Part 2` can be another song than `Title`, of another length. A found song is only used when its artist and its title match (also in the other order, for video titles that name the song first). When LRCLIB has several versions of the song (album, single, live), the one closest in length to the audio is taken, since versions of a different length often differ in their lyrics too. Of the versions about as close as that one (up to 5 s further off), one with synced lyrics is preferred. The length never rules a version out: when none has the length of the audio, the closest one is still used.

**Lyrics in another language are ignored:** LRCLIB sometimes returns another language version of a song (e.g. the English lyrics of a song sung in German). UltraSinger identifies the language of the found lyrics text and ignores them when it clearly differs from a reliably known sung language — set with `--language`, detected by the full transcription, or detected by the fast detection with at least 80 % confidence. Video platform metadata alone does not count, since it reflects upload settings and is often a default. When the language comes from the metadata or from a less certain detection, the lyrics are kept (and used for the alignment) and a note suggests setting `--language`.

**Auto-recovery from wrong language detection:** If the fast language detection (Whisper tiny, ~2-3s) misidentifies the language (e.g. "cy" instead of "en"), the reference pipeline may fail because no alignment model exists for that language. In this case, UltraSinger automatically runs the full transcription, which detects the correct language, then **retries the reference pipeline** with the corrected language — recovering to full synced-lyrics quality without user intervention.

**Reading the lines:** A line with several times (`[00:12.00][01:30.00] ...`, a repeated chorus) is sung at each of them. Word times of enhanced LRC (`<00:12.50>`), ID tags like `[ar:...]`, section labels at the start of a line, like `[Chorus]` or `[Verse 2]`, lines that are only a label, like `(Chorus)`, `Chorus:`, `Chorus` or `Chorus x2`, also numbered or in German, Spanish or Italian (`1. Strophe`, `Coro`; but not `Break`, which may be shouted), repeat marks like `(x2)`, also inside a label (`[Refrain (2x)]`), an `x2` at the end of a line and a line that is only `x2` or `Repeat chorus` are not sung and are left out. Other words in brackets, like `(oh yeah)` or `[oh yeah]`, are backing vocals and become freestyle notes. Punctuation or a symbol standing alone between the words is no note of its own: closing punctuation joins the word before it (`Hello , world` is sung as `Hello, world`), dashes and music notes (`-`, `♪`) are left out, and symbols that are sung as words (`&`, `+`, `%`) stay.

**Placing the lines:** The words are placed with wav2vec2 forced alignment over the whole vocal track, and the LRC times of the lines check the result. Four alignments are made: with the model run over the whole song or over 30-s pieces, each with two ways of ending the path after the last word. The one with the most lines at their LRC times is used. A line that lands in a poor place moves to a clearly better place near its LRC time, and lines at the end that the recording does not have (lyrics of a longer version) are left out. On a benchmark of 89 songs, the share of words that start more than 1 s off went from 6.5 % to 4.9 %. When the free GPU memory does not hold the run over the whole song at once (about 2 GB for each 100 s of audio with the English model; it is measured on 40 s of the song first), or the memory runs out, the 30-s pieces are used alone. On Windows such a run would not run out of memory but go on in the shared system memory, many times slower and taking the memory from the rest of the system.

**Plain lyrics** from LRCLIB (without timestamps) make the notes when no synced lyrics were found or the synced ones could not be aligned. Their lines are read as above, and each one becomes a line of the chart. The words of the transcription give the lines their times: a line starts where the transcription heard the first of its words (counted back by the words before it in the line), and lines without a heard word are spread between the others by their number of words. When at least half of the lines (and at least 8) get a time this way, the lines are placed as synced lyrics are. Otherwise all lines are placed on one path over the whole vocal track; that is also the case when the transcription was skipped for synced lyrics found before it. With remote speech-to-text (`--remote_stt`), which replaces the transcription, the words the remote service heard give the lines their times in the same way. On the benchmark, with the LRC lines without their times as plain lyrics, the share of words that start more than 1 s off went from 6.5 % to 4.8 % with the words of a local transcription, and to 5.0 % with those of remote speech-to-text. The settings info file (`--write_settings_info`) tells which way the lines were placed. Only when the plain lyrics cannot be aligned either do the words come from remote speech-to-text (`--remote_stt`) or the transcription. When the transcription runs, the plain lyrics also **post-correct** its words by comparing them against the verified lyrics text, improving recall by ~5 percentage points.

The GUI's quality badge shows LRCLIB lyrics availability (✅ Synced lyrics / ⚠️ Plain lyrics only / 📝 No lyrics) so you can see at a glance whether the reference-lyrics-first pipeline will be used. It searches as a conversion of the video does: with the artist and title that the conversion takes from the video and looks up in the song database, and with the length of the video.

### 🧪 Experimental Features

These features are experimental and disabled by default. They may change or be removed in future versions.

#### Library Triage Tool

`tools/library_triage.py` scans a song library and moves broken songs (or, optionally, ones that score badly against their own extracted vocals) into a separate directory. It defaults to a safe dry run that changes nothing. Because it **moves files on disk**, read the full documentation before using it:

* 📄 **[Library Triage Tool documentation](docs/library-triage.md)**

#### Chart Benchmark Tool

`tools/chart_benchmark.py` converts a reproducible sample of songs from your own UltraStar library and measures how close the generated charts come to the existing hand-made charts (note timing, length, pitch, which passages are charted at all and whether the lyrics sit under the right notes), including game-like piano-roll images. Use it to check whether a pipeline change actually improves the charts — the game score alone cannot tell. All data stays in a work directory outside the repository:

* 📄 **[Chart Benchmark Tool documentation](docs/chart-benchmark.md)**

#### LLM Lyric Correction (`--llm_correct`)

Post-corrects WhisperX transcription using an OpenAI-compatible LLM API. The LLM sees sentence-level context and fixes misheard or misspelled words while preserving all timing data. If the API is unreachable or returns an error, the original lyrics are kept unchanged (fail-open). It corrects the words of the local transcription, so it does not run when that is skipped (synced lyrics found before it, or remote speech-to-text).

Related flags:
* `--llm_api_base_url` -- Base URL of the API (default: `https://api.openai.com/v1`)
* `--llm_api_key` -- API key (can also be set via `LLM_API_KEY` environment variable)
* `--llm_model` -- Model name (default: `gpt-4o-mini`)
* `--llm_no_retry` -- Disable automatic retry on rate limit errors (retry is enabled by default)
* `--llm_retry_wait` -- Seconds to wait between retries (default: `60`)
* `--llm_retry_max` -- Maximum retries per text chunk (default: `3`)

When using free-tier APIs like Groq, you may encounter HTTP 429 (Too Many Requests) errors during peak usage.
By default, UltraSinger automatically waits and retries. The retry status is logged to the console and recorded
in the `ultrasinger_parameter.info` file (if `--write_settings_info` is enabled).

**Model recommendations based on benchmark testing:**

| Scenario | Provider | Model | Notes |
|----------|----------|-------|-------|
| Best quality | Groq | `qwen/qwen3-32b` | 99% accuracy, 0 degradations |
| Fastest | Groq | `meta-llama/llama-4-scout-17b-16e-instruct` | 96% accuracy, 0.2s latency |
| Safe choice | Groq | `llama-3.3-70b-versatile` | 90% accuracy, 0 degradations |
| Default | OpenAI | `gpt-4o-mini` | Good quality, pay-per-use |
| Local (Ollama) | Ollama | >=32B models recommended | ⚠️ 8B models may degrade lyrics |

**Cost:** Groq offers a free plan (no credit card required) that is sufficient for typical UltraSinger usage.
A typical song requires ~5 API requests and ~700 tokens. With the free plan limits for `qwen/qwen3-32b`
(1,000 requests/day, 500K tokens/day as of March 2026), you can process **~200 songs per day** at no cost.
Paid plans are only needed for bulk processing well beyond that. See
[Groq rate limits](https://console.groq.com/docs/rate-limits) for current limits.

```bash
# With Groq (free plan available, recommended)
UltraSinger.py -i song.mp3 --llm_correct \
  --llm_api_base_url https://api.groq.com/openai/v1 \
  --llm_api_key gsk_xxx \
  --llm_model qwen/qwen3-32b

# With OpenAI
UltraSinger.py -i song.mp3 --llm_correct --llm_api_key sk-xxx

# With local Ollama (>=32B model recommended)
UltraSinger.py -i song.mp3 --llm_correct \
  --llm_api_base_url http://localhost:11434/v1 \
  --llm_api_key ollama \
  --llm_model qwen2.5:32b

# API key via environment variable
export LLM_API_KEY=gsk_xxx
UltraSinger.py -i song.mp3 --llm_correct --llm_api_base_url https://api.groq.com/openai/v1
```

#### Remote Speech-to-Text (`--remote_stt`)

Sends the song's audio to an external OpenAI-compatible speech-to-text API (e.g. Groq's Whisper endpoint) instead of running local Whisper transcription. This is meant for machines without a capable GPU, where local Whisper large-v2 can take several minutes per song.

**Text and rough times; the notes are timed locally.** The remote service supplies the transcript *text* and, when it returns them, the rough times of its words and segments. The notes never take these times: the words are placed afterward by the same local wav2vec2 CTC forced-alignment model the reference-lyrics-first pipeline already uses for LRCLIB plain-text lyrics. This matters because prior testing (see the Groq API evaluation in project history) showed remote Whisper timestamps are roughly 3x worse than local wav2vec2 alignment. The remote times only tell that alignment where each line is sung: each remote segment becomes a line, and the remote words give the lines their times as the words of a local transcription give them to plain lyrics (see *Plain lyrics* above) — the lines of the remote text, and those of LRCLIB plain lyrics. On the benchmark of 89 songs, the share of words that start more than 1 s off went from 5.5 % to 3.7 % for the remote text, and from 6.5 % to 5.0 % for plain lyrics. A service that does not return times is asked for the text alone, which is then placed on one path over the whole song.

**Where it fits in the lyrics-source order:**
1. LRCLIB synced lyrics (unchanged, highest priority)
2. LRCLIB plain lyrics + local alignment, their lines placed by the remote words
3. **Remote STT text + local alignment** (only if enabled and no LRCLIB lyrics were found or could be aligned)
4. Local transcription, Qwen3-ASR or Whisper (default fallback, and still the default when remote STT is disabled)

With remote STT enabled, the local transcription is skipped: it only runs when none of the sources above could be aligned. The remote service is asked once per song, when no synced lyrics made the notes (also when synced lyrics were found but could not be aligned); when synced lyrics make the notes, no audio is sent.

Any remote STT failure (network error, timeout, auth error, oversized file, empty response) fails open to the local transcription — a conversion never aborts because of it.

**Privacy note:** enabling this sends the song's audio to a third-party service. Only enable it if you accept your audio leaving your machine, and review the provider's data-retention policy.

**Cost/size note:** the audio is uploaded as 16 kHz mono FLAC (about 1 MB per minute of song). Most OpenAI-compatible providers cap uploads around 25 MB; UltraSinger checks the file size before uploading and falls back to local Whisper if it's too large rather than attempting to chunk it.

Related flags:
* `--remote_stt_api_base_url` -- Base URL of the API (default: `https://api.groq.com/openai/v1`)
* `--remote_stt_api_key` -- API key (can also be set via `ULTRASINGER_REMOTE_STT_API_KEY` environment variable)
* `--remote_stt_model` -- Model name (default: `whisper-large-v3`)
* `--remote_stt_no_retry` -- Disable automatic retry on rate limit errors (retry is enabled by default)
* `--remote_stt_retry_wait` -- Seconds to wait between retries (default: `60`)
* `--remote_stt_retry_max` -- Maximum retries (default: `3`)

When using free-tier APIs like Groq, you may encounter HTTP 429 (Too Many Requests) errors during peak usage.
By default, UltraSinger automatically waits and retries — honoring the `Retry-After` response header when the
provider sends one, otherwise waiting `--remote_stt_retry_wait` seconds. This is especially useful for GPU-less
users who would otherwise want to skip a multi-minute local Whisper fallback. The retry status is logged to the
console and recorded in the `ultrasinger_parameter.info` file (if `--write_settings_info` is enabled).

```bash
# With Groq (reference provider, whisper-large-v3)
UltraSinger.py -i song.mp3 --remote_stt \
  --remote_stt_api_base_url https://api.groq.com/openai/v1 \
  --remote_stt_api_key gsk_xxx \
  --remote_stt_model whisper-large-v3

# API key via environment variable
export ULTRASINGER_REMOTE_STT_API_KEY=gsk_xxx
UltraSinger.py -i song.mp3 --remote_stt
```

#### Syllable-Level Note Splitting (`--syllable_split`)

Splits word-level notes into syllable-level notes using hyphenation (pyhyphen). This produces output closer to how hand-made karaoke charts are usually formatted, where each syllable gets its own note instead of one note per word.

```commandline
-i XYZ --syllable_split
```

#### Vocal Gap Fill (`--vocal_gap_fill`)

Detects unrecognized vocal segments between transcribed words -- such as ad-libs, melismas, or sustained vowels ("oohs") -- and fills them with placeholder notes. Uses SwiftF0 pitch confidence to distinguish singing from silence. Gaps are filled with `~` marker notes.

```commandline
-i XYZ --vocal_gap_fill
```

#### Pitch-Based Note Generation (`--pitch_notes`)

Generates notes directly from the pitch contour (SwiftF0) instead of the word-level timing boundaries of the transcription. This is best for melismatic songs with runs, slides, and ornaments where word-level timing produces flat, unusable notes (e.g. R&B runs, opera, yodel passages).

Notes are segmented by pitch stability (sustained pitch changes of 2+ semitones held for 80ms+), then split at word boundaries. The transcribed lyrics are overlaid by time alignment, and when lyrics lookup is active, LRCLIB reference lyrics fill remaining placeholder notes.

```commandline
-i XYZ --pitch_notes
```

#### Model-Based Note Segmentation (`--segmentation_model`)

Instead of deriving notes from the word timing, a small neural network predicts directly on the separated vocal where notes start, how long they are, and which passages are charted at all (e.g. spoken parts become freestyle, background vocals are left out). The lyrics from the usual sources (synced lyrics or Whisper, with forced alignment) are then split into syllables and placed onto these notes; syllables held over several notes get `~` continuations, and syllables without a note of their own join a neighbouring note of the same line.

No model is shipped with UltraSinger. You train your own on your UltraStar song library with `tools/train_segmentation.py` or with **Train...** next to the Segmentation Model setting in the GUI — the more well-timed charts it sees, the better. Keep the model and the extracted training data private. Measure the result on songs the model was not trained on with the [Chart Benchmark Tool](#chart-benchmark-tool).

* 📄 **[Segmentation model documentation](docs/segmentation-model.md)**

```commandline
-i XYZ --segmentation_model D:\Models\segmentation.pt
```

GUI: Settings → Experimental Features → "Segmentation Model". Requires vocal separation; if the model file is missing or the step fails, the word-based notes are kept. With a model, the timing refinement (snapping note starts to audio onsets) and the GAP sweep (shifting all notes together to where the game scores them best) are skipped, because the model places note starts more precisely.

To share one model between computers, keep it in a (private) GitHub repository and let UltraSinger download it: `--segmentation_model_repo owner/repo[/path/model.pt]` (default file `segmentation.pt`) with the `ULTRASINGER_MODEL_TOKEN` environment variable (or `--segmentation_model_token <token>`) for private repositories. The model is cached locally and only downloaded again when it changed; a download is only kept when it arrived complete and loads as a model. GUI: "Model Repository" and "Repository Token" (stored in the system keyring).

```commandline
-i XYZ --segmentation_model_repo owner/models-repo
```
**Lead-vocal pitch** (on by default with a model; `--disable_lead_vocal_pitch`, GUI: "Lead Vocal Pitch"): harmonies, backing vocals and duet parts make the pitch tracker follow the loudest voice. The vocal stem is therefore split once more into lead and backing vocals with a karaoke separation model, and the note pitches are taken from the lead vocal — but only when the lead stem kept at least 80 % of the singing (otherwise the separation misjudged the melody and the full vocal stem is used). The pitch refinement and the game-score report then compare the notes with the lead stem as well (notes without a detected tone there with the full vocal stem). On a 100-song benchmark this raised the median agreement with hand-made charts by about 1.7 points, mainly on songs with several voices. The separation needs about 2.2 GB of free GPU memory; with less it is skipped, since it would run in shared system memory and take very long, and the note pitches come from the full vocal stem.

#### Freestyle Detection (`--detect_freestyle`)

Detects vocal passages that cannot be reliably pitched and marks them as freestyle notes (displayed but not scored). This covers growls, screams, harsh vocals, rap, spoken word, and any non-melodic vocal style — useful for any song where parts of the vocal performance fall outside traditional singing.

The primary detection method uses **HPSS (Harmonic-Percussive Source Separation)**: clean singing has a high harmonic-to-total energy ratio (typically 0.7+), while unpitchable passages have a low ratio (below 0.40). This approach is **genre- and gender-independent**. When HPSS is unavailable (e.g. no separated vocal audio or HPSS preprocessing fails), a fallback method uses SwiftF0 pitch confidence and pitch stability analysis.

```commandline
-i XYZ --detect_freestyle
```

You can tune the detection thresholds:
```commandline
-i XYZ --detect_freestyle --freestyle_harmonicity 0.35 --freestyle_energy 0.005
```

#### ptAKF Chart Refit (`--ptakf_refit`)

Rebuilds every note's boundaries and pitch from the **game's own pitch detector** (ptAKF, via [ultrastar-score](https://github.com/MrDix/ultrastar-score)) instead of SwiftF0. The USDX-style scorer samples one ptAKF frame per beat — this pass charts exactly what that detector hears: only voiced beats are charted (breaths, consonants and reverb tails stay note-free), and notes are split at sustained pitch changes. Lyrics, line structure, BPM and GAP are kept; the first sub-note keeps the syllable, continuations become `~`.

This maximizes the score an exact-match singer (or the extracted vocal track itself) can achieve. Benchmark over 10 songs: Medium score 72.8% → 90.0%, Easy 81.1% → 94.2%. The trade-off is a higher note count (roughly +40%, many short `~` notes). Enabled by default (disable with `--disable_ptakf_refit`).

```commandline
-i XYZ --ptakf_refit
```

Notes shorter than `--ptakf_refit_min_note_ms` (default 100) are merged back into a neighbour whenever that loses no score:
```commandline
-i XYZ --ptakf_refit --ptakf_refit_min_note_ms 150
```

With `--ptakf_refit_fill`, sung passages that no note covers (ad-libs, vocalises, melisma tails beyond word boundaries) are additionally charted as `~` notes — the chart then covers everything the singer actually sings. Only uncharted voiced runs of at least `--ptakf_refit_fill_min_ms` (default 300) are filled, so separation bleed and noise stay note-free:
```commandline
-i XYZ --ptakf_refit --ptakf_refit_fill
```

Requires the optional scoring dependency (`pip install "ultrastar-score"` — already included by the install scripts). Without it the pass is skipped gracefully.

#### Golden Notes (`--golden_notes`)

Marks held notes as **golden** (`*`) bonus notes, which are worth double score in-game. UltraSinger's charts otherwise never contain any golden notes.

Hand-made professional charts mark only a handful of golden notes per song (typically about ten single notes, whatever the song's length), mostly long held notes spread over the whole song. This pass does the same: it marks 10 normal notes (`note_type == ":"`, held for at least 200ms; a long `~` continuation is a held vowel and counts too), but never more than 15% of all scorable notes (short songs) and at most 3 in each tenth of the song, so they do not pile up in one section — long notes tend to gather at the end of a song. Notes are ranked by duration, weighted by how well the singing stays on the note's pitch (the share of the note's pitch frames that are confident and within one semitone of it, octaves ignored), because a golden note nobody can hit is worse than none; the highest note of a phrase gets a bonus. Freestyle and rap notes are never marked. Runs last, after refinement and the ptAKF chart refit, so it always marks the final note boundaries.

```commandline
-i XYZ --golden_notes
```

Disabled by default, since it changes the in-game score distribution.

### 🏆 Ultrastar Score Calculation

The score that the singer in the audio would receive will be measured.

#### Game score (ptAKF)

UltraSinger reports the **game score**: the written chart scored against the extracted vocals with [ultrastar-score](https://github.com/MrDix/ultrastar-score) - the same C++ ptAKF pitch-detection and scoring algorithm the games (Vocaluxe/USDX) use - at all three difficulties (Easy +-2 / Medium +-1 / Hard 0 semitones, octave-folded):

```text
[UltraSinger] Game score (ptAKF): Easy 96.7% | Medium 87.1% | Hard 66.6%
```

**This is the number to trust when comparing conversions**, and it is UltraSinger's primary score output. It also appears with a per-difficulty breakdown (total %, notes/golden/line-bonus points, beats hit/total) in the settings info file, and the Medium score is appended to the chart's `#CREATOR` header line. It shows up in the GUI queue result line as well.

#### Chart style: singable vs score (`--chart_style`, GUI: Settings → Post-Processing → "Chart Style")

There is a real trade-off between a chart that *scores* high and one that is good to *sing*:

* **`singable`** (default): natural, held notes like a professionally-made karaoke chart. This is the best result to actually sing.
* **`score`**: rebuilds every note onto the game's exact per-beat tones (the "ptAKF refit"), which maximises the reported game score but produces many short notes that trace vibrato and ornaments — accurate to the recording, but harder to sing.

A measurement against professional reference charts makes the trade-off concrete: those hand-made charts score only ~77% Medium against the isolated vocals, and the `singable` style lands at that same level — because the scorer rewards tracing the vocal exactly, a simplified singable chart *cannot* score as high as one that traces every wiggle. So a lower game score in `singable` style is expected and is not a defect; the `score` style's higher number reflects over-fitting to the recording, not a better karaoke chart. Advanced users can still force the refit on or off directly with `--ptakf_refit` / `--disable_ptakf_refit`, which override `--chart_style`.

#### Simple / accurate score (fallback only)

Without the optional `ultrastar-score` package installed, UltraSinger falls back to an internal simple/accurate estimate instead. Ultrastar is not interested in pitch heights: as long as a note is in the pitch range A-G you get one point (simple), while accurate requires the exact octave. Golden notes count double, as in the games, and a chart sung right throughout reaches the full 10000 points. This fallback measures against the SwiftF0 pitch data the chart was built from rather than the actual in-game ptAKF detector, so it can rank charts differently — prefer the game score above whenever it is available.


### 📟 Use GPU

With a GPU you can speed up the process. Also the quality of the transcription and pitching is better.

You need an NVIDIA CUDA device for this to work. Sorry, there is no CUDA device for macOS.

For GPU support on Windows and Linux, the installation script automatically installs PyTorch with CUDA support (CUDA 12.8).

It is optional (but recommended) to install the CUDA driver for your GPU: see [CUDA driver](https://developer.nvidia.com/cuda-downloads).
Also check your GPU CUDA support. See [CUDA support](https://gist.github.com/standaloneSA/99788f30466516dbcc00338b36ad5acf)
The CUDA build needs a GPU with compute capability 7.0 or newer on Linux (6.1 or newer on Windows) and a driver that supports CUDA 12 (`nvidia-smi` shows the supported CUDA version). `auto_install` checks both and installs the CPU build otherwise.

For a manual installation, do what the CUDA install script does: in `pyproject.toml`, change the URL of the `pytorch` index from `https://download.pytorch.org/whl/cpu` to `https://download.pytorch.org/whl/cu128`, then run:
```bash
uv lock
uv sync --extra gui --extra scoring --extra potoken
```

#### Crashes due to low VRAM

At full batch size (16) the default Whisper model (`large-v2`) needs roughly
8 GB of VRAM. UltraSinger scales the batch size to the detected GPU memory
automatically (16 on ~8 GB and larger, 8 on 5-7 GB, 4 on smaller cards), so
most GPUs fit as-is. If it still runs out of memory, there are two independent
levers, in order of preference:

1. **Lower the batch size further** (`--whisper_batch_size 2`, or 1). This
   processes fewer audio segments in parallel: it is slower, but the
   transcription is **unchanged** — so this is the safe lever, try it first.
2. **Switch to int8** (`--whisper_compute_type int8`). This halves the model's
   memory at a small accuracy cost — add it only if lowering the batch size
   alone is not enough.

You can also offload transcription to the cloud with `--remote_stt` (frees the
GPU entirely; see [Remote Speech-to-Text](#remote-speech-to-text---remote_stt)),
use a smaller Whisper model, or force CPU with `--force_cpu`.

### 📦 Containerized (Docker or Podman)

Run UltraSinger in a container — no local Python install required.

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

The image contains the conversion pipeline (CLI) with FFmpeg and PyTorch, but not the GUI, the game-score engine or the PO-token provider for full-quality video downloads. The models are downloaded on first use and kept between runs. For detailed setup (volumes, cookies, Podman), see [container/README.md](container/README.md)
