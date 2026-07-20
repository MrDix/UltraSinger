[![Discord](https://img.shields.io/discord/1048892118732656731?logo=discord)](https://discord.gg/rYz9wsxYYK)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rakuri255/UltraSinger/blob/master/colab/UltraSinger.ipynb)
![Status](https://img.shields.io/badge/status-development-yellow)

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/rakuri255/UltraSinger/main.yml)
[![GitHub](https://img.shields.io/github/license/rakuri255/UltraSinger)](https://github.com/rakuri255/UltraSinger/blob/main/LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/rakuri255/ultrasinger/badge)](https://www.codefactor.io/repository/github/rakuri255/ultrasinger)
[![Check Requirements](https://github.com/rakuri255/UltraSinger/actions/workflows/main.yml/badge.svg)](https://github.com/rakuri255/UltraSinger/actions/workflows/main.yml)
[![Pytest](https://github.com/rakuri255/UltraSinger/actions/workflows/pytest.yml/badge.svg)](https://github.com/rakuri255/UltraSinger/actions/workflows/pytest.yml)
[![docker](https://github.com/rakuri255/UltraSinger/actions/workflows/docker.yml/badge.svg)](https://hub.docker.com/r/rakuri255/ultrasinger)
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
    - [🧪 Experimental Features](#-experimental-features)
      - [Library Triage Tool](#library-triage-tool)
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
* [Node.js](https://nodejs.org) — *recommended:* needed for full-quality video downloads (the install script builds the local PO-token provider with it; without Node.js, downloads fall back to reduced quality — you can install it later and re-run the install script).
* Python 3.12 or 3.13 ([Download](https://www.python.org/downloads/)) — *optional:* if neither is installed (e.g. you only have a newer version), the install scripts automatically download a portable, self-contained Python 3.12 via uv (into uv's per-user directory, or into an app-local `.uv-python/` folder if the global store is unusable); your system Python is never touched.
* Behind a corporate proxy? See [Corporate proxy / firewall](#-corporate-proxy--firewall) — the installer detects `HTTP(S)_PROXY` automatically.

**Install**

1. Get the source code:
   ```bash
   git clone https://github.com/rakuri255/UltraSinger.git
   cd UltraSinger
   ```
2. Run the installer — `install\auto_install.bat` (Windows) or `install/auto_install.sh` (Linux/macOS). It takes care of everything:
   * detects your NVIDIA GPU via `nvidia-smi` and picks the CUDA or CPU build automatically (force with `--cuda` / `--cpu`),
   * installs all dependencies including the GUI, scoring engine and PO-token plugin,
   * builds the local PO-token provider when Node.js is available,
   * detects proxy environment variables and enables OS-store TLS for uv automatically,
   * and prints tailored advice when your GPU has little VRAM (< 8 GB) or none was found — including how a free cloud key can replace the slowest local step (`--remote_stt`).

   You can also run a specific sub-script directly: `install/CUDA/…` (NVIDIA GPU, Windows/Linux) or `install/CPU/…` (no NVIDIA GPU, or macOS — Apple Silicon has no CUDA).

**Update an existing installation**

Run `install\update.bat` (Windows) or `install/update.sh` (Linux/macOS). **This is the only command you need to update** — you never have to re-run `auto_install`. It pulls the latest changes, syncs the Python packages into `.venv` (so newer dependency versions land too), refreshes the PO-token provider, and re-checks ffmpeg — and it transparently handles the CUDA case, where the installer protects `pyproject.toml`/`uv.lock` from git resets (a plain `git pull` would refuse to update those files with "Your local changes ... would be overwritten"). `auto_install` is only for the first-time install.

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
  - **Language badge** — detected language code with color-coded background: green = synced lyrics (best quality), orange = plain lyrics or fallback, red = transcribed only (Whisper)
  - **Lyrics source** — "Synced lyrics", "Plain lyrics", or "Transcribed"
  - **Info button** — view the settings used for this conversion
  - **Folder button** — open the output folder in your system file manager (hover to see the path)
  - **Re-queue button** — re-queue the song with different settings (opens per-song settings dialog; useful for correcting wrong language detection)
- **Preferences** — Default output folder, LLM/Groq API configuration, cookie management.

> **Note:** The GUI is a native desktop application (Qt). It requires a display and cannot run inside a Docker container. Use the [CLI](#run-cli) for containerized workflows.

#### Full-quality video downloads (PO token)

The video platform now delivers its player streams via **SABR**, where the media data and the required *Proof-of-Origin* (PO) token are sent in binary POST bodies that an embedded browser cannot read. Without a PO token, `yt-dlp` is limited to a reduced format set (typically 360p) or blocked with HTTP 403.

To restore full-quality downloads UltraSinger uses the maintained [`bgutil-ytdlp-pot-provider`](https://github.com/Brainicism/bgutil-ytdlp-pot-provider) yt-dlp plugin (installed by the install scripts via the `potoken` extra). The plugin fetches PO tokens from a small local **provider server**.

- **Node.js (default, no Docker):** if [Node.js](https://nodejs.org) is installed, the install scripts build the provider automatically into `.potoken/` and the **GUI starts it on launch and stops it on exit** — nothing else to do. If Node.js was missing during install, install it and re-run the install script. The console tab reports the provider status (`[PO-Token] …`).
- **Docker (fallback):** if no Node.js provider is set up but Docker is available, the GUI auto-starts the container `brainicism/bgutil-ytdlp-pot-provider` on port 4416 instead. You can also run it yourself:
  ```bash
  docker run -d --rm -p 4416:4416 brainicism/bgutil-ytdlp-pot-provider
  ```

Once a provider responds on `http://127.0.0.1:4416`, `yt-dlp` uses it transparently for every download (GUI and CLI). If no provider is available the app still works, but video downloads fall back to the limited formats.

Provider behaviour can be tuned in the GUI config (`~/.ultrasinger`): `potoken_auto_start` (check on launch), `potoken_auto_start_node` / `potoken_auto_start_docker` (allow the respective auto-launch), `potoken_base_url` (custom server URL).

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

**Provider behind an HTTP proxy:** the provider setup automatically applies a small workaround for an upstream axios bug (missing `proxy: false`) that otherwise makes the provider send absolute-form GETs to the proxy instead of CONNECT tunnels — enterprise proxies answer those with 502 and token generation fails even though the same URLs work via `curl`. Re-run the install script once after updating to get the patched build.

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
    # Default is whisper
    --whisper               Multilingual model > tiny|base|small|medium|large-v1|large-v2|large-v3  >> ((default) is large-v2)
                            English-only model > tiny.en|base.en|small.en|medium.en
    --whisper_align_model   Custom wav2vec2 forced-alignment model from HuggingFace
                            (e.g. --whisper_align_model gigant/romanian-wav2vec2)
    --language              Override language for alignment and hyphenation.
                            Priority: --language > video platform metadata > Whisper auto-detect.
                            For video URLs, the video language is used automatically.
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
    --ignore_audio          Skip transcription for audio/video input and only recompute pitch (advanced;
                            e.g. reuse existing lyrics/timing). Automatically enabled when the input (-i)
                            is an ultrastar.txt file (re-pitch mode).

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
                            accuracy. Falls back to standard Whisper pipeline when disabled or when no
                            synced lyrics are available. Enabled by default.

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
    --disable_refine_pitch      Disable pitch refinement (enabled by default when refine is on)
    --disable_refine_timing     Disable timing refinement (enabled by default when refine is on)
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
    --golden_notes              Mark a subset of held notes as golden "*" bonus notes, worth double score
                                in-game (experimental). Disabled by default, since it changes the score
                                distribution. Only real syllable notes held for at least 350ms are
                                eligible; freestyle, rap and tilde-continuation notes are never marked.
                                Golden notes are capped at 15% of all scorable notes and spread across
                                the whole song rather than clustered in one section.


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
                                  for GPU-less machines (disabled by default). Text only --
                                  timing is always computed locally.
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
    --disable_karaoke       Disable creation of karaoke txt file. Karaoke is enabled by default.
    --create_audio_chunks   Enable creation of audio chunks. Disabled by default.
    --disable_midi          Disable MIDI file creation. MIDI is created by default.
    --no_metadata_tags      Skip writing ID3/Vorbis tags (title/artist/year/genre/cover) to output
                            audio. Tags are written by default.
    --keep_audio_in_video   Keep full audio (vocals+instrumental) in the output video. Disabled by default.
    --write_settings_info   Write ultrasinger_parameter.info with all settings and scores to output dir.
    --plot                  Enable creation of plots. Disabled by default.
    --keep_cache            Keep cache folder after creation. Removed by default.

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
                            local file (-i), but artist/title/thumbnail are fetched from
                            the video URL instead of relying on the filename.
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

    [device]
    --force_cpu             Force all steps to be processed on CPU.
    --force_whisper_cpu     Only whisper will be forced to cpu

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

This re-pitch the audio and creates a new txt file.

```commandline
-i "input/ultrastar.txt"
```

### 🗣 Transcriber

Keep in mind that while a larger model is more accurate, it also takes longer to transcribe.

#### Whisper

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

#### Octave consistency (`--octave_consistency`, GUI: Settings → Post-Processing → "Octave Consistency")

The stronger octave repair: pitch trackers scatter individual notes *and short runs* into the wrong octave — measured against professional reference charts, generated charts contained about **10× as many jarring octave-size jumps** between adjacent notes (507 vs 47 across 8 songs), which makes passages extremely confusing to sing. `--octave_consistency` keeps every note's pitch class and re-chooses only its octave via dynamic programming over the whole song, balancing melodic smoothness against fidelity to what the tracker detected. The balance is self-limiting: short wrong-octave scatter (roughly 1–3 notes) is folded onto the melody line, while a genuine octave passage of about five notes or more is cheaper to keep — so real octave jumps and wide-range songs survive (validated on the same 8 reference songs: jumps dropped to professional-chart level while octave-sensitive pitch agreement with the references stayed within 1 percentage point). The game score is unaffected because scoring folds octaves. Disabled by default.

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

### Settings Info File

When enabled, UltraSinger writes a `ultrasinger_parameter.info` file to the output directory containing all conversion settings and a detailed pipeline trace. The `[Pipeline]` section shows exactly how the song was processed:

- Which pipeline was used (Reference-Lyrics-First vs Standard Whisper)
- How the language was determined (manual / video platform metadata / Whisper fast-detect / Whisper full)
- Whether Whisper received an explicit language hint or used auto-detect
- Language correction history (if fast-detect was wrong and Whisper corrected it)
- Whether the reference pipeline was recovered after language correction
- LRCLIB lyrics availability (synced / plain / none)
- LLM correction results and score breakdown

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
  2. Video platform metadata (yt-dlp extracts the video's language automatically)
  3. Whisper tiny fast-detect (~2-3 seconds)
  4. Whisper full transcription (fallback)

The pipeline is **enabled by default** and falls back to standard Whisper transcription when no synced lyrics are available on LRCLIB. Use `--disable_reference_lyrics` to force the Whisper path.

**Auto-recovery from wrong language detection:** If the fast language detection (Whisper tiny, ~2-3s) misidentifies the language (e.g. "cy" instead of "en"), the reference pipeline may fail because no alignment model exists for that language. In this case, UltraSinger automatically runs full Whisper transcription which detects the correct language, then **retries the reference pipeline** with the corrected language — recovering to full synced-lyrics quality without user intervention.

**Plain lyrics** from LRCLIB (without timestamps) do not enable the reference pipeline. They are used to **post-correct** Whisper transcription by comparing words against the verified lyrics text, improving recall by ~5 percentage points. Timing still comes from Whisper in this case.

The GUI's quality badge shows LRCLIB lyrics availability (✅ Synced lyrics / ⚠️ Plain lyrics only / 📝 No lyrics) so you can see at a glance whether the reference-lyrics-first pipeline will be used.

### 🧪 Experimental Features

These features are experimental and disabled by default. They may change or be removed in future versions.

#### Library Triage Tool

`tools/library_triage.py` scans a song library and moves broken songs (or, optionally, ones that score badly against their own extracted vocals) into a separate directory. It defaults to a safe dry run that changes nothing. Because it **moves files on disk**, read the full documentation before using it:

* 📄 **[Library Triage Tool documentation](docs/library-triage.md)**

#### LLM Lyric Correction (`--llm_correct`)

Post-corrects WhisperX transcription using an OpenAI-compatible LLM API. The LLM sees sentence-level context and fixes misheard or misspelled words while preserving all timing data. If the API is unreachable or returns an error, the original lyrics are kept unchanged (fail-open).

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

**Text only, never timing.** The remote service only ever supplies the transcript *text*. Timing is always computed afterward by the same local wav2vec2 CTC forced-alignment model the reference-lyrics-first pipeline already uses for LRCLIB plain-text lyrics — remote timestamps are discarded and never trusted. This matters because prior testing (see the Groq API evaluation in project history) showed remote Whisper timestamps are roughly 3x worse than local wav2vec2 alignment.

**Where it fits in the lyrics-source order:**
1. LRCLIB synced lyrics (unchanged, highest priority)
2. LRCLIB plain lyrics + local alignment (unchanged)
3. **Remote STT text + local alignment** (new — only if enabled and no LRCLIB lyrics found)
4. Local Whisper transcription (default fallback, and still the default when remote STT is disabled)

Any remote STT failure (network error, timeout, auth error, oversized file, empty response) fails open to local Whisper — a conversion never aborts because of it.

**Privacy note:** enabling this sends the song's audio to a third-party service. Only enable it if you accept your audio leaving your machine, and review the provider's data-retention policy.

**Cost/size note:** most OpenAI-compatible providers cap uploads around 25 MB; UltraSinger checks the file size before uploading and falls back to local Whisper if it's too large rather than attempting to chunk it.

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

Splits word-level notes into syllable-level notes using hyphenation (pyhyphen). This produces output closer to how commercial karaoke games like SingStar format their songs, where each syllable gets its own note instead of one note per word.

```commandline
-i XYZ --syllable_split
```

#### Vocal Gap Fill (`--vocal_gap_fill`)

Detects unrecognized vocal segments between transcribed words -- such as ad-libs, melismas, or sustained vowels ("oohs") -- and fills them with placeholder notes. Uses SwiftF0 pitch confidence to distinguish singing from silence. Gaps are filled with `~` marker notes.

```commandline
-i XYZ --vocal_gap_fill
```

#### Pitch-Based Note Generation (`--pitch_notes`)

Generates notes directly from the pitch contour (SwiftF0) instead of Whisper word-level timing boundaries. This is best for melismatic songs with runs, slides, and ornaments where Whisper's word-level timing produces flat, unusable notes (e.g. R&B runs, opera, yodel passages).

Notes are segmented by pitch stability (sustained pitch changes of 2+ semitones held for 80ms+), then split at word boundaries. Whisper lyrics are overlaid by time alignment, and when lyrics lookup is active, LRCLIB reference lyrics fill remaining placeholder notes.

```commandline
-i XYZ --pitch_notes
```

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

Marks a subset of held notes as **golden** (`*`) bonus notes, which are worth double score in-game. UltraSinger's charts otherwise never contain any golden notes.

Only real syllable notes are eligible: `note_type == ":"`, not a `~` tilde-continuation, and held for at least 350ms — short notes rarely stay on-pitch long enough to be reliably hit. The number of golden notes is capped at 15% of all scorable notes, and eligible candidates are spread across the whole song (split into as many chunks as golden slots, picking the longest note per chunk) instead of clustering in one section. Runs last, after refinement and the ptAKF chart refit, so it always marks the final note boundaries.

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

Without the optional `ultrastar-score` package installed, UltraSinger falls back to an internal simple/accurate estimate instead. Ultrastar is not interested in pitch heights: as long as a note is in the pitch range A-G you get one point (simple), while accurate requires the exact octave. This fallback measures against the SwiftF0 pitch data the chart was built from rather than the actual in-game ptAKF detector, so it can rank charts differently — prefer the game score above whenever it is available.


### 📟 Use GPU

With a GPU you can speed up the process. Also the quality of the transcription and pitching is better.

You need an NVIDIA CUDA device for this to work. Sorry, there is no CUDA device for macOS.

For GPU support on Windows and Linux, the installation script automatically installs PyTorch with CUDA support.

It is optional (but recommended) to install the CUDA driver for your GPU: see [CUDA driver](https://developer.nvidia.com/cuda-downloads).
Also check your GPU CUDA support. See [CUDA support](https://gist.github.com/standaloneSA/99788f30466516dbcc00338b36ad5acf)

For manual installation, you can use:
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
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
# Build the image (once)
docker compose -f container/compose-cpu.yml build

# Convert a video URL (CPU)
docker compose -f container/compose-cpu.yml run --rm ultrasinger \
    uv run python /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/

# With NVIDIA GPU
docker compose -f container/compose-gpu.yml run --rm ultrasinger \
    uv run python /app/UltraSinger/src/UltraSinger.py \
    -i "https://www.youtube.com/watch?v=XXXXX" -o /app/UltraSinger/output/
```

For detailed setup (volumes, cookies, Podman), see [container/README.md](container/README.md)
