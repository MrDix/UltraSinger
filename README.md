[![Discord](https://img.shields.io/discord/1048892118732656731?logo=discord)](https://discord.gg/rYz9wsxYYK)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rakuri255/UltraSinger/blob/master/colab/UltraSinger.ipynb)
![Status](https://img.shields.io/badge/status-development-yellow)

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/rakuri255/UltraSinger/main.yml)
[![GitHub](https://img.shields.io/github/license/rakuri255/UltraSinger)](https://github.com/rakuri255/UltraSinger/blob/main/LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/rakuri255/ultrasinger/badge)](https://www.codefactor.io/repository/github/rakuri255/ultrasinger)
[![Check Requirements](https://github.com/rakuri255/UltraSinger/actions/workflows/main.yml/badge.svg)](https://github.com/rakuri255/UltraSinger/actions/workflows/main.yml)
[![Pytest](https://github.com/rakuri255/UltraSinger/actions/workflows/pytest.yml/badge.svg)](https://github.com/rakuri255/UltraSinger/actions/workflows/pytest.yml)
[![docker](https://github.com/rakuri255/UltraSinger/actions/workflows/docker.yml/badge.svg)](https://hub.docker.com/r/rakuri255/ultrasinger)

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
      - [LLM Lyric Correction](#llm-lyric-correction---llm_correct)
      - [Syllable-Level Note Splitting](#syllable-level-note-splitting---syllable_split)
      - [Vocal Gap Fill](#vocal-gap-fill---vocal_gap_fill)
    - [🏆 Ultrastar Score Calculation](#-ultrastar-score-calculation)
    - [📟 Use GPU](#-use-gpu)
      - [Considerations for Windows users](#considerations-for-windows-users)
      - [Crashes due to low VRAM](#crashes-due-to-low-vram)
    - [🖥️ GUI](#️-run-gui)
    - [📦 Containerized](#containerized-docker-or-podman)

## 💻 How to use this source code

### Installation

* Install Python 3.12 or 3.13. [Download](https://www.python.org/downloads/)
* Also download or install ffmpeg with PATH. [Download](https://www.ffmpeg.org/download.html)
* Go to folder `install` and run install script for your OS:
  * Choose `GPU` if you have an NVIDIA CUDA GPU.
  * Choose `CPU` if you don't have an NVIDIA GPU or want CPU-only processing.

### Run (CLI)

* In root folder just run `run_on_windows.bat`, `run_on_linux.sh` or `run_on_mac.command` to start the app.
* Now you can use the UltraSinger source code with `py UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]`. See [How to use](#-how-to-use-the-app) for more information.

### 🖥️ Run (GUI)

UltraSinger includes an optional graphical interface with an embedded video browser,
full settings panel, and real-time conversion log.

#### Install GUI dependencies

```bash
uv sync --extra gui
```

> This installs [PySide6](https://doc.qt.io/qtforpython-6/) (Qt 6 for Python) including WebEngine for the embedded browser.

#### Start the GUI

```bash
uv run python src/gui_main.py
```

Or use the platform-specific launcher scripts:

- **Windows:** `run_gui_on_windows.bat`
- **Linux:** `./run_gui_on_linux.sh`
- **macOS:** `./run_gui_on_mac.command`

#### Features

- **Video Browser** — Browse video platforms, log in to your account, and send videos directly to conversion. Cookies are captured automatically for authenticated downloads.
- **Settings Panel** — All CLI parameters available as form controls: Whisper model, language, post-processing options, experimental features, LLM lyric correction, and more.
- **Conversion Queue** — Real-time color-coded log output with stage detection (Separating Vocals → Transcribing → Pitching → …), elapsed timer, and cancel support.
- **Preferences** — Default output folder, LLM/Groq API configuration, cookie management.

> **Note:** The GUI is a native desktop application (Qt). It requires a display and cannot run inside a Docker container. Use the [CLI](#run-cli) for containerized workflows.

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
    # Default is htdemucs
    --demucs              Model name htdemucs|htdemucs_ft|htdemucs_6s|hdemucs_mmi|mdx|mdx_extra|mdx_q|mdx_extra_q >> ((default) is htdemucs)

    [transcription]
    # Default is whisper
    --whisper               Multilingual model > tiny|base|small|medium|large-v1|large-v2|large-v3  >> ((default) is large-v2)
                            English-only model > tiny.en|base.en|small.en|medium.en
    --whisper_align_model   Use other languages model for Whisper provided from huggingface.co
    --language              Override the language detected by whisper for alignment and hyphenation.
                            Default: auto-detect. WARNING: setting this for non-matching songs
                            will degrade alignment quality (e.g. --language en for German songs).
    --whisper_batch_size    Reduce if low on GPU mem >> ((default) is 16)
    --whisper_compute_type  Change to "int8" if low on GPU mem (may reduce accuracy) >> ((default) is "float16" for cuda devices, "int8" for cpu)
    --keep_numbers          Numbers will be transcribed as numerics instead of as words
    --vad_onset             VAD speech activation threshold (0.0-1.0). Lower values capture more vocal
                            segments including soft/breathy singing. >> ((default) is 0.35, WhisperX default: 0.5)
    --vad_offset            VAD speech deactivation threshold (0.0-1.0). Lower values keep segments active
                            longer during vocal dips. >> ((default) is 0.20, WhisperX default: 0.363)
    --no_speech_threshold   No-speech probability threshold (0.0-1.0). Lower values prevent Whisper from
                            classifying singing as silence. >> ((default) is 0.4, WhisperX default: 0.6)

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
    --pitch_change_split    Split notes at pitch change boundaries within a syllable (experimental). Detects
                            sustained pitch changes (melismas, runs) and creates separate notes for each pitch
                            region. Disabled by default.

    [refinement]
    --disable_refine            Disable the reverse-scoring refinement pass. Refinement is enabled by default
                                and uses the game's C++ ptAKF pitch detector to find and fix poorly-scoring notes.
    --refine_from_vocal         (legacy) Explicitly enable refinement (now the default)
    --disable_refine_pitch      Disable pitch refinement (enabled by default when refine is on)
    --disable_refine_timing     Disable timing refinement (enabled by default when refine is on)
    --refine_hit_ratio          Notes below this hit ratio are pitch-corrected (0.0-1.0) >> ((default) is 0.4)
    --refine_timing_threshold   Milliseconds threshold before correcting timing >> ((default) is 30)


    [llm lyric correction]
    --llm_correct           Enable LLM-based lyric post-correction (disabled by default)
    --llm_api_base_url      LLM API base URL >> ((default) is https://api.openai.com/v1)
    --llm_api_key           LLM API key (or set LLM_API_KEY env var)
    --llm_model             LLM model name >> ((default) is gpt-4o-mini)
    --llm_no_retry          Disable automatic retry on rate limit (HTTP 429). Retry is enabled by default.
    --llm_retry_wait        Seconds to wait between retries >> ((default) is 60)
    --llm_retry_max         Maximum retries per text chunk >> ((default) is 3)

    [output]
    --format_version        0.3.0|1.0.0|1.1.0|1.2.0 >> ((default) is 1.2.0)
    --disable_karaoke       Disable creation of karaoke txt file. Karaoke is enabled by default.
    --create_audio_chunks   Enable creation of audio chunks. Disabled by default.
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
    --youtube_url           YouTube URL for metadata lookup when -i is a local file.
                            When -i is a YouTube URL, UltraSinger extracts artist/title
                            directly from YouTube. When -i is a local file, UltraSinger
                            parses the filename ("Artist - Title.ext") and searches
                            MusicBrainz for metadata and cover art. This works well when
                            the filename is descriptive but fails for generic names like
                            "video.avi".
                            --youtube_url provides a fallback: the audio is loaded from the
                            local file (-i), but artist/title/thumbnail are fetched from
                            the YouTube URL instead of relying on the filename.
                            The GUI uses this automatically when downloading via the
                            embedded browser (pre-downloaded audio + YouTube metadata).
                            Not needed when -i is already a YouTube URL.

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

Pitching is done with the `SwiftF0` model, which is faster and more accurate than CREPE.
SwiftF0 automatically detects pitch frequencies between 46.875 Hz (G1) and 2093.75 Hz (C7).
UltraSinger uses 60hz and 400hz

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

When enabled, UltraSinger writes a `ultrasinger_parameter.info` file to the output directory containing all conversion settings, detected language, LLM correction status, and score results. Useful for tracking what parameters produced a given output.

```commandline
-i XYZ --write_settings_info
```

### 🧪 Experimental Features

These features are experimental and disabled by default. They may change or be removed in future versions.

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

### 🏆 Ultrastar Score Calculation

The score that the singer in the audio would receive will be measured.
You get 2 scores, simple and accurate. You wonder where the difference is?
Ultrastar is not interested in pitch hights. As long as it is in the pitch range A-G you get one point.
This makes sense for the game, because otherwise men don't get points for high female voices and women don't get points
for low male voices. Accurate is the real tone specified in the txt. I had txt files where the pitch was in a range not
singable by humans, but you could still reach the 10k points in the game. The accuracy is important here, because from
this MIDI and sheet are created. And you also want to have accurate files


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

If something crashes because of low VRAM then use a smaller Whisper model.
Whisper needs more than 8GB VRAM in the `large` model!

You can also force CPU usage with the extra option `--force_cpu`.

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
