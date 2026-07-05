# Version: 0.1.0.dev1
Date: 2026.07.05
- Changes since 0.0.13dev16 (fork):

  **Chart quality & scoring**
  - ptAKF chart refit (on by default): note boundaries and pitches are rebuilt from the same pitch detector the karaoke games use for scoring — real in-game Medium scores rose from ~73% to ~92-98% on test material
  - Exact-tone segmentation: runs and ornaments become visible note staircases instead of staying flat on one pitch
  - Refit fill (on by default): sung-but-uncharted regions (ad-libs, vocalises) get placeholder notes
  - Unhittable notes (no voiced frame in their window) are marked freestyle instead of counting as guaranteed misses
  - Real game-score report (ptAKF vs. the extracted vocals) for Easy/Medium/Hard after every conversion — in the console, the info file and the GUI queue tooltip; the internal simple/accurate estimate is now fallback-only
  - Syllables are distributed across split notes (nicer karaoke text); golden note generation (`--golden_notes`, opt-in)
  - Weighted-median pitch, octave-error correction passes, vocal-centre safety net, improved BPM detection and linebreaks, onset-based timing correction, `--bpm` and `--octave` overrides, key quantization fixes

  **Pitch & separation pipeline**
  - FCPE added as alternative pitch backend (`--pitcher fcpe`); SwiftF0 remains the default
  - Mel-Band-Roformer is the default separation model (better real-world pipeline results); deterministic separation
  - Separate audio paths for transcription vs. pitch detection; gentler denoising; pitch-frame reuse makes the refinement/scoring phase ~27x faster

  **Lyrics**
  - Reference-lyrics-first pipeline: synced/plain LRCLIB lyrics with local forced alignment take priority over transcription
  - Remote speech-to-text (`--remote_stt`, experimental): an OpenAI-compatible cloud endpoint supplies the text for users without a capable GPU — timing always stays local; with rate-limit retry and a filterable model dropdown in the GUI
  - LLM lyric correction (`--llm_correct`, experimental) with configurable providers; language misdetection on long instrumental intros fixed (VAD-filtered detection), language-aware alignment-model selection
  - Syllable-level note splitting and vocal gap fill (experimental); freestyle detection for unpitchable passages (growls, rap, spoken word)

  **GUI**
  - Full desktop GUI: embedded video browser with quality badge and cookie capture, conversion queue with per-song settings/re-queue/clone, real-time log with stage detection, unified settings page with per-option tooltips, LLM provider manager, yt-dlp one-click updater
  - Full-quality video downloads restored via the bgutil PO-token provider (auto-built by the installer, auto-started by the GUI); provider startup is antivirus-tolerant and logs to .potoken/provider.log

  **Installer & environment**
  - Hardware-aware entry points `install/auto_install.bat|.sh`: GPU auto-detection with CUDA/CPU pick, VRAM-tier advice (CLI flags and GUI paths), ffmpeg check with automatic install on Windows
  - Portable Python fallback: without a system Python 3.12/3.13, uv downloads a self-contained interpreter (app-local store if the global one is broken)
  - Corporate-proxy support: standard proxy variables everywhere, loopback bypass for the local provider, OS certificate store for TLS-inspecting proxies (runtime via truststore, installer via UV_SYSTEM_CERTS auto-enable), GUI proxy settings
  - Persistent CPU/CUDA PyTorch index configuration; hardened install scripts

  **Tools**
  - Library triage tool (experimental, see docs/library-triage.md): moves corrupt or badly-scoring songs out of a library, dry-run by default
  - Library rescore tool (find the weakest generated charts), regression benchmark (real game scores vs. a baseline), UltraStar compatibility fixer, format-aware chart comparison tool

  **Infrastructure & fixes**
  - CI: full pytest suite on every push/PR, with GUI tests genuinely executed (Qt runtime + ffmpeg on the runner); 850+ tests
  - MusicBrainz cover-art embedding compatible with strict players (size-capped, zero padding); routine cover-art 404s are quiet
  - Metadata tags (ID3/Vorbis) for output audio; language hint from video metadata; many CLI/GUI consistency fixes and documentation overhauls

# Version: 0.0.13dev16
Date: 2026.03.04
- Changes:
  - Download Cover from MusicBrainz
  - Fix selection of whisper model
  - Added colab notebook
  - Optimized whisper language check with confidence
  - Improved arguments, so you dont need to use true or false
  - Added interactive mode
  - Use user defined ffmpeg path
  - Show GPU Name and VRAM
  - No linebreak before the last 'E' word for UltraStar txt files
  - Download yt video/audio only once
  - Support for video as input
  - Optimise scale detection
  - Added quantization by key
  - Changed installer to uv
  - Drop crepe for SwiftF0
  - upgrade to python 3.12

# Version: 0.0.12
Date: 2024.12.19
- Changes:
  - Reduce Memory usage by clearing cache in whisper
  - Add lyrics to midi file
  - Split word by note changes
  - Upgrade UltraStar Format to Version 1.2.0 (Use of VIDEOURL)
  - Use yt with cookies
  - Some docker container improvements
  - Fix keep-cache option
  - Fix numbers in lyrics and transcribtion
  - Fix model path option
  - Fix error in PDF sheet generation
  - Fix hypen language download
  - Fix install scripts
  - Some bug fixes and improved error handling and logs

# Version: 0.0.11
Date: 2024.07.06
- Changes:
  - Better linebreak calculation
  - Remove cache folder when finished
  - Remove audio from yt video
  - Added install and start scripts
  - Added Docker support
  - Added sheet music generation

# Version: 0.0.10
Date: 2024.05.01
- Fix:
  - remove whitespace from the beginning and end of title and artist
  - image conversion to jpeg for transparent or RGBA
  - index out of range error in when list is empty from musicbrainz

# Version: 0.0.9
Date: 2024.02.06
- Fix:
  - Re-Pitch mode now re-pitch the audio again
  - Re-Pitch mode now show the text and lines in plot

# Version: 0.0.8
Date: 2024.01.03
- Changes:
  - Plot words
- Fix:
  - Missing word lines in plot

# Version: 0.0.7
Date: 2023.12.29
- Changes:
  - Added format version support for 0.3.0, 1.0.0 and 1.1.0

# Version: 0.0.6
Date: 2023.12.28
- Changes:
  - Optimized the removing of silence in transcription data
  - Mute the processing audio in parts where no singing is detected
  - Plot muted audio

# Version: 0.0.5
Date: 2023.12.23
- Changes:
  - Format GENRE string
  - Extract year from date

# Version: 0.0.4
Date: 2023.12.16
- Changes:
  - Optimized the conversion to Mono
  - Removed limitation to mp3 and wav audio formats
  - Added option float32 to demucs