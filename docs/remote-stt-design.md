# Design: Remote (Cloud) Speech-to-Text as a Whisper Alternative

Status: draft / RFC
Author: agent-assisted design pass
Scope: design only — no implementation in this document

## 1. Motivation

UltraSinger's default transcription path runs local Whisper large-v2 via
WhisperX (`transcribe_with_whisper()` in
`src/modules/Speech_Recognition/Whisper.py:141`). Large-v2 is accurate but
heavy: on CPU-only machines it is slow (multi-minute transcription per song),
and it is the single biggest reason the tool is unpleasant to use without a
CUDA GPU (`settings.pytorch_device` / `--force_cpu`,
`src/Settings.py:90-91`).

Users without a capable GPU currently have no good option: they either wait a
long time for CPU Whisper, or drop to a smaller/faster Whisper model with
worse transcription quality. The goal of this feature is to let those users
send the **audio only** to an external, OpenAI-API-compatible
speech-to-text (STT) service (e.g. Groq's Whisper endpoint) to get the
**text** back quickly, while keeping all timing/alignment work local.

This mirrors a pattern UltraSinger already ships for a different subsystem:
`--llm_correct` sends lyrics **text** to an external OpenAI-compatible chat
endpoint for correction (`src/modules/Speech_Recognition/llm_corrector.py`,
wired at `src/UltraSinger.py:1410-1430`). Remote STT is the same idea applied
one stage earlier in the pipeline (audio -> text instead of text -> text),
and should reuse as much of that provider infrastructure as possible.

### Non-Goals

- **This is not a replacement for local alignment.** Word/line timing must
  keep coming from the local wav2vec2 CTC forced-alignment model
  (`whisperx.load_align_model` / `whisperx.align`, see
  `src/modules/Speech_Recognition/Whisper.py:228-254` and
  `src/modules/Speech_Recognition/reference_lyrics_aligner.py:176-215`).
  Remote STT only ever supplies **text**, never timestamps that get trusted
  directly.
- Not a replacement for the Reference-Lyrics-First (LRCLIB) pipeline — it is
  a further fallback *below* it, not a competitor.
- Not a change to the `--llm_correct` lyric-correction feature, though the
  two will likely end up sharing provider-config plumbing.
- Not a guarantee of feature parity with local Whisper for every language;
  see Open Questions.

## 2. Key Prior-Art Finding: Remote Whisper Alone Gives Bad Timing

During the Groq API evaluation (Quality Phase 5.5, recorded in project
memory `groq-evaluation.md`), UltraSinger already tested sending audio
directly to Groq's remote Whisper endpoint and using its returned segment
timestamps as note timing. Result:

- Lyrics recall: WhisperX (local) won 4 of 6 test songs.
- **Timing accuracy: local WhisperX was ~3x better** — 85% of words within
  ±100 ms of reference vs. only 28% for Groq's remote timestamps.
- Root cause: WhisperX's timing quality comes from its local **wav2vec2
  forced-alignment** pass (a dedicated CTC alignment model, not Whisper's own
  attention-based cross-attention timestamps). Groq (and OpenAI-compatible
  STT APIs generally) return only Whisper's native attention-head timestamps,
  which were never as precise, and Groq's implementation does not expose or
  run a forced-alignment step at all.

**Design consequence:** this feature must **never** trust timestamps that
come back from a remote STT provider. Remote STT is used exclusively as a
**text source**. Timing is always produced afterward by the existing local
CTC forced-alignment model (WAV2VEC2_ASR_BASE_960H, 95 M parameters, CPU-
capable — already the tool's default alignment model per the A/B-test
recorded in project memory). This is exactly the mechanism the
Reference-Lyrics-First pipeline already uses today when it takes LRCLIB
plain-text lyrics (no timestamps) and aligns them to audio via
`create_midi_segments_from_plain_lyrics()` /
`align_lyrics_to_audio()` (`reference_lyrics_aligner.py:899-996` and
`:151-271`). Remote STT text should be piped into that exact same function,
not a new alignment code path.

## 3. Architecture: Remote STT as a Third Lyrics-Source Fallback

Today the lyrics-source waterfall in `RunUltraSinger()` is:

1. LRCLIB **synced** lyrics -> `create_midi_segments_from_reference_lyrics()`
   (`src/UltraSinger.py:324-362`)
2. LRCLIB **plain** lyrics (no sync) -> `create_midi_segments_from_plain_lyrics()`
   (`src/UltraSinger.py:365-402`)
3. Local Whisper full transcription -> `TranscribeAudio()` /
   `transcribe_with_whisper()` (`src/UltraSinger.py:404-459`, `:1373-1432`)

Proposed new waterfall, inserting remote STT as step 3 (pushing local
Whisper to step 4, the last-resort default):

1. LRCLIB synced lyrics (unchanged)
2. LRCLIB plain lyrics (unchanged)
3. **Remote STT transcription** (new) — audio uploaded to the configured
   OpenAI-compatible `/audio/transcriptions` endpoint, returned text treated
   exactly like LRCLIB plain lyrics and passed through
   `create_midi_segments_from_plain_lyrics()` for local CTC alignment +
   pitch assignment.
4. Local Whisper large-v2 (today's default `TranscribeAudio()` path) —
   always available as the final fallback, and remains the default when
   remote STT is not configured/enabled.

This ordering only activates remote STT for users who opted in (see
Settings below) **and** who lack synced/plain LRCLIB lyrics — i.e. it is
additive, not a replacement for the higher-confidence sources that already
exist. It also fires before local Whisper specifically so that users who
enabled it *because* they have no usable GPU never have to run local
Whisper at all.

Language handling follows the existing pattern used for the early-LRCLIB
path (`src/UltraSinger.py:262-269`): if `settings.language` is not
explicit, run the existing fast local language-detection
(`detect_language_from_audio()`, Whisper-tiny, `Whisper.py:50-90`) before
alignment, since the local CTC aligner needs a language code and remote STT
responses are not guaranteed to include a reliable one from every provider.

Failure handling: any exception from the remote call (network error,
auth failure, quota, unsupported language) falls through to step 4 (local
Whisper) exactly like the existing `try/except` fallbacks around the
reference-first pipeline do (`src/UltraSinger.py:360-362`, `:400-402`) —
fail-open, never a hard error.

## 4. Provider Abstraction

### 4.1 Wire protocol

Reuse the OpenAI-compatible `/audio/transcriptions` REST endpoint
(multipart/form-data): `file`, `model`, optional `language`,
`response_format` (`json` or `verbose_json` — request `json` since segment
timestamps are discarded anyway per §2), and `temperature=0` for
determinism. This is the same shape Groq, OpenAI, and most self-hosted
Whisper-compatible servers (e.g. faster-whisper-server) already implement,
so a single client covers multiple providers without provider-specific
branches beyond `api_base_url` + `model`.

### 4.2 Reuse of `LLMProvider`

The GUI already has a provider-config abstraction built for `--llm_correct`:
`LLMProvider` (`src/gui/models.py:9-42`) with fields `name`,
`api_base_url`, `default_model`, `is_default`, `id`, serialized via
`to_dict()`/`from_dict()`, API key deliberately kept **out** of the
dataclass and stored in the system keyring under
`llm_api_key_{id}` (loaded/saved in `preferences_tab.py:203-206, 286-289`,
via `src/gui/secrets.py`).

Proposal: extend this same structure rather than fork a parallel one, since
an STT provider and an LLM-correction provider are the same shape (base URL
+ model + keyring-backed key). Two options, in order of preference:

- **Option A (preferred): generalize `LLMProvider` fields.** Add an
  optional `default_stt_model: str = ""` field (mirroring
  `default_model`, which stays for chat/completions use). A provider that
  only fills in one of the two model fields is only offered for that
  purpose in the relevant dropdowns. Keyring key stays
  `llm_api_key_{id}` (shared secret — Groq's key works for both its chat
  and transcription endpoints).
- **Option B: separate `STTProvider` dataclass**, structurally identical to
  `LLMProvider`, with its own keyring namespace (`stt_api_key_{id}`). Cleaner
  separation if STT and chat providers diverge later (e.g. different auth
  schemes), at the cost of duplicated list-management UI code
  (`LLMProviderListWidget` in `src/gui/widgets.py` would need an `STT`
  counterpart or a generalization).

Recommendation: start with Option A; it is less code and the two concerns
(remote chat vs. remote transcription) are both "call an OpenAI-compatible
endpoint with a base URL, model name, and bearer token," so one list of
providers with two optional model fields is simpler for users to manage
than two separate provider lists in Preferences.

### 4.3 Reference provider: Groq

Groq is the reference/default provider, consistent with the rest of the
codebase's Groq-first stance (LLM correction benchmarks, memory notes).
Model: `whisper-large-v3` (Groq's fastest accurate STT model,
OpenAI-endpoint-compatible, already free-tier at time of writing per the
existing LLM Groq free-plan notes in project memory — pricing/limits should
be re-verified at implementation time since they change).

### 4.4 Upload size / runtime limits

Must be surfaced to the user and enforced client-side before upload, not
discovered via API error:

- Groq's `/audio/transcriptions`: 25 MB file size limit on the free tier
  (larger on paid tiers), ~25 MB roughly corresponds to ~20-25 minutes of
  16 kHz mono audio depending on format — most songs fit, but pre-flight
  size check is required regardless of provider since limits vary.
  OpenAI's own Whisper endpoint has the same 25 MB cap.
  These numbers should be re-verified against current provider docs at
  implementation time (both have changed limits before).
- Recommended client behavior: encode/downsample the uploaded audio to a
  compact mono format (e.g. 16 kHz mono OGG/Opus or MP3) before upload —
  UltraSinger already produces a mono `whisper_audio_path` for this exact
  purpose (`src/UltraSinger.py:204-206`), so no new audio processing is
  needed, only possibly re-encoding to a smaller container for transfer.
  If the encoded file still exceeds the provider's limit, fail open to
  local Whisper (§3) rather than attempting client-side chunking in v1
  (see Open Questions on chunking).
- Request timeout: set generously (e.g. 120s) since upload + remote
  inference of a multi-minute song is not instant; on timeout, fail open.

## 5. Settings / CLI / GUI Sketch

### 5.1 Settings fields (`src/Settings.py`)

Mirroring the existing `llm_*` block (`Settings.py:103-109`):

```python
remote_stt_enabled = False           # Enable remote STT fallback
remote_stt_api_base_url: str = "https://api.groq.com/openai/v1"
remote_stt_api_key: str | None = None   # or REMOTE_STT_API_KEY env var
remote_stt_model: str = "whisper-large-v3"
remote_stt_timeout: int = 120         # seconds
```

### 5.2 CLI flags (`src/UltraSinger.py`, alongside the `--llm_*` block at
`:1945-1950` and the getopt long-options list at `:2058-2060`)

```
--remote_stt                 Enable remote STT fallback (text-only; timing
                              always computed locally)
--remote_stt_api_base_url=X  OpenAI-compatible base URL (default: Groq)
--remote_stt_api_key=X       API key (or REMOTE_STT_API_KEY env var)
--remote_stt_model=X         Model name (default: whisper-large-v3)
```

Follows the existing `elif opt in (...)` dispatch pattern used for
`--llm_correct` / `--llm_api_base_url` / `--llm_api_key`
(`UltraSinger.py:1945-1950`) and should be added to the same settings-info
dump block that already documents LLM correction
(`UltraSinger.py:960-962`) for parity — a `Remote STT:` section listing
enabled state, base URL, and model, but never the key.

### 5.3 Implementation touch points

- New module `src/modules/Speech_Recognition/remote_stt.py` (analogous to
  `llm_corrector.py`): a `transcribe_remote(audio_path, config) -> str`
  function that does the multipart upload and returns plain text (or raises
  on failure — caller decides fail-open behavior, consistent with how
  `correct_lyrics_with_llm` is called inside a `try/except` at
  `UltraSinger.py:1410-1430`).
- Wiring point: a new block in `RunUltraSinger()` between the existing
  plain-lyrics block (ends `UltraSinger.py:402`) and the
  `if not reference_first_used:` local-Whisper fallback
  (`UltraSinger.py:404`), which calls `transcribe_remote()` and, on
  success, feeds the returned text into
  `create_midi_segments_from_plain_lyrics()` exactly like
  `process_data.plain_lyrics` does today.

### 5.4 GUI placement

Preferences tab already has an "LLM Providers" section
(`src/gui/preferences_tab.py`, `LLMProviderListWidget`) for managing
`LLMProvider` entries and showing keyring status
(`preferences_tab.py:90-106`). With Option A (§4.2), this section is
renamed/relabeled generically (e.g. "AI Providers") and the per-provider
edit form grows an optional "Transcription model" field next to the
existing "Chat model" field — same list, same keyring plumbing, one new
column. In `settings_tab.py`, add a checkbox "Use remote transcription
when no GPU / as fallback" next to the existing LLM-correction toggle,
with a provider + model picker sourced from the same provider list.

### 5.5 Fail-open behavior (summary)

Remote STT failure (network, auth, quota, timeout, oversized file,
unsupported response) must never abort the run. It always falls through
to local Whisper, printing a clear log line (`ULTRASINGER_HEAD`-prefixed,
consistent with existing fallback messages like
`UltraSinger.py:360-362`) so the user understands why local Whisper ran
despite having remote STT configured.

## 6. Open Questions for the Maintainer

1. **Languages**: Whisper-based remote endpoints (Groq, OpenAI) support the
   same language set as local Whisper large-v3, but quality per language
   is unverified for remote — should remote STT be restricted to the
   `CORE_LANGUAGES` set (`language_constants.py`) the same way low-
   confidence language detection already falls back to English
   (`Whisper.py:80-88`), or offered unconditionally?
2. **Cost/privacy disclosure in the UI**: should enabling remote STT show
   an explicit one-time warning ("your song's audio will be uploaded to
   \<provider\>") the way there should probably be one for `--llm_correct`
   sending lyric text off-device? Should the settings-info file
   (`UltraSinger.py:960` area) record that a given run used remote STT, for
   user auditability?
3. **Chunking long songs**: v1 proposes fail-open to local Whisper when a
   song exceeds the provider's upload size limit (§4.4) instead of
   client-side chunking + reassembly. Is that acceptable, or is chunking
   (e.g. split at silence, transcribe segments, concatenate text in order)
   a v1 requirement given how common >20 min recordings (live albums,
   medleys) might be for this tool's users?
4. **Third-party data handling**: audio leaves the user's machine. Do we
   need an explicit README/GUI disclosure of which provider(s) are
   supported and pointers to their data-retention policies, similar to
   how `--llm_correct` already documents its provider recommendations in
   the experimental-features docs?
5. **Provider-list unification (Option A vs B, §4.2)**: does the
   maintainer want STT and LLM-correction providers to genuinely share one
   list (simpler UI, implies same key works for both) or be kept separate
   in case some users want a Groq key for STT only, or an OpenAI key for
   chat only, without cross-wiring?

## 7. Effort Estimate (person-hours)

| Work package | Estimate |
|---|---|
| `remote_stt.py` module (multipart upload, response parsing, size/timeout guards, unit tests) | 6-8 h |
| `Settings.py` fields + CLI flags + getopt wiring + settings-info dump | 2-3 h |
| Pipeline wiring in `RunUltraSinger()` (fallback ordering, language handling, fail-open, `transcribed_data` rebuild akin to `UltraSinger.py:350-359`) | 3-4 h |
| `LLMProvider` extension (Option A) + keyring reuse verification | 2-3 h |
| GUI: Preferences provider-form field, Settings tab toggle + provider/model picker, `ultrasinger_runner.py` flag pass-through | 4-6 h |
| Tests (unit: remote_stt module with mocked HTTP; integration: fallback ordering with monkeypatched provider) | 4-6 h |
| Docs (experimental-features README section, CLI help text in `common_print.py`) | 1-2 h |
| **Total** | **22-32 h** |

This excludes time spent resolving the Open Questions in §6, which may
change scope (e.g. chunking support would add roughly 6-10 h on its own).
