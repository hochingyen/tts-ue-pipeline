# CLAUDE.md — TTS-UE Pipeline: Codebase Understanding

## Overview

This project is an end-to-end pipeline that:
1. Generates AI voice audio (TTS) from text using ChatterBox
2. Passes the audio into Unreal Engine 5.6
3. Renders a lip-synced MetaHuman character video (MP4)

The pipeline runs across two machines: **Mac** (for TTS generation) and **Windows** (for UE rendering).

---

## File Map

```
tts-ue-pipeline/
├── generate_and_render_ue.py         # Top-level orchestrator — run this
├── chatterbox_ue_pipeline.py         # Core TTS class (ChatterBoxUEPipeline)
├── openai_analyzer.py                # GPT-4o: gender/emotion detection, prosody, translation
├── emotion_config.py                 # 12 emotions → TTS param mappings
├── ue_metadata.py                    # Generates JSON metadata alongside WAV
├── launch_ue_remote.py               # Launches UE, waits for Remote Control API, sends Python
├── ue_render_script.py               # Runs INSIDE UE — automates Movie Render Queue
├── test_multilingual_benchmark.py    # Benchmark: generates WAVs for 5 languages
├── config/language_config.py         # 23-language mappings + default voice prompt URLs
├── emotion_config.py                 # Exaggeration/temp/cfg_weight per emotion
├── config.yaml                       # Config template (paths mostly TODO)
├── requirements.txt                  # Python deps (PyTorch 2.6.0 required exact)
├── input/                            # Staging folder: WAV copied here before UE launch
├── output/                           # TTS output: WAV + metadata JSON + final MP4
├── clone_voice/                      # Reference WAVs for voice cloning (5+ sec)
├── logs/                             # Pipeline logs
└── ue_preset/                        # UE rendering preset storage
```

---

## Pipeline Stages

### Stage 1 — TTS Generation (`chatterbox_ue_pipeline.py`)

**Class**: `ChatterBoxUEPipeline`

Steps performed:
1. **Device detection**: picks MPS (Apple Silicon) → CUDA → CPU
2. **Language detection**: `langdetect` auto-detects language from input text
3. **OpenAI analysis** (`openai_analyzer.py`): GPT-4o detects gender, emotion, rewrites text for prosody
4. **TTS synthesis**: ChatterBox with emotion params from `emotion_config.py`
5. **Save WAV**: 24 kHz, mono, 16-bit PCM
6. **Generate filename**: `{character}_{gender}_{emotion}_{8-char-uuid}.wav`
7. **Generate metadata JSON**: full voice asset info including emotion params, audio info, UE plugin config

**12 supported emotions**: neutral, calm, sad, serious, confident, happy, excited, angry, fear, nervous, surprise, disgust

**Emotion parameters** (from `emotion_config.py`):
- `exaggeration`: 0.25–2.0 (expressiveness)
- `temperature`: 0.05–5.0 (randomness)
- `cfg_weight`: 0.0–1.0 (pace/quality)

**Voice cloning**: optional — pass a 5+ sec reference WAV via `--voice`

---

### Stage 2 — Orchestration (`generate_and_render_ue.py`)

This is the entry point for the full pipeline.

Workflow:
```
generate_tts()
  → calls test_multilingual_benchmark.py
  → returns newest WAV in output/

copy_to_ue_input()
  → copies WAV to input/ folder
  → writes absolute path to output/current_audio.txt

launch_ue_remote.py (subprocess)
  → launches UE, sends ue_render_script.py

copy_mp4_to_output()
  → finds newest MP4 in UE render folder
  → copies to output/ named after the WAV
```

**Key file**: `output/current_audio.txt` — this is how the audio path is communicated from the Mac side to the UE-side script.

**CLI flags**:
- `--language {en,zh,ja,ar,es}` — generate specific language
- `--all` — all 5 benchmark languages
- `--emotion EMOTION` — override emotion (default: auto)
- `--copy-only --audio PATH` — skip TTS, use existing WAV
- `--no-launch` — skip UE rendering

---

### Stage 3 — UE Launch (`launch_ue_remote.py`)

**Class**: `UnrealRemoteControl`

Steps:
1. Validates that UE editor and project paths exist
2. Launches `UnrealEditor.exe` with Remote Control plugin enabled
3. Polls `http://localhost:30000` until the WebApp is ready (timeout: 120 s)
4. Sends `ue_render_script.py` content via HTTP PUT to `http://localhost:30010/remote/object/call`
5. Waits for UE process to exit

**Ports**: 30000 (WebApp health check), 30010 (Remote Control API)

**Hardcoded paths** (need to be updated for each machine):
- UE editor: `C:\Users\marketing\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe`
- UE project: `C:\Users\marketing\Documents\Unreal Projects\male_runtime\MyMHProject.uproject`

---

### Stage 4 — UE Rendering (`ue_render_script.py`)

This script runs **inside Unreal Engine** via the Remote Control API.

Steps:
1. **Read audio path**: checks `--AudioFile` CLI arg → `output/current_audio.txt` → newest WAV in input folder
2. **Clean render folder**: deletes all files in UE's render output directory
3. **Update Lip Sync AnimBP**: loads AnimBP CDO, sets `File_Path` variable to audio file path, saves
4. **Wait 3 seconds**: allows Georgy Dev Lip Sync plugin to process the audio
5. **Adjust Level Sequence timeline**: reads WAV duration, sets playback range to match
6. **Create Movie Pipeline job**: clears queue, allocates new job
7. **Load preset**: `/Game/Cinematics/Pending_MoviePipelinePrimaryConfig`
8. **Start render**: `MoviePipelinePIEExecutor`
9. **Monitor via tick callback** (`check_render_status`): polls `is_rendering()` until complete
10. **Auto-quit UE** when render finishes

**UE asset paths** (hardcoded, project-specific):
- Preset: `/Game/Cinematics/Pending_MoviePipelinePrimaryConfig`
- Level Sequence: `/Game/NewLevelSequence`
- Map: `/Game/NewMap`
- Lip Sync AnimBP: `/RuntimeMetaHumanLipSync/LipSyncData/MyLipSync_Face_AnimBP1`
- Input audio folder: `C:/Users/marketing/Desktop/A2F_cynthia/tts-ue-pipeline/output`
- Render output: `C:/Users/marketing/Documents/Unreal Projects/male_runtime/Saved/MovieRenders`

**AnimBP variable**: must be named exactly `File_Path` and promoted in the EventGraph.

---

## OpenAI Analyzer (`openai_analyzer.py`)

Three capabilities:

1. **English normalization** (no LLM, deterministic regex):
   - Numbers, currency, time, percentages, abbreviations, slang → spoken form

2. **GPT-4o analysis** (`analyze()`):
   - Detects gender and emotion from text
   - Rewrites text with prosody punctuation for natural TTS pacing
   - Returns: `{gender, emotion, spoken_text}`

3. **Translation** (`translate()`):
   - Translates English to target language
   - Normalizes numbers/abbreviations for TTS in target language

---

## Benchmark Test (`test_multilingual_benchmark.py`)

Tests 5 languages: English, Chinese, Japanese, Arabic, Spanish

- Uses voice cloning from `prof_1min.wav`
- Adds 3-second silence buffer at end of each audio
- Output naming: `cpu_gpu_{language}_{emotion}.wav`

---

## Language Support (`config/language_config.py`)

23 languages with:
- ISO 639-1 code → language name mapping
- Default voice prompt URLs (ChatterBox demo FLAC files)
- `langdetect` code → ChatterBox code mapping (e.g., `zh-cn` → `zh`)

---

## Output Files

For each generation:
```
output/
├── {character}_{gender}_{emotion}_{uuid}.wav           # 24 kHz, mono, 16-bit WAV
├── {character}_{gender}_{emotion}_{uuid}_metadata.json  # Full voice asset metadata
└── {character}_{gender}_{emotion}_{uuid}.mp4            # Rendered MetaHuman video
```

---

## Key Constraints & Gotchas

| Constraint | Detail |
|---|---|
| PyTorch version | `torch==2.6.0` + `torchaudio==2.6.0` must match exactly |
| Platform split | TTS runs on Mac (MPS), UE rendering requires Windows |
| AnimBP variable name | Must be exactly `File_Path` (capital F, capital P, underscore) |
| Lip sync delay | 3-second wait before render is mandatory for Georgy Dev plugin |
| Audio format | ChatterBox outputs 24 kHz mono 16-bit PCM WAV |
| Remote Control ports | 30000 (health check), 30010 (Python execution API) |
| current_audio.txt | Written to `output/` by orchestrator; read by `ue_render_script.py` |
| Voice cloning | Reference WAV must be 5+ seconds |

---

## Common Entry Points

```bash
# Full pipeline (TTS → UE render → MP4)
python generate_and_render_ue.py --language en --emotion serious

# TTS only (no UE)
python chatterbox_ue_pipeline.py --text "Hello world" --character npc01

# Use existing audio, skip TTS
python generate_and_render_ue.py --copy-only --audio output/my_audio.wav

# Multilingual benchmark
python test_multilingual_benchmark.py --language zh --emotion happy

# All 5 languages
python generate_and_render_ue.py --all
```
