# TTS-UE Pipeline

End-to-end pipeline for generating AI voice audio and rendering it as a lip-synced MetaHuman video in Unreal Engine.

**Flow:** Text → OpenAI analysis → ChatterBox TTS → WAV + metadata → UE Remote Control → Georgy Dev lip sync → MP4

---

## Overview

| Stage | Script | What it does |
|-------|--------|--------------|
| TTS generation | `chatterbox_ue_pipeline.py` | Detects language, analyzes text with GPT-4o, generates WAV with emotion-tuned parameters, writes UE metadata JSON |
| Analysis | `openai_analyzer.py` | Classifies gender/emotion, expands abbreviations, normalizes numbers/dates, adds prosody punctuation |
| Orchestration | `generate_and_render_ue.py` | Runs TTS → copies WAV to `input/` → launches UE → copies rendered MP4 to `output/` |
| UE launch | `launch_ue_remote.py` | Starts UnrealEditor.exe with Remote Control Web Server, polls until ready, sends Python script via HTTP PUT |
| UE rendering | `ue_render_script.py` | Runs inside UE: updates Georgy Dev AnimBP file path, waits 3s for lip sync processing, adjusts timeline to audio duration, triggers Movie Render Queue, quits when done |

---

## Requirements

**Python (TTS side)**
- Python 3.10+
- PyTorch 2.6.0 + torchaudio 2.6.0 (versions must match exactly)
- See `requirements.txt` for full list

**Unreal Engine (render side)**
- Unreal Engine 5.6
- Plugins enabled: **Remote Control Web Interface**, **Python Script Plugin**, **Georgy Dev Lip Sync**
- MetaHuman actor configured in a Level Sequence
- Saved Movie Pipeline preset at `/Game/Cinematics/Pending_MoviePipelinePrimaryConfig`
- Georgy Dev AnimBP with a promoted `File_Path` string variable connected to `Import Audio from File`

---

## Setup

**1. Install Python dependencies**

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install PyTorch (choose one):
# CUDA (NVIDIA GPU):
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# CPU only:
pip install torch==2.6.0 torchaudio==2.6.0

pip install -r requirements.txt
```

**2. Set OpenAI API key**

```bash
export OPENAI_API_KEY="sk-..."    # Windows: set OPENAI_API_KEY=sk-...
```

Or create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

**3. Configure paths**

Edit `launch_ue_remote.py` → `load_config()` to point at your machine:

```python
config = {
    'ue_editor':       r'C:\...\UnrealEditor.exe',
    'ue_project':      r'C:\...\MyProject.uproject',
    'ue_python_script': 'ue_render_script.py',
    'remote_control_port': 30010,
}
```

Edit `ue_render_script.py` constants to match your UE project:

```python
PRESET_PATH          = "/Game/Cinematics/Pending_MoviePipelinePrimaryConfig..."
LEVEL_SEQUENCE_PATH  = "/Game/NewLevelSequence"
MAP_PATH             = "/Game/NewMap"
INPUT_AUDIO_FOLDER   = "C:/...path.../tts-ue-pipeline/output"
RENDER_OUTPUT_FOLDER = "C:/...UEProject.../Saved/MovieRenders"
LIPSYNC_ANIMBP_PATH  = "/RuntimeMetaHumanLipSync/LipSyncData/MyLipSync_Face_AnimBP1..."
```

**4. Configure `config.yaml`** (optional — used by monitoring/future automation)

Fill in `paths.unreal_project`, `paths.unreal_editor`, and `unreal.sequence_path` as prompted by the TODO comments in the file.

---

## Usage

### Generate TTS only

```bash
# Auto-detect gender/emotion with OpenAI
python chatterbox_ue_pipeline.py --text "Hello, welcome to the demo." --character npc01

# Force gender and emotion
python chatterbox_ue_pipeline.py --text "I will not allow it!" --character villain \
    --gender male --emotion angry

# From a text file, skip OpenAI analysis
python chatterbox_ue_pipeline.py --text-file dialogue.txt --character hero --no-openai

# Specify language explicitly
python chatterbox_ue_pipeline.py --text "你好世界" --language zh --character npc_zh

# With voice cloning (reference WAV, 5+ seconds)
python chatterbox_ue_pipeline.py --text "Clone this voice." --voice clone_voice/ref.wav \
    --character custom
```

Output goes to `output/` by default:
```
output/
├── npc01_male_neutral_a3f8e1b2.wav
└── npc01_male_neutral_a3f8e1b2_metadata.json
```

### Full pipeline (TTS → UE render → MP4)

```bash
# Generate English audio, render in UE
python generate_and_render_ue.py --language en --emotion serious

# Generate all 5 benchmark languages
python generate_and_render_ue.py --all --emotion happy

# Copy an existing WAV and render (skip TTS generation)
python generate_and_render_ue.py --copy-only --audio output/my_file.wav

# Generate TTS but skip UE launch
python generate_and_render_ue.py --language en --emotion calm --no-launch
```

### Launch UE rendering manually

```bash
python launch_ue_remote.py

# Override paths at runtime
python launch_ue_remote.py \
    --ue-editor "C:/Epic/UE_5.6/Engine/Binaries/Win64/UnrealEditor.exe" \
    --ue-project "C:/Projects/MyProject/MyProject.uproject" \
    --port 30010
```

---

## TTS Options Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--text TEXT` | — | Input text |
| `--text-file FILE` | — | Path to text file (alternative to `--text`) |
| `--character NAME` | `character` | Character ID used in output filename |
| `--output-dir PATH` | `output` | Output directory |
| `--gender {male,female}` | auto | Override gender detection |
| `--emotion EMOTION` | auto | Override emotion detection (see list below) |
| `--language CODE` | auto | Language code (e.g. `en`, `zh`, `ja`) |
| `--voice PATH` | — | Reference WAV for voice cloning |
| `--device {cuda,mps,cpu}` | auto | Computing device |
| `--no-openai` | off | Disable OpenAI analysis, use defaults |
| `--id CUSTOM_ID` | auto-UUID | Custom ID for output filename |

### Supported emotions

`neutral` · `calm` · `sad` · `serious` · `confident` · `happy` · `excited` · `angry` · `fear` · `nervous` · `surprise` · `disgust`

### Supported languages

23 languages via auto-detection: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish

---

## Output Files

**WAV** — 24 kHz, mono, 16-bit PCM

**Metadata JSON** — UE asset info:
```json
{
  "version": "1.0",
  "character_name": "npc01",
  "voice_gender": "male",
  "emotion": "serious",
  "language": "en",
  "unreal_plugin_config": {
    "plugin_id": "UE_TTS_Male_Neutral",
    "voice_profile": "calm_male_01",
    "emotion_blend": {"serious": 1.0}
  },
  "audio_info": {
    "sample_rate": 24000,
    "channels": 1,
    "duration_seconds": 3.12
  },
  "emotion_parameters": {
    "exaggeration": 0.35,
    "temperature": 0.6,
    "cfg_weight": 0.7
  },
  "text_data": {
    "original_text": "Hello, welcome to the demo.",
    "spoken_text": "Hello, welcome to the demo."
  }
}
```

---

## UE Rendering Details

The render pipeline inside UE (`ue_render_script.py`) does the following:

1. Reads the newest WAV from `INPUT_AUDIO_FOLDER` (or `current_audio.txt` for exact path)
2. Sets the `File_Path` variable on the Georgy Dev Lip Sync AnimBP Class Default Object
3. Waits 3 seconds for the plugin to analyze the audio
4. Adjusts the Level Sequence playback range to match the audio duration
5. Creates a Movie Pipeline Queue job with the saved preset (4K @ 30fps)
6. Monitors render status via a slate tick callback; quits UE when done

The rendered MP4 is then copied to `output/` and renamed to match the source WAV filename.

---

## Directory Structure

```
tts-ue-pipeline/
├── chatterbox_ue_pipeline.py     # Main TTS pipeline
├── openai_analyzer.py            # GPT-4o text analysis
├── emotion_config.py             # 12-emotion parameter definitions
├── ue_metadata.py                # UE metadata JSON generator
├── generate_and_render_ue.py     # End-to-end orchestrator
├── launch_ue_remote.py           # UE Remote Control launcher
├── ue_render_script.py           # UE-side Python render script
├── test_multilingual_benchmark.py# 5-language benchmark test
├── config.yaml                   # Pipeline configuration
├── config/
│   └── language_config.py        # Language mappings
├── clone_voice/                  # Place reference WAV files here
├── input/                        # Staging folder for UE audio input
├── output/                       # Generated WAV + JSON + MP4
├── ue_preset/                    # UE rendering preset storage
└── logs/                         # Pipeline logs
```

---

## Troubleshooting

**Remote Control API not ready**
- Make sure the Remote Control Web Interface plugin is enabled in UE
- Default ports: WebApp on 30000, API on 30010

**`File_Path` variable not found in AnimBP**
- Open the AnimBP in UE, find the variable in the Event Graph, and make sure it is promoted (has a blue pin) and named exactly `File_Path`

**Lip sync not animating**
- Confirm the `File_Path` variable is wired to the `Import Audio from File` node in the AnimBP's EventGraph
- The 3-second pre-render delay can be increased in `ue_render_script.py` if needed

**PyTorch version mismatch**
- `torch` and `torchaudio` must be the same version (2.6.0). Mixing versions causes silent errors.

**Apple Silicon (Mac) — TTS generation only**
- The pipeline auto-selects MPS on Apple Silicon. UE rendering requires Windows.
