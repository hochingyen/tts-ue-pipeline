# Complete TTS → UE → MP4 Pipeline

Full automated workflow from text input to rendered video in Unreal Engine.

## Quick Start

### Interactive Mode (Recommended)

```bash
python generate_and_render_ue.py --use-openai-prosody
```

You will:
1. **Select language** from 23 options (menu)
2. **Type your text** (multi-line supported)
3. Wait for TTS generation
4. Wait for UE to launch and render
5. Get your MP4 in the `output/` folder

### Direct Language Mode

```bash
# Skip menu, specify language directly
python generate_and_render_ue.py --language ar --use-openai-prosody
```

## Pipeline Steps

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Language Selection (23 languages)                        │
│    ├─ Interactive menu OR --language flag                   │
│    └─ Arabic, Chinese, English, Spanish, etc.               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Text Input                                               │
│    ├─ Keyboard: Type/paste your text (Ctrl+D when done)     │
│    └─ JSON: --text-input-json example_texts.json            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. TTS Generation (ChatterBox)                             │
│    ├─ GPU: CUDA device 1 (default, use --gpu 0 to change)  │
│    ├─ OpenAI: Prosody optimization (--use-openai-prosody)  │
│    ├─ Emotion: Auto-detected or specify with --emotion     │
│    └─ Output: output/cpu_gpu_{lang}_{emotion}.wav          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Prepare for UE                                          │
│    ├─ Write path to output/current_audio.txt               │
│    └─ Copy WAV to input/ folder                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Launch Unreal Engine                                     │
│    ├─ Automated via launch_ue_remote.py                     │
│    ├─ Loads audio from output/current_audio.txt             │
│    └─ Renders video with synchronized lip-sync              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Wait for UE to Close                                     │
│    └─ Rendering completes when you close UE                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Copy MP4 to Output                                       │
│    ├─ From: UE render folder                                │
│    ├─ To: output/cpu_gpu_{lang}_{emotion}.mp4               │
│    └─ Final video ready!                                    │
└─────────────────────────────────────────────────────────────┘
```

## Supported Languages (23)

Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish

## Command-Line Options

### Basic Usage

```bash
# Interactive (menu + keyboard input)
python generate_and_render_ue.py --use-openai-prosody

# Specify language
python generate_and_render_ue.py --language ja --use-openai-prosody

# Use JSON file
python generate_and_render_ue.py --language en --text-input-json my_texts.json --use-openai-prosody
```

### Advanced Options

```bash
# Custom emotion
python generate_and_render_ue.py --language en --emotion happy --use-openai-prosody

# Different GPU
python generate_and_render_ue.py --language zh --use-openai-prosody --gpu 0

# Multiple languages (batch processing)
python generate_and_render_ue.py --all --text-input-json example_texts.json --use-openai-prosody

# Test TTS only (skip UE)
python generate_and_render_ue.py --language ar --use-openai-prosody --no-launch

# Render existing audio
python generate_and_render_ue.py --copy-only --audio output/cpu_gpu_ar_happy.wav
```

## JSON File Format

Create a JSON file with your texts:

```json
{
  "en": {
    "name": "English",
    "text": "Hello! This is a test of the multilingual TTS system."
  },
  "ar": {
    "name": "Arabic",
    "text": "مرحبا! هذا اختبار لنظام تحويل النص إلى كلام."
  },
  "zh": {
    "name": "Chinese",
    "text": "你好！这是多语言TTS系统的测试。"
  }
}
```

Then run:
```bash
python generate_and_render_ue.py --language ar --text-input-json my_texts.json --use-openai-prosody
```

## Keyboard Input Tips

When entering text interactively:

1. **Type or paste** your text
2. **Press Enter** for new lines (multi-line supported)
3. **Press Ctrl+D** (Mac/Linux) or **Ctrl+Z then Enter** (Windows) when done

Example:
```
Enter your text below (press Ctrl+D when done):
--------------------------------------------------------------------------------
The quick brown fox jumps over the lazy dog.
This is a multi-line example.
Press Ctrl+D when you're finished typing.
^D
```

## Output Files

All files are saved to the `output/` folder:

- **WAV**: `cpu_gpu_{language}_{emotion}.wav` - Generated TTS audio
- **MP4**: `cpu_gpu_{language}_{emotion}.mp4` - Rendered video from UE
- **JSON**: `cpu_gpu_{language}_{emotion}.json` - Metadata

## Troubleshooting

### TTS Generation Issues

```bash
# Test TTS only (skip UE)
python generate_and_render_ue.py --language en --use-openai-prosody --no-launch
```

### GPU Memory Issues

```bash
# Use different GPU
python generate_and_render_ue.py --language en --use-openai-prosody --gpu 0
```

### UE Not Launching

Check that:
1. `launch_ue_remote.py` exists
2. UE Remote Control is enabled (ports 30000/30010)
3. You're running on Windows or have network access to Windows machine

## Examples

### Quick Test
```bash
python generate_and_render_ue.py --use-openai-prosody
# Select "1" for Arabic
# Type: مرحبا! كيف حالك؟
# Press Ctrl+D
# Wait for UE to render
# Check output/cpu_gpu_ar_auto.mp4
```

### Batch Processing
```bash
# Create example_texts.json with 5 languages
python generate_and_render_ue.py --all --text-input-json example_texts.json --use-openai-prosody
# Processes all 5 languages through UE sequentially
```

### Production Workflow
```bash
# 1. Create your JSON file with all dialogue
vim project_dialogue.json

# 2. Run pipeline for all languages
python generate_and_render_ue.py --all --text-input-json project_dialogue.json --use-openai-prosody

# 3. All MP4s will be in output/ folder
ls output/*.mp4
```
