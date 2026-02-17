#!/usr/bin/env python3
"""
Generate TTS Audio and Prepare for Unreal Engine Rendering

This script:
1. Generates TTS audio using test_multilingual_benchmark.py
2. Copies the audio to UE input folder
3. Launches UE remote rendering (if on Windows or network share available)

Usage:
    # Generate English audio and copy to UE
    python generate_and_render_ue.py --language en --emotion serious

    # Generate all languages
    python generate_and_render_ue.py --all --emotion happy

    # Just copy existing audio (skip generation)
    python generate_and_render_ue.py --copy-only --audio output/cpu_gpu_en_serious.wav
"""

import os
import sys
import argparse
import shutil
import subprocess
import glob
from pathlib import Path


# UE input folder (relative to this project)
UE_INPUT_FOLDER = "input"

# UE launch script (relative to this project)
UE_LAUNCH_SCRIPT = "launch_ue_remote.py"

# Default output directory for TTS generation
DEFAULT_TTS_OUTPUT = "output"

# UE render output folder (must match ue_render_script.py)
RENDER_OUTPUT_FOLDER = r"C:\Users\marketing\Documents\Unreal Projects\male_runtime\Saved\MovieRenders"


def generate_tts(language=None, emotion="auto", use_openai_prosody=True):
    """
    Generate TTS audio using test_multilingual_benchmark.py

    Args:
        language: Language code (en, zh, ja, ar, es) or None for all
        emotion: Emotion for TTS
        use_openai_prosody: Use OpenAI prosody optimization

    Returns:
        Path to generated audio file
    """
    print("\n" + "="*80)
    print("[STEP 1/3] Generating TTS Audio")
    print("="*80)

    cmd = [sys.executable, "test_multilingual_benchmark.py"]

    if language:
        cmd.extend(["--language", language])
    else:
        cmd.append("--all")

    cmd.extend(["--emotion", emotion])

    if use_openai_prosody:
        cmd.append("--use-openai-prosody")

    print(f"Running: {' '.join(cmd)}")

    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', env=env)
        print(result.stdout)

        # Find the generated WAV using glob (emotion may be auto-detected, name differs)
        lang = language if language else "en"
        pattern = os.path.join(DEFAULT_TTS_OUTPUT, f"cpu_gpu_{lang}_*.wav")
        wav_files = glob.glob(pattern)
        if wav_files:
            # Pick the newest one
            wav_files.sort(key=os.path.getmtime, reverse=True)
            audio_file = wav_files[0]
            print(f"\n[SUCCESS] Generated: {audio_file}")
            return audio_file
        else:
            print(f"[ERROR] No WAV found matching: {pattern}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] TTS generation failed: {e}")
        print(e.stderr)
        return None


def copy_to_ue_input(audio_file, ue_input_folder):
    """
    Copy audio file to UE input folder

    Args:
        audio_file: Path to audio file
        ue_input_folder: UE input folder path

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*80)
    print("[STEP 2/3] Copying Audio to UE Input Folder")
    print("="*80)

    if not os.path.exists(audio_file):
        print(f"[ERROR] Audio file not found: {audio_file}")
        return False

    # Check if UE folder is accessible
    if os.path.exists(ue_input_folder):
        try:
            dest_file = os.path.join(ue_input_folder, os.path.basename(audio_file))
            shutil.copy2(audio_file, dest_file)
            print(f"[SUCCESS] Copied to: {dest_file}")

            # Write exact path to current_audio.txt so ue_render_script.py uses the right file
            marker_file = os.path.join(ue_input_folder, "current_audio.txt")
            # Convert to Windows-style path with forward slashes for UE
            ue_path = dest_file.replace("\\", "/")
            with open(marker_file, "w", encoding="utf-8") as f:
                f.write(ue_path)
            print(f"[SUCCESS] Audio path written to: {marker_file}")
            return True
        except Exception as e:
            print(f"[ERROR] Copy failed: {e}")
            return False
    else:
        print(f"[WARNING] UE input folder not accessible: {ue_input_folder}")
        print(f"\nManual copy required:")
        print(f"  Source: {os.path.abspath(audio_file)}")
        print(f"  Destination: {ue_input_folder}")
        return False


def write_current_audio_marker(audio_file, output_dir):
    """
    Write the absolute path of the audio file to output/current_audio.txt
    so ue_render_script.py knows exactly which file to use.
    """
    marker_file = os.path.join(output_dir, "current_audio.txt")
    abs_path = os.path.abspath(audio_file).replace("\\", "/")
    with open(marker_file, "w", encoding="utf-8") as f:
        f.write(abs_path)
    print(f"[SUCCESS] Audio path written to: {marker_file} → {abs_path}")


def run_pipeline_for_audio(audio_file, ue_input_folder, ue_launch_script, no_launch):
    """
    Run the full UE render pipeline for a single audio file:
    write current_audio.txt → copy WAV to input → launch UE → wait for close → copy MP4 to output
    """
    # Write exact audio path to output/current_audio.txt for ue_render_script.py
    write_current_audio_marker(audio_file, DEFAULT_TTS_OUTPUT)

    # Copy WAV to UE input folder (optional, kept for backward compat)
    copy_success = copy_to_ue_input(audio_file, ue_input_folder)
    if not copy_success:
        print(f"[WARNING] Copy to UE input failed for: {audio_file}")

    # Launch UE and wait for render to complete
    if not no_launch:
        launch_success = launch_ue_rendering(ue_launch_script)
        if not launch_success:
            print(f"[WARNING] UE rendering launch failed for: {audio_file}")
        else:
            copy_mp4_to_output(audio_file)
    else:
        print("\n[SKIPPED] UE rendering launch (--no-launch flag)")


def copy_mp4_to_output(audio_file):
    """
    Copy rendered MP4 from UE render folder to output/, named after the WAV file.

    Args:
        audio_file: Path to the source WAV file (used for naming the MP4)
    """
    print("\n" + "="*80)
    print("[STEP 4/4] Copying MP4 to Output Folder")
    print("="*80)

    # Find newest MP4 in UE render folder
    mp4_pattern = os.path.join(RENDER_OUTPUT_FOLDER, "*.mp4")
    mp4_files = glob.glob(mp4_pattern)
    if not mp4_files:
        print(f"[ERROR] No MP4 found in: {RENDER_OUTPUT_FOLDER}")
        return False

    mp4_files.sort(key=os.path.getmtime, reverse=True)
    rendered_file = mp4_files[0]

    # Name MP4 after WAV file
    if audio_file:
        basename = os.path.splitext(os.path.basename(audio_file))[0]
    else:
        import time
        basename = f"render_{time.strftime('%Y%m%d_%H%M%S')}"

    dest_path = os.path.join(DEFAULT_TTS_OUTPUT, f"{basename}.mp4")

    try:
        os.makedirs(DEFAULT_TTS_OUTPUT, exist_ok=True)
        shutil.copy2(rendered_file, dest_path)
        print(f"[SUCCESS] MP4 copied to: {dest_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Copy failed: {e}")
        return False


def launch_ue_rendering(ue_launch_script):
    """
    Launch UE remote rendering

    Args:
        ue_launch_script: Path to launch_ue_remote.py

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*80)
    print("[STEP 3/3] Launching UE Remote Rendering")
    print("="*80)

    if os.path.exists(ue_launch_script):
        try:
            print(f"Running: {sys.executable} {ue_launch_script}")
            subprocess.run([sys.executable, ue_launch_script], check=True)
            print("[SUCCESS] UE rendering launched")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to launch UE rendering: {e}")
            return False
    else:
        print(f"[WARNING] UE launch script not found: {ue_launch_script}")
        print(f"\nManual launch required:")
        print(f"  Run: python {ue_launch_script}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS Audio and Prepare for UE Rendering"
    )

    parser.add_argument(
        '--language',
        type=str,
        choices=['en', 'zh', 'ja', 'ar', 'es'],
        help='Generate specific language'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all languages'
    )

    parser.add_argument(
        '--emotion',
        type=str,
        default='auto',
        help='Emotion for TTS (default: auto - detected by OpenAI)'
    )

    parser.add_argument(
        '--no-openai-prosody',
        action='store_true',
        help='Skip OpenAI prosody optimization'
    )

    parser.add_argument(
        '--copy-only',
        action='store_true',
        help='Only copy existing audio (skip TTS generation)'
    )

    parser.add_argument(
        '--audio',
        type=str,
        help='Path to existing audio file (for --copy-only)'
    )

    parser.add_argument(
        '--ue-input',
        type=str,
        default=UE_INPUT_FOLDER,
        help=f'UE input folder path (default: {UE_INPUT_FOLDER})'
    )

    parser.add_argument(
        '--ue-launch-script',
        type=str,
        default=UE_LAUNCH_SCRIPT,
        help=f'UE launch script path (default: {UE_LAUNCH_SCRIPT})'
    )

    parser.add_argument(
        '--no-launch',
        action='store_true',
        help='Skip UE rendering launch'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.copy_only and not args.language and not args.all:
        parser.error("Either --language, --all, or --copy-only must be specified")

    if args.copy_only and not args.audio:
        parser.error("--audio must be specified with --copy-only")

    print("\n" + "="*80)
    print("GENERATE TTS AND PREPARE FOR UE RENDERING")
    print("="*80)

    # Step 1: Generate TTS (or use existing)
    if args.copy_only:
        audio_file = args.audio
        print(f"\n[COPY-ONLY MODE] Using existing audio: {audio_file}")
    else:
        audio_file = generate_tts(
            language=args.language,
            emotion=args.emotion,
            use_openai_prosody=not args.no_openai_prosody
        )
        if not audio_file:
            print("\n[FAILED] TTS generation failed")
            sys.exit(1)

    run_pipeline_for_audio(audio_file, args.ue_input, args.ue_launch_script, args.no_launch)

    print("\n" + "="*80)
    print("[COMPLETE] Workflow finished")
    print("="*80)


if __name__ == "__main__":
    main()
