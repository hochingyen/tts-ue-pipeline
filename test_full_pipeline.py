#!/usr/bin/env python3
"""
Quick Pipeline Test Script

Tests the complete TTS → UE → MP4 workflow with a simple test case.

Usage:
    # Test with keyboard input (interactive)
    python test_full_pipeline.py

    # Test with example JSON file
    python test_full_pipeline.py --use-json

    # Test without OpenAI (faster, for testing setup)
    python test_full_pipeline.py --no-openai

    # Test without launching UE (just generate TTS)
    python test_full_pipeline.py --tts-only
"""

import os
import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="Quick pipeline test")
    parser.add_argument(
        '--use-json',
        action='store_true',
        help='Use example_texts.json instead of keyboard input'
    )
    parser.add_argument(
        '--no-openai',
        action='store_true',
        help='Skip OpenAI prosody optimization (faster test)'
    )
    parser.add_argument(
        '--tts-only',
        action='store_true',
        help='Only test TTS generation (skip UE rendering)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language to test (default: en)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=1,
        help='CUDA GPU device ID (default: 1)'
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print("PIPELINE TEST - TTS → UE → MP4")
    print("="*80)

    # Build command
    cmd = [sys.executable, "generate_and_render_ue.py"]
    cmd.extend(["--language", args.language])
    cmd.extend(["--gpu", str(args.gpu)])

    if args.use_json:
        if not os.path.exists("example_texts.json"):
            print("[ERROR] example_texts.json not found")
            print("Create it with:")
            print('  {"en": {"name": "English", "text": "Hello world!"}}')
            sys.exit(1)
        cmd.extend(["--text-input-json", "example_texts.json"])
        print(f"\n[MODE] Using JSON file: example_texts.json")
    else:
        print(f"\n[MODE] Keyboard input - you will type your text")

    if not args.no_openai:
        cmd.append("--use-openai-prosody")
        print("[QUALITY] OpenAI prosody optimization enabled")
    else:
        print("[SPEED] OpenAI prosody optimization disabled")

    if args.tts_only:
        cmd.append("--no-launch")
        print("[TEST] TTS generation only (UE rendering skipped)")
    else:
        print("[FULL] Complete pipeline: TTS → UE → MP4")

    print(f"\n[LANGUAGE] Testing with: {args.language}")
    print("\n" + "="*80)
    print("Running command:")
    print(" ".join(cmd))
    print("="*80 + "\n")

    # Run the pipeline
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*80)
        print("[SUCCESS] Pipeline test completed!")
        print("="*80)
        print("\nCheck the output/ folder for:")
        print("  - WAV file: cpu_gpu_{language}_{emotion}.wav")
        if not args.tts_only:
            print("  - MP4 file: cpu_gpu_{language}_{emotion}.mp4")
        print("\n" + "="*80)
    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print("[FAILED] Pipeline test failed")
        print("="*80)
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Test cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
