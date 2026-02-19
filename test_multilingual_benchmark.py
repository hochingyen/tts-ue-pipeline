#!/usr/bin/env python3
"""
Multilingual TTS Generation with Voice Cloning

Generate high-quality TTS audio with voice cloning support for 23 languages.

Features:
- Two text input modes: Keyboard input or JSON file
- OpenAI prosody optimization for natural speech
- 3-second silence buffer at end of audio
- Emotion control via --emotion flag
- GPU acceleration support

Usage:
    # Single language - Keyboard input (interactive)
    python test_multilingual_benchmark.py --language ar --use-openai-prosody

    # Single language - JSON file input
    python test_multilingual_benchmark.py --language en --text-input-json my_texts.json

    # Multiple languages - JSON file required
    python test_multilingual_benchmark.py --all --text-input-json example_texts.json --use-openai-prosody

    # Windows - Specify GPU device
    set CUDA_VISIBLE_DEVICES=1
    python test_multilingual_benchmark.py --language zh --use-openai-prosody
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List
import json

# Import our modules
from chatterbox_ue_pipeline import ChatterBoxUEPipeline
from openai_analyzer import OpenAIVoiceAnalyzer
import torch
import torchaudio as ta

# Import language config for all 23 languages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))
from language_config import SUPPORTED_LANGUAGES




def show_language_menu() -> str:
    """
    Display interactive language selection menu for all 23 supported languages.

    Returns:
        Selected language code or 'all' for all languages
    """
    print("\n" + "=" * 80)
    print("MULTILINGUAL TTS BENCHMARK - LANGUAGE SELECTION")
    print("=" * 80)

    # Get all 23 supported languages in alphabetical order
    languages = sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1])

    # Display languages with numbers
    print("\nSupported Languages (23 total):\n")

    # Display in 2 columns for better readability
    col_width = 40
    for i in range(0, len(languages), 2):
        lang1_code, lang1_name = languages[i]
        line = f"  {i+1:2}. [{lang1_code}] {lang1_name}"

        if i + 1 < len(languages):
            lang2_code, lang2_name = languages[i+1]
            line += f"{' ' * (col_width - len(line))} {i+2:2}. [{lang2_code}] {lang2_name}"

        print(line)

    print(f"\n  {len(languages)+1:2}. [all] Test ALL {len(languages)} languages")
    print("\n" + "=" * 80)

    # Get user choice
    while True:
        try:
            choice = input("\nEnter your choice (number or language code): ").strip().lower()

            # Check if it's "all"
            if choice == "all" or choice == str(len(languages) + 1):
                return "all"

            # Check if it's a valid language code
            if choice in SUPPORTED_LANGUAGES:
                return choice

            # Check if it's a valid number
            try:
                num = int(choice)
                if 1 <= num <= len(languages):
                    return languages[num - 1][0]
            except ValueError:
                pass

            print(f"Invalid choice: '{choice}'. Please enter a number (1-{len(languages)+1}) or language code.")

        except KeyboardInterrupt:
            print("\n\nSelection cancelled.")
            sys.exit(0)


def add_silence_to_audio(
    audio_file: str,
    silence_seconds: float = 3.0
) -> bool:
    """
    Add silence to the end of an audio file.

    Args:
        audio_file: Path to audio file
        silence_seconds: Seconds of silence to add

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"  [Post-processing] Adding {silence_seconds}s silence buffer...")

        # Load audio
        waveform, sr = ta.load(audio_file)
        print(f"    Loaded audio: {waveform.shape[1]} samples at {sr}Hz")

        # Create silence
        silence_samples = int(silence_seconds * sr)
        silence = torch.zeros((waveform.shape[0], silence_samples))
        print(f"    Creating silence: {silence_samples} samples ({silence_seconds}s)")

        # Concatenate
        audio_with_silence = torch.cat([waveform, silence], dim=1)
        print(f"    New audio length: {audio_with_silence.shape[1]} samples")

        # Save back to same file
        ta.save(audio_file, audio_with_silence, sr)
        print(f"  [SUCCESS] Added {silence_seconds}s silence")

        return True

    except Exception as e:
        print(f"  [ERROR] Error adding silence: {e}")
        import traceback
        traceback.print_exc()
        return False


def set_gpu_device(gpu_id: int = None):
    """
    Set GPU device for CUDA (Windows compatible).

    Args:
        gpu_id: GPU device ID (e.g., 0, 1). If None, uses existing environment variable.
    """
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"Using existing CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print("No GPU device specified, using default")


def get_text_input(language: str, lang_name: str) -> str:
    """
    Get text input from user via keyboard (interactive mode).

    Args:
        language: Language code
        lang_name: Full language name for display

    Returns:
        User-provided text string
    """
    print("\n" + "=" * 80)
    print(f"TEXT INPUT - {lang_name} ({language})")
    print("=" * 80)
    print("\nEnter your text below (press Ctrl+D on Mac/Linux or Ctrl+Z on Windows when done):")
    print("TIP: For multi-line text, keep typing. Press Enter for new lines.")
    print("-" * 80)

    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    except KeyboardInterrupt:
        print("\n\nInput cancelled.")
        sys.exit(0)

    text = '\n'.join(lines).strip()

    if not text:
        print("\n[ERROR] No text provided!")
        sys.exit(1)

    print("\n" + "-" * 80)
    print(f"[INPUT] Received {len(text)} characters")
    print(f"[PREVIEW] {text[:100]}{'...' if len(text) > 100 else ''}")
    print("=" * 80)

    return text


def load_texts_from_json(json_path: str) -> Dict:
    """
    Load multilingual texts from JSON file.

    Expected format:
    {
        "en": {"name": "English", "text": "..."},
        "es": {"name": "Spanish", "text": "..."},
        ...
    }

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary in MULTILINGUAL_TEXTS format
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\n[JSON] Loaded {len(data)} language(s) from {json_path}")
        for lang_code, lang_data in data.items():
            lang_name = lang_data.get('name', lang_code)
            text_len = len(lang_data.get('text', ''))
            print(f"  [{lang_code}] {lang_name}: {text_len} characters")

        return data

    except FileNotFoundError:
        print(f"\n[ERROR] JSON file not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] Invalid JSON format: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to load JSON: {e}")
        sys.exit(1)


def generate_language(
    language: str,
    pipeline: ChatterBoxUEPipeline,
    voice_path: str,
    output_dir: str,
    emotion: str = "auto",
    use_openai_prosody: bool = False,
    analyzer: OpenAIVoiceAnalyzer = None,
    text_source: Dict = None,
    custom_text: str = None
) -> Dict:
    """
    Generate TTS for a specific language using GPU with voice cloning.

    Args:
        language: Language code ('zh', 'ja', 'ar', 'es', etc.)
        pipeline: Pre-initialized ChatterBox pipeline (on GPU)
        voice_path: Path to professor voice file for cloning
        output_dir: Output directory for generated audio
        emotion: Emotion for TTS generation (default: "auto")
        use_openai_prosody: Use OpenAI to optimize prosody
        analyzer: Pre-initialized OpenAI analyzer (if prosody enabled)
        text_source: Dictionary of texts from JSON file
        custom_text: Direct text string from keyboard input

    Returns:
        Dictionary with generation results
    """
    # Determine text source priority: custom_text > text_source
    if custom_text:
        text = custom_text
        lang_name = SUPPORTED_LANGUAGES.get(language, language)
        print(f"\n[Text Source] Keyboard input ({len(text)} chars)")
    elif text_source and language in text_source:
        config = text_source[language]
        text = config['text']
        lang_name = config['name']
        print(f"\n[Text Source] JSON file ({len(text)} chars)")
    else:
        raise ValueError(f"No text provided for language: {language}")

    print("\n" + "=" * 80)
    print(f"Generating: {lang_name}")
    if use_openai_prosody:
        print("OpenAI Prosody Optimization: ENABLED")
        print("3-Second Silence Buffer: ENABLED")
    print("=" * 80)

    # Analyze and optimize text with OpenAI (if enabled)
    prosody_time = 0
    original_text = text
    spoken_text = text
    if use_openai_prosody and analyzer:
        print(f"\n[1/3] Analyzing and optimizing with OpenAI GPT-4o ({config['name']})...")
        prosody_start = time.time()
        try:
            # Single call for all languages: gender + emotion + normalization + prosody
            result = analyzer.analyze_and_optimize(text, language)
            detected_emotion = result.get("emotion", "neutral")
            optimized_text = result.get("spoken_text", text)

            if emotion == "auto":
                emotion = detected_emotion
                print(f"  [Auto emotion] Detected: {emotion}")
            else:
                print(f"  [Manual emotion] Using '{emotion}' (OpenAI detected '{detected_emotion}')")

            prosody_time = time.time() - prosody_start
            print(f"[PROSODY] Optimized in {prosody_time:.2f}s")
            print(f"  Original:  {text[:80]}...")
            print(f"  Optimized: {optimized_text[:80]}...")
            spoken_text = optimized_text
            text = optimized_text
        except Exception as e:
            print(f"[WARNING] Analysis/optimization failed: {e}")
            print(f"   Continuing with original text")
            prosody_time = 0

    # Safety fallback: if emotion is still "auto" (e.g. OpenAI disabled), use neutral
    if emotion == "auto":
        emotion = "neutral"

    # Generate audio with voice cloning on GPU
    step_num = "[2/3]" if use_openai_prosody else "[1/2]"
    print(f"\n{step_num} Generating {lang_name} TTS on GPU with voice cloning...")
    print(f"  Text length: {len(text)} characters")
    print(f"  Voice file: {voice_path}")

    gen_start = time.time()

    # Output filename: cpu_gpu_language_emotion.wav
    character_name = f"cpu_gpu_{language}_{emotion}"

    try:
        result = pipeline.generate(
            text=text,
            character_name=character_name,
            output_dir=output_dir,
            language=language,
            force_gender="female",
            force_emotion=emotion,
            enable_analysis=False,
            audio_prompt_path=voice_path
        )

        gen_time = time.time() - gen_start
        print(f"[SUCCESS] Audio generated in {gen_time:.2f}s")

        # Rename to simple format: cpu_gpu_language.wav
        old_file = result['audio_file']
        old_metadata = result['metadata_file']

        simple_filename = f"{character_name}.wav"
        simple_metadata = f"{character_name}.json"

        new_file = os.path.join(output_dir, simple_filename)
        new_metadata = os.path.join(output_dir, simple_metadata)

        # Rename audio file
        if os.path.exists(old_file):
            # Remove destination if it exists
            if os.path.exists(new_file):
                os.remove(new_file)
            os.rename(old_file, new_file)
            result['audio_file'] = new_file
            print(f"  Renamed to: {simple_filename}")
        else:
            print(f"  [WARNING] Audio file not found: {old_file}")
            new_file = old_file

        # Add 3-second silence buffer (if enabled)
        silence_added = False
        if use_openai_prosody:
            step_num = "[3/3]" if use_openai_prosody else "[2/2]"
            print(f"\n{step_num} Post-processing audio...")
            silence_added = add_silence_to_audio(new_file, silence_seconds=3.0)

        # Update metadata file
        if os.path.exists(old_metadata):
            # Update JSON content to reference new filename
            with open(old_metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Update filename field in metadata
            metadata['filename'] = simple_filename

            # Save text comparison for OpenAI quality verification
            metadata['text_data'] = {
                'original_text': original_text,
                'spoken_text': spoken_text
            }

            # Rename metadata file (remove destination if exists)
            if os.path.exists(new_metadata):
                os.remove(new_metadata)
            os.rename(old_metadata, new_metadata)

            # Write updated metadata
            with open(new_metadata, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            result['metadata_file'] = new_metadata
            print(f"  Metadata: {simple_metadata}")

    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'language': language,
            'language_name': lang_name,
            'success': False,
            'error': str(e),
            'prosody_time': prosody_time
        }

    total_time = prosody_time + gen_time

    print(f"\n[COMPLETE] {lang_name} generation complete!")
    if use_openai_prosody:
        print(f"  Prosody optimization: {prosody_time:.2f}s")
    print(f"  TTS generation: {gen_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    if use_openai_prosody:
        print(f"  3s silence buffer: {'[ADDED]' if silence_added else '[FAILED]'}")
    print("=" * 80)

    return {
        'language': language,
        'language_name': lang_name,
        'voice_file': voice_path,
        'voice_found': os.path.exists(voice_path) if voice_path else False,
        'text_length': len(text),
        'output_file': new_file,
        'prosody_time': prosody_time,
        'generation_time': gen_time,
        'total_time': total_time,
        'use_openai_prosody': use_openai_prosody,
        'silence_added': silence_added if use_openai_prosody else False,
        'success': True
    }


def print_results_table(results: List[Dict]):
    """
    Print a formatted results table.

    Args:
        results: List of generation result dictionaries
    """
    print("\n" + "=" * 100)
    print("[RESULTS] MULTILINGUAL TTS GENERATION RESULTS")
    print("=" * 100)

    # Check if OpenAI prosody was used
    has_prosody = any(r.get('prosody_time', 0) > 0 for r in results if r.get('success'))

    # Print table header
    if has_prosody:
        print(f"\n{'Language':<15} {'Prosody (s)':<12} {'Generation (s)':<15} {'Total (s)':<12} {'3s Silence':<12}")
        print("-" * 100)
    else:
        print(f"\n{'Language':<15} {'Generation (s)':<15} {'Total (s)':<12}")
        print("-" * 100)

    # Print results for each language
    total_prosody = 0
    total_gen = 0
    successful = 0

    for result in results:
        if not result.get('success'):
            continue

        lang_name = result['language_name']
        prosody_time = result.get('prosody_time', 0)
        gen_time = result['generation_time']
        total_time = result['total_time']
        silence = "[YES]" if result.get('silence_added') else "[NO]"

        if has_prosody:
            print(f"{lang_name:<15} {prosody_time:<12.2f} {gen_time:<15.2f} {total_time:<12.2f} {silence:<12}")
        else:
            print(f"{lang_name:<15} {gen_time:<15.2f} {total_time:<12.2f}")

        total_prosody += prosody_time
        total_gen += gen_time
        successful += 1

    print("-" * 100)

    # Print summary statistics
    if successful > 0:
        avg_prosody = total_prosody / successful
        avg_gen = total_gen / successful
        avg_total = (total_prosody + total_gen) / successful

        print("\nSummary:")
        if has_prosody:
            print(f"  Average prosody time: {avg_prosody:.2f}s")
        print(f"  Average generation time: {avg_gen:.2f}s")
        print(f"  Average total time: {avg_total:.2f}s")
        print(f"  Successful generations: {successful}")

    print("=" * 100)


def save_results(results: List[Dict], output_dir: str):
    """
    Save generation results to JSON file.

    Args:
        results: List of generation results
        output_dir: Output directory
    """
    results_file = os.path.join(output_dir, 'multilingual_generation_results.json')

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual TTS Generation with Voice Cloning - Supports keyboard and JSON file input"
    )

    parser.add_argument(
        '--language',
        type=str,
        help='Test a specific language (e.g., en, zh, ja, ar, es, fr, de, etc.)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate TTS for all languages in JSON file (requires --text-input-json)'
    )

    parser.add_argument(
        '--voice',
        type=str,
        default='clone_voice/prof_1min.wav',
        help='Path to professor voice file for cloning (default: clone_voice/prof_1min.wav)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for generated audio and results'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        help='GPU device ID for CUDA (e.g., 0, 1)'
    )

    parser.add_argument(
        '--no-table',
        action='store_true',
        help='Skip printing results table'
    )

    parser.add_argument(
        '--use-openai-prosody',
        action='store_true',
        help='Use OpenAI to optimize prosody + add 3s silence buffer at end'
    )

    parser.add_argument(
        '--emotion',
        type=str,
        default='auto',
        choices=['auto', 'neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust',
                 'excited', 'calm', 'confident', 'nervous', 'serious'],
        help='Emotion for TTS generation (default: auto - detected by OpenAI)'
    )

    parser.add_argument(
        '--openai-model',
        type=str,
        default='gpt-5.2',
        help='OpenAI model to use (default: gpt-5.2). '
             'GPT-5: gpt-5.2, gpt-5-mini, gpt-5-nano | '
             'GPT-4: gpt-4o, gpt-4o-mini | '
             'O1: o1-preview, o1-mini'
    )

    parser.add_argument(
        '--text-input-json',
        type=str,
        help='Path to JSON file with texts (format: {"en": {"name": "English", "text": "..."}, ...}). Required for --all mode.'
    )

    args = parser.parse_args()

    # Interactive mode: if no language specified and --all not used
    if not args.language and not args.all:
        choice = show_language_menu()

        if choice == "all":
            args.all = True
        else:
            args.language = choice

    # Validate arguments
    if args.language and args.all:
        parser.error("Cannot specify both --language and --all")

    # Load texts from JSON if provided
    text_source = None
    if args.text_input_json:
        text_source = load_texts_from_json(args.text_input_json)

    # Determine which languages to test
    if args.all:
        # For --all mode, JSON file is required
        if not args.text_input_json:
            print("\n[ERROR] --all mode requires --text-input-json")
            print("\nExample:")
            print("  python test_multilingual_benchmark.py --all --text-input-json example_texts.json")
            print("\nSee example_texts.json for format")
            sys.exit(1)

        # Use languages from JSON file
        languages = list(text_source.keys())
        print(f"[START] Generating TTS for {len(languages)} languages from JSON file...")
    else:
        # Single language mode
        if args.language in SUPPORTED_LANGUAGES:
            languages = [args.language]
            lang_name = SUPPORTED_LANGUAGES[args.language]
            print(f"[START] Generating TTS for {lang_name} ({args.language})...")
        else:
            print(f"[ERROR] Language '{args.language}' is not supported.")
            print(f"\nSupported languages: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}")
            sys.exit(1)

    # Check voice file
    if not os.path.exists(args.voice):
        print(f"\n[WARNING] Voice file not found: {args.voice}")
        print("   Will proceed with default voices")
        voice_path = None
    else:
        print(f"\n[FOUND] Voice file: {args.voice}")
        voice_path = args.voice

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"[OUTPUT] Directory: {output_dir}")

    # Set GPU device
    if args.gpu is not None:
        set_gpu_device(args.gpu)

    # Initialize OpenAI analyzer (if prosody optimization enabled)
    analyzer = None
    if args.use_openai_prosody:
        print("\n[Setup] Initializing OpenAI analyzer...")
        try:
            analyzer = OpenAIVoiceAnalyzer(model=args.openai_model)
            print("[READY] OpenAI analyzer ready")
        except Exception as e:
            print(f"[ERROR] Failed to initialize OpenAI analyzer: {e}")
            print("  Continuing without prosody optimization")
            args.use_openai_prosody = False

    # Initialize pipeline ONCE on GPU
    print("\n[Setup] Initializing ChatterBox pipeline on GPU...")
    init_start = time.time()

    try:
        pipeline = ChatterBoxUEPipeline(
            device="cuda",
            enable_openai=False  # We handle OpenAI separately for prosody
        )
        init_time = time.time() - init_start
        print(f"[INIT] Pipeline initialized on GPU in {init_time:.2f}s")

        # Verify GPU is being used
        if torch.cuda.is_available():
            print(f"[GPU] {torch.cuda.get_device_name(0)}")
        else:
            print("[WARNING] CUDA not available, will run on CPU (very slow!)")

    except Exception as e:
        print(f"[ERROR] Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Run generation for all languages
    results = []
    total_start = time.time()

    # For single language mode without JSON, get keyboard input
    keyboard_input = None
    if len(languages) == 1 and not args.text_input_json:
        lang_code = languages[0]
        lang_name = SUPPORTED_LANGUAGES[lang_code]
        keyboard_input = get_text_input(lang_code, lang_name)

    for language in languages:
        try:
            result = generate_language(
                language=language,
                pipeline=pipeline,
                voice_path=voice_path,
                output_dir=output_dir,
                emotion=args.emotion,
                use_openai_prosody=args.use_openai_prosody,
                analyzer=analyzer,
                text_source=text_source,
                custom_text=keyboard_input
            )
            results.append(result)

            # Small delay between tests
            if language != languages[-1]:
                print("\n[PAUSE] Pausing 2 seconds before next language...")
                time.sleep(2)

        except Exception as e:
            print(f"\n[ERROR] Generation failed for {language}: {e}")
            import traceback
            traceback.print_exc()

            # Get language name
            if text_source and language in text_source:
                lang_name = text_source[language]['name']
            else:
                lang_name = SUPPORTED_LANGUAGES.get(language, language)

            results.append({
                'language': language,
                'language_name': lang_name,
                'success': False,
                'error': str(e)
            })
            continue

    total_elapsed = time.time() - total_start

    # Print results table
    if not args.no_table:
        print_results_table(results)

    # Save results
    save_results(results, output_dir)

    # Print final summary
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]

    print(f"\n{'='*80}")
    print("[SUMMARY] GENERATION SUMMARY")
    print('='*80)
    print(f"Total time: {total_elapsed:.1f}s (including {init_time:.1f}s pipeline initialization)")
    print(f"Languages processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if args.use_openai_prosody:
        silence_count = sum(1 for r in successful if r.get('silence_added'))
        print(f"3s silence buffer added: {silence_count}/{len(successful)}")

    if failed:
        print("\n[FAILED] Failed generations:")
        for r in failed:
            print(f"  {r['language_name']}: {r.get('error', 'Unknown')}")

    print('='*80)

    if failed:
        sys.exit(1)
    else:
        print("[SUCCESS] All generations completed successfully!")


if __name__ == "__main__":
    main()
