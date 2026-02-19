#!/usr/bin/env python3
"""
Multilingual Voice Cloning Benchmark - GPU Generation with Voice Cloning

Uses CPU to load prof_1min.wav voice file once, then generates TTS
for 5 languages (English, Chinese, Japanese, Arabic, Spanish) on GPU.

Features:
- OpenAI prosody optimization for natural speech
- 3-second silence buffer at end of each audio file
- Emotion control via --emotion flag
- Simple output naming: cpu_gpu_language_emotion.wav

Usage:
    # Windows - Specify GPU device
    set CUDA_VISIBLE_DEVICES=1
    python test_multilingual_benchmark.py --all --use-openai-prosody

    # Test specific language with specific emotion
    python test_multilingual_benchmark.py --language en --emotion happy --use-openai-prosody

    # Test without OpenAI (faster, but less natural)
    python test_multilingual_benchmark.py --all --emotion calm
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


# Multilingual test texts (Dr. O'Brien passage translated)
MULTILINGUAL_TEXTS = {
    'en': {
        'name': 'English',
        'text': "Dr. O'Brien read the lead article at 3:45 PM about bass fishing in 1999. The tear in her eye appeared as she tried to tear the paper. The WWW address costs $99.99 plus 15% tax. Will the invalid's invalid license be close enough to close the door? The dove dove into the bushes. LOL!"
    },
    'es': {
        'name': 'Spanish',
        'text': 'El Doctor O\'Brien leyó el artículo principal a las tres y cuarenta y cinco de la tarde sobre la pesca de lubina en mil novecientos noventa y nueve. La lágrima en su ojo apareció mientras intentaba rasgar el papel. La dirección doble uve doble uve doble uve cuesta noventa y nueve con noventa y nueve dólares más quince por ciento de impuesto. ¿Estará la licencia inválida del inválido lo suficientemente cerca para cerrar la puerta? La paloma se lanzó a los arbustos... ¡JAJA!'
    },
    'zh': {
        'name': 'Chinese',
        'text': '奥布莱恩博士在下午三点四十五分阅读了一九九九年关于鲈鱼钓鱼的头条文章。她试图撕开报纸时，眼中流露出泪水。万维网地址的费用是九十九点九九美元，加上百分之十五的税。残疾人的无效驾照是否足够接近以关闭门鸽子飞入了灌木丛中。哈哈！'
    },
    'ja': {
        'name': 'Japanese',
        'text': 'オブライエン博士は、午後三時四十五分に、一九九九年のバス釣りについての主要記事を読みました。彼女がその紙を破ろうとしたとき、彼女の目に涙が浮かびました。ダブリューダブリューダブリューのアドレスは、九十九ドル九十九セントに、十五パーセントの税金がかかります。無効な免許証は、ドアを閉めるのに十分近いでしょうか？ハトは茂みに飛び込みました。笑い！'
    },
    'ar': {
        'name': 'Arabic',
        'text': 'الدُّكْتُورُ أُوبْرَايِنُ قَرَأَ الْمَقَالَ الرَّئِيسِيَّ فِي السَّاعَةِ الثَّالِثَةِ وَخَمْسٍ وَأَرْبَعِينَ مَسَاءً عَنْ صَيْدِ السَّمَكِ فِي عَامِ أَلْفٍ وَتِسْعِمِائَةٍ وَتِسْعٍ وَتِسْعِينَ. ظَهَرَتِ الدُّمُوعُ فِي عَيْنَيْهَا عِنْدَمَا حَاوَلَتْ تَمْزِيقَ الْوَرَقَةِ. عُنْوَانُ الشَّبَكَةِ الْعَالَمِيَّةِ يُكَلِّفُ تِسْعَةً وَتِسْعِينَ دُولَارًا وَتِسْعَةً وَتِسْعِينَ سِنْتًا، بِالْإِضَافَةِ إِلَى خَمْسَةَ عَشَرَ بِالْمِئَةِ ضَرِيبَةً. هَلْ سَتَكُونُ رُخْصَةُ الْمُعَاقِينَ غَيْرُ الصَّالِحَةِ قَرِيبَةً بِمَا يَكْفِي لِإِغْلَاقِ الْبَابِ؟ الْحَمَامَةُ انْقَضَّتْ إِلَى الشُّجَيْرَاتِ. هَاهَا!'
    }
}


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


def benchmark_language(
    language: str,
    pipeline: ChatterBoxUEPipeline,
    voice_path: str,
    output_dir: str,
    emotion: str = "auto",
    use_openai_prosody: bool = False,
    analyzer: OpenAIVoiceAnalyzer = None
) -> Dict:
    """
    Benchmark TTS generation for a specific language using GPU with voice cloning.

    Args:
        language: Language code ('zh', 'ja', 'ar', 'es')
        pipeline: Pre-initialized ChatterBox pipeline (on GPU)
        voice_path: Path to professor voice file for cloning
        output_dir: Output directory for generated audio
        emotion: Emotion for TTS generation (default: "serious")
        use_openai_prosody: Use OpenAI to optimize prosody
        analyzer: Pre-initialized OpenAI analyzer (if prosody enabled)

    Returns:
        Dictionary with benchmark results
    """
    if language not in MULTILINGUAL_TEXTS:
        raise ValueError(f"Unsupported language: {language}")

    config = MULTILINGUAL_TEXTS[language]
    text = config['text']
    lang_name = config['name']

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
        results: List of benchmark result dictionaries
    """
    print("\n" + "=" * 100)
    print("[RESULTS] MULTILINGUAL TTS GENERATION RESULTS (GPU with Voice Cloning)")
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
        print(f"  Languages generated: {successful}/4")

    print("=" * 100)


def save_results(results: List[Dict], output_dir: str):
    """
    Save benchmark results to JSON file.

    Args:
        results: List of benchmark results
        output_dir: Output directory
    """
    results_file = os.path.join(output_dir, 'multilingual_benchmark_results.json')

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual Voice Cloning Benchmark - GPU Generation"
    )

    parser.add_argument(
        '--language',
        type=str,
        help='Test a specific language (e.g., en, zh, ja, ar, es, fr, de, etc.)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all benchmark languages (en, zh, ja, ar, es)'
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
        default='gpt-4o',
        help='OpenAI model to use (default: gpt-4o). Options: gpt-4o, gpt-4o-mini, o1-preview, o1-mini'
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

    # Determine which languages to test
    if args.all:
        languages = ['en', 'zh', 'ja', 'ar', 'es']
        print("[START] Generating TTS for ALL benchmark languages (5 total) with voice cloning...")
    else:
        # Check if language has test text in MULTILINGUAL_TEXTS
        if args.language in MULTILINGUAL_TEXTS:
            languages = [args.language]
            print(f"[START] Generating TTS for {MULTILINGUAL_TEXTS[args.language]['name']} with benchmark text...")
        elif args.language in SUPPORTED_LANGUAGES:
            # Language is supported but doesn't have benchmark text
            print(f"[ERROR] Language '{args.language}' ({SUPPORTED_LANGUAGES[args.language]}) is supported")
            print(f"        but doesn't have benchmark test text in MULTILINGUAL_TEXTS.")
            print(f"\nAvailable benchmark languages: {', '.join(MULTILINGUAL_TEXTS.keys())}")
            sys.exit(1)
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

    for language in languages:
        try:
            result = benchmark_language(
                language=language,
                pipeline=pipeline,
                voice_path=voice_path,
                output_dir=output_dir,
                emotion=args.emotion,
                use_openai_prosody=args.use_openai_prosody,
                analyzer=analyzer
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
            results.append({
                'language': language,
                'language_name': MULTILINGUAL_TEXTS[language]['name'],
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
