"""
Test script to verify UE timeline adjustment works correctly.

This script ONLY tests timeline adjustment without rendering.
Run this in UE to debug timeline issues.

Usage:
    1. Set the AUDIO_FILE_PATH below to your actual WAV file
    2. Run this script in UE via Remote Control or Python console
    3. Check the logs to see if timeline adjustment worked
"""

import unreal

# ===== CONFIGURATION - CHANGE THIS TO YOUR ACTUAL AUDIO FILE =====
AUDIO_FILE_PATH = "C:/Users/marketing/Desktop/A2F_cynthia/tts-ue-pipeline/output/cpu_gpu_en_neutral.wav"
LEVEL_SEQUENCE_PATH = "/Game/backup_NewLevelSequence1"  # Updated to match the sequence with correct timeline

def test_timeline_adjustment():
    """
    Test timeline adjustment without rendering.
    """
    import wave
    import os

    unreal.log("=" * 80)
    unreal.log("UE TIMELINE ADJUSTMENT TEST")
    unreal.log("=" * 80)

    # Load the sequence
    unreal.log(f"\n[1/5] Loading Level Sequence: {LEVEL_SEQUENCE_PATH}")
    sequence = unreal.load_asset(LEVEL_SEQUENCE_PATH)
    if not sequence:
        unreal.log_error(f"Failed to load sequence: {LEVEL_SEQUENCE_PATH}")
        return False

    unreal.log("✓ Sequence loaded successfully")

    # Get current timeline BEFORE adjustment
    unreal.log(f"\n[2/5] Reading CURRENT timeline (before adjustment):")
    frame_rate = sequence.get_display_rate()
    fps = frame_rate.numerator / frame_rate.denominator

    current_start = sequence.get_playback_start()
    current_end = sequence.get_playback_end()
    current_duration_frames = current_end - current_start
    current_duration_seconds = current_duration_frames / fps

    unreal.log(f"  Frame rate: {fps} fps")
    unreal.log(f"  Start frame: {current_start}")
    unreal.log(f"  End frame: {current_end}")
    unreal.log(f"  Duration: {current_duration_frames} frames = {current_duration_seconds:.2f} seconds")
    unreal.log(f"  👆 THIS IS YOUR CURRENT 42-SECOND TIMELINE")

    # Read audio file
    unreal.log(f"\n[3/5] Reading audio file: {AUDIO_FILE_PATH}")

    if not os.path.exists(AUDIO_FILE_PATH):
        unreal.log_error(f"Audio file not found: {AUDIO_FILE_PATH}")
        unreal.log_error("Please update AUDIO_FILE_PATH in this script to point to your actual WAV file!")
        return False

    with wave.open(AUDIO_FILE_PATH, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        audio_duration = frames / float(rate)

    unreal.log(f"✓ Audio file found: {os.path.basename(AUDIO_FILE_PATH)}")
    unreal.log(f"  Sample rate: {rate} Hz")
    unreal.log(f"  Frames: {frames}")
    unreal.log(f"  Duration: {audio_duration:.2f} seconds")

    # Calculate target timeline
    target_end_frame = int(audio_duration * fps) + 100  # Add 100-frame buffer

    unreal.log(f"\n[4/5] Adjusting timeline to match audio:")
    unreal.log(f"  Target start: 0 frames")
    unreal.log(f"  Target end: {target_end_frame} frames")
    unreal.log(f"  Target duration: {audio_duration:.2f} seconds + buffer")

    # Set all ranges
    sequence.set_playback_start(0)
    sequence.set_playback_end(target_end_frame)
    sequence.set_work_range_start(0)
    sequence.set_work_range_end(target_end_frame)
    sequence.set_view_range_start(0)
    sequence.set_view_range_end(target_end_frame)

    # Save the sequence
    unreal.EditorAssetLibrary.save_loaded_asset(sequence)

    # Refresh sequencer UI
    unreal.LevelSequenceEditorBlueprintLibrary.refresh_current_level_sequence()

    unreal.log("✓ Timeline ranges set and saved")

    # VERIFY: Read back the values
    unreal.log(f"\n[5/5] VERIFICATION - Reading timeline AFTER adjustment:")

    new_start = sequence.get_playback_start()
    new_end = sequence.get_playback_end()
    new_duration_frames = new_end - new_start
    new_duration_seconds = new_duration_frames / fps

    unreal.log(f"  Start frame: {new_start}")
    unreal.log(f"  End frame: {new_end}")
    unreal.log(f"  Duration: {new_duration_frames} frames = {new_duration_seconds:.2f} seconds")

    # Check if it worked
    unreal.log("\n" + "=" * 80)
    if abs(new_duration_seconds - audio_duration) < 5.0:  # Allow 5 second tolerance for buffer
        unreal.log("✅ SUCCESS! Timeline was adjusted correctly!")
        unreal.log(f"   Before: {current_duration_seconds:.2f}s → After: {new_duration_seconds:.2f}s")
        unreal.log("=" * 80)
        return True
    else:
        unreal.log("❌ FAILED! Timeline was NOT adjusted correctly!")
        unreal.log(f"   Expected: ~{audio_duration:.2f}s")
        unreal.log(f"   Got: {new_duration_seconds:.2f}s")
        unreal.log(f"   Difference: {abs(new_duration_seconds - audio_duration):.2f}s")
        unreal.log("\nPossible issues:")
        unreal.log("  1. Sequence asset is read-only or locked")
        unreal.log("  2. Another script is overriding the timeline")
        unreal.log("  3. UE is caching old values (try restarting UE)")
        unreal.log("=" * 80)
        return False

if __name__ == "__main__":
    test_timeline_adjustment()
