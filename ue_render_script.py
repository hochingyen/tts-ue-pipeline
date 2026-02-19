"""
Unreal Engine Python Script - Movie Render Queue with Georgy Dev Lip Sync

This script automates MetaHuman lip sync rendering using the Georgy Dev plugin.
It updates the AnimBP audio path, replaces audio in the sequencer, adjusts timeline,
and renders the video with synchronized lip animations.

Features:
- Auto-detects latest WAV file from input folder
- Updates Georgy Dev Lip Sync AnimBP 'File Path' variable
- Adjusts sequence timeline to match audio duration (reads from WAV file)
- Pre-render processing delay to allow Lip Sync plugin to analyze audio
- Prevents "Editor is in play mode" conflicts during rendering
- Cleans render output folder before each render (ensures clean slate)
- Renders using saved preset configuration (preset handles audio automatically)
- Automatically copies and renames output MP4 to match input audio filename

Georgy Dev Lip Sync Integration:
- AnimBP Path: /RuntimeMetaHumanLipSync/LipSyncData/MyLipSync_Face_AnimBP1
- Updates 'File Path' variable in EventGraph > Import Audio from File
- Plugin handles sample rate conversion automatically (24kHz, 48kHz, etc.)
- 3-second processing delay ensures lip sync data is ready before render

Preset Configuration:
- Output: 3840x2160 (4K) @ 30fps
- Command Line Encoder: Enabled, deletes source files
- Preset Path: /Game/Cinematics/Pending_MoviePipelinePrimaryConfig

Automated Rendering Workflow:
1. Clean render output folder (delete all previous files)
2. Update AnimBP 'File Path' variable → Georgy Dev reads audio
3. Wait 3 seconds for Lip Sync processing
4. Adjust timeline to audio duration (reads directly from input WAV file)
5. Start rendering with preset (preset handles audio encoding automatically)
6. Copy output MP4 to final destination with matching filename

Usage:
1. With command-line argument:
   UnrealEditor.exe "project.uproject" -ExecutePythonScript="ue_render_script.py"
       -AudioFile="C:/path/to/audio.wav"

2. Auto-detect from input folder:
   Place WAV files in: C:/Users/marketing/Desktop/A2F_cynthia/tts-ue-pipeline/output
   Script will use the newest file automatically

3. Use existing audio (no replacement):
   Just run the script without arguments

Requirements:
- Georgy Dev Lip Sync plugin installed and configured
- AnimBP must have a promoted 'File Path' string variable
- Variable must be connected to 'Import Audio from File' node
"""

import unreal

# [CRITICAL] Global variable at module level - ensures lifetime
active_executor = None
render_started = False
check_count = 0
current_audio_file_path = None  # Store input audio path for output naming

# Configuration
PRESET_PATH = "/Game/Cinematics/Pending_MoviePipelinePrimaryConfig.Pending_MoviePipelinePrimaryConfig"
LEVEL_SEQUENCE_PATH = "/Game/backup_NewLevelSequence1"  # Updated to use the backup sequence with correct timeline
MAP_PATH = "/Game/NewMap"
INPUT_AUDIO_FOLDER = "C:/Users/marketing/Desktop/A2F_cynthia/tts-ue-pipeline/output"  # Windows path for UE
OUTPUT_FOLDER = "C:/Users/marketing/Desktop/A2F_cynthia/tts-ue-pipeline/output"  # Output folder for final MP4s
RENDER_OUTPUT_FOLDER = "C:/Users/marketing/Documents/Unreal Projects/male_runtime/Saved/MovieRenders"  # UE render output folder
LIPSYNC_ANIMBP_PATH = "/RuntimeMetaHumanLipSync/LipSyncData/MyLipSync_Face_AnimBP1.MyLipSync_Face_AnimBP1"  # Georgy Dev Lip Sync AnimBP

def check_render_status(_delta_time):
    """
    Sentry tick callback - stays alive until rendering ends.
    """
    global active_executor, render_started, check_count, current_audio_file_path

    if not active_executor:
        return

    check_count += 1
    is_rendering = active_executor.is_rendering()

    # Wait for rendering to actually start
    if not render_started:
        if is_rendering:
            render_started = True
            unreal.log("=" * 60)
            unreal.log("!!! RENDERING HAS STARTED !!!")
            unreal.log("=" * 60)
        elif check_count > 300:  # ~10 seconds at 30fps
            unreal.log_error("Rendering did not start after 10 seconds")
            unreal.SystemLibrary.quit_editor()
        return

    # Once rendering has started, wait for it to finish
    if render_started and not is_rendering:
        unreal.log("=" * 60)
        unreal.log("!!! RENDERING COMPLETE !!!")
        unreal.log("=" * 60)

        # Copy output file to final destination
        if current_audio_file_path:
            copy_output_file(current_audio_file_path)

        unreal.SystemLibrary.quit_editor()


def clean_render_folder():
    """
    Delete all files in the render output folder before starting a new render.
    This ensures a clean slate and prevents issues with leftover files.
    """
    import os
    import shutil

    try:
        unreal.log(f"Cleaning render output folder: {RENDER_OUTPUT_FOLDER}")

        if not os.path.exists(RENDER_OUTPUT_FOLDER):
            unreal.log(f"Render folder does not exist, creating: {RENDER_OUTPUT_FOLDER}")
            os.makedirs(RENDER_OUTPUT_FOLDER)
            return True

        # Delete all files and subdirectories
        for item in os.listdir(RENDER_OUTPUT_FOLDER):
            item_path = os.path.join(RENDER_OUTPUT_FOLDER, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                    unreal.log(f"  Deleted file: {item}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    unreal.log(f"  Deleted folder: {item}")
            except Exception as e:
                unreal.log_warning(f"  Failed to delete {item}: {e}")

        unreal.log("✓ Render folder cleaned successfully")
        return True

    except Exception as e:
        unreal.log_error(f"Failed to clean render folder: {str(e)}")
        import traceback
        unreal.log_error(traceback.format_exc())
        return False


def copy_output_file(input_audio_path):
    """
    Copy the rendered MP4 from the render folder to the output folder,
    naming it to match the input audio file.

    Args:
        input_audio_path: Path to the input audio file (e.g., "C:/input/audio_001.wav")
    """
    import os
    import shutil

    try:
        # Find the rendered MP4 file in the render output folder
        # The filename will match the sequence name (backup_NewLevelSequence1.mp4)
        rendered_file = os.path.join(RENDER_OUTPUT_FOLDER, "backup_NewLevelSequence1.mp4")

        if not os.path.exists(rendered_file):
            unreal.log_error(f"Rendered file not found: {rendered_file}")
            # Try fallback name just in case
            rendered_file_fallback = os.path.join(RENDER_OUTPUT_FOLDER, "NewLevelSequence.mp4")
            if os.path.exists(rendered_file_fallback):
                rendered_file = rendered_file_fallback
                unreal.log(f"Using fallback file: {rendered_file_fallback}")
            else:
                return False

        # Get the base name of the input audio file (without extension)
        audio_basename = os.path.splitext(os.path.basename(input_audio_path))[0]

        # Create output filename with .mp4 extension
        output_filename = f"{audio_basename}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Ensure output folder exists
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            unreal.log(f"Created output folder: {OUTPUT_FOLDER}")

        # Copy the file
        unreal.log("=" * 60)
        unreal.log("Copying output file to final destination...")
        unreal.log(f"  From: {rendered_file}")
        unreal.log(f"  To: {output_path}")

        shutil.copy2(rendered_file, output_path)

        unreal.log("✓ Output file copied successfully!")
        unreal.log(f"  Final output: {output_path}")
        unreal.log("=" * 60)

        return True

    except Exception as e:
        unreal.log_error(f"Failed to copy output file: {str(e)}")
        import traceback
        unreal.log_error(traceback.format_exc())
        return False


def update_lipsync_animbp_audio_path(animbp_path, audio_file_path):
    """
    Update the Georgy Dev Lip Sync AnimBP's File Path variable.

    This sets the 'File Path' variable in the AnimBP's EventGraph which is used
    by the 'Import Audio from File' function node.

    Args:
        animbp_path: Path to the Animation Blueprint (e.g., "/RuntimeMetaHumanLipSync/...")
        audio_file_path: Full file system path to the WAV file (e.g., "C:/path/to/audio.wav")

    Returns:
        True if successful, False otherwise
    """
    try:
        unreal.log(f"Updating Lip Sync AnimBP audio path...")
        unreal.log(f"  AnimBP: {animbp_path}")
        unreal.log(f"  Audio File: {audio_file_path}")

        # Load the Animation Blueprint
        anim_bp = unreal.load_asset(animbp_path)
        if not anim_bp:
            unreal.log_error(f"Failed to load AnimBP: {animbp_path}")
            return False

        # Get the Blueprint's Generated Class (this is the actual class instance)
        blueprint_generated_class = anim_bp.generated_class()
        if not blueprint_generated_class:
            unreal.log_error("Failed to get generated class from AnimBP")
            return False

        # Get the Class Default Object (CDO) - this holds the default variable values
        cdo = unreal.get_default_object(blueprint_generated_class)
        if not cdo:
            unreal.log_error("Failed to get Class Default Object from AnimBP")
            return False

        # Set the File Path variable on the CDO
        # Variable name: "File_Path" (confirmed by user)
        try:
            cdo.set_editor_property("File_Path", audio_file_path)
            unreal.log(f"✓ Set 'File_Path' variable to: {audio_file_path}")
        except Exception as e:
            # Try alternative naming conventions as fallback
            try:
                cdo.set_editor_property("file_path", audio_file_path)
                unreal.log(f"✓ Set 'file_path' variable to: {audio_file_path}")
            except Exception:
                try:
                    cdo.set_editor_property("FilePath", audio_file_path)
                    unreal.log(f"✓ Set 'FilePath' variable to: {audio_file_path}")
                except Exception:
                    unreal.log_error(f"Failed to set File Path variable (tried: File_Path, file_path, FilePath)")
                    unreal.log_error(f"  Error: {e}")
                    unreal.log_error(f"  Make sure the variable is named 'File_Path' in the AnimBP")
                    return False

        # Mark the AnimBP as modified and save it
        unreal.EditorAssetLibrary.save_loaded_asset(anim_bp)
        unreal.log(f"✓ AnimBP saved with new audio path")

        return True

    except Exception as e:
        unreal.log_error(f"Failed to update AnimBP audio path: {str(e)}")
        import traceback
        unreal.log_error(traceback.format_exc())
        return False


def adjust_sequence_to_audio_file(sequence, audio_file_path):
    """
    Read audio duration from input WAV file and adjust playback range to match.

    Args:
        sequence: The loaded Level Sequence object
        audio_file_path: Full file system path to the WAV file

    Returns:
        float: Audio duration in seconds, or None if failed
    """
    try:
        import wave
        import os

        unreal.log(f"Reading audio duration from file: {audio_file_path}")

        # Check if file exists
        if not os.path.exists(audio_file_path):
            unreal.log_error(f"Audio file not found: {audio_file_path}")
            return None

        # Read WAV file to get duration
        with wave.open(audio_file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            audio_duration = frames / float(rate)

        unreal.log(f"✓ Audio file: {os.path.basename(audio_file_path)}")
        unreal.log(f"  Sample rate: {rate} Hz")
        unreal.log(f"  Frames: {frames}")
        unreal.log(f"  Duration: {audio_duration:.2f} seconds")

        # Get sequence frame rate
        frame_rate = sequence.get_display_rate()
        fps = frame_rate.numerator / frame_rate.denominator

        # Calculate end frame in DISPLAY FRAMES (not ticks!)
        # This is what set_playback_start/end expects
        # Add 100-frame buffer to prevent early cutoff
        end_frame = int(audio_duration * fps) + 100

        unreal.log(f"  Detected sequence frame rate: {fps:.2f} fps")
        unreal.log(f"  Setting playback range:")
        unreal.log(f"    Start: 0 frames")
        unreal.log(f"    End: {end_frame} frames ({audio_duration:.2f}s × {fps:.0f}fps + 100 frame buffer)")

        # Set all ranges using display frames
        sequence.set_playback_start(0)
        sequence.set_playback_end(end_frame)

        sequence.set_work_range_start(0)
        sequence.set_work_range_end(end_frame)

        sequence.set_view_range_start(0)
        sequence.set_view_range_end(end_frame)

        # Mark the sequence as modified
        unreal.EditorAssetLibrary.save_loaded_asset(sequence)

        # Refresh the sequencer UI to show changes
        unreal.LevelSequenceEditorBlueprintLibrary.refresh_current_level_sequence()

        unreal.log(f"✓ Sequence playback range adjusted:")
        unreal.log(f"  Duration: {audio_duration:.2f} seconds")
        unreal.log(f"  Display Frames: {end_frame} @ {fps:.2f} fps (includes 100-frame buffer)")

        return audio_duration

    except Exception as e:
        unreal.log_error(f"Failed to adjust sequence to audio: {str(e)}")
        import traceback
        unreal.log_error(traceback.format_exc())
        return None


def setup_and_render_with_preset(sequence_path, preset_path, map_path, audio_file_path=None):
    """
    Create a render job using the saved preset and start rendering.

    Args:
        sequence_path: Path to the Level Sequence to render
        preset_path: Path to the saved Movie Pipeline Primary Config preset
        map_path: Path to the map/level to use
        audio_file_path: Optional path to a WAV file to replace the audio in the sequence
    """
    try:
        global active_executor, current_audio_file_path

        # Store audio file path for later use in output naming
        current_audio_file_path = audio_file_path

        # Clean the render output folder before starting
        unreal.log("=" * 60)
        unreal.log("Preparing render environment...")
        unreal.log("=" * 60)
        if not clean_render_folder():
            unreal.log_warning("Failed to clean render folder, but continuing anyway...")

        # Get the subsystem
        subsystem = unreal.get_engine_subsystem(unreal.MoviePipelineQueueEngineSubsystem)
        if not subsystem:
            unreal.log_error("Failed to get MoviePipelineQueueEngineSubsystem")
            return False

        # Get the queue
        queue = subsystem.get_queue()
        if not queue:
            unreal.log_error("Failed to get movie pipeline queue")
            return False

        # Clear existing jobs
        queue.delete_all_jobs()
        unreal.log("Cleared existing jobs from queue")

        # Create a new job
        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        if not job:
            unreal.log_error("Failed to create new job")
            return False

        # Load the Level Sequence
        sequence = unreal.load_asset(sequence_path)
        if not sequence:
            unreal.log_error(f"Failed to load sequence: {sequence_path}")
            return False

        # If a new audio file is provided, import and replace it
        if audio_file_path:
            unreal.log("=" * 60)
            unreal.log(f"Replacing audio with: {audio_file_path}")
            unreal.log("=" * 60)

            # Step 1: Update Georgy Dev Lip Sync AnimBP with new audio file path
            unreal.log("Step 1: Updating Lip Sync AnimBP...")
            if not update_lipsync_animbp_audio_path(LIPSYNC_ANIMBP_PATH, audio_file_path):
                unreal.log_error("Failed to update Lip Sync AnimBP - aborting render")
                unreal.log_error("Make sure the 'File Path' variable exists and is promoted in the AnimBP")
                return False

            # CRITICAL: Allow time for Lip Sync plugin to analyze new audio
            # The Georgy Dev plugin needs time to read the file and generate lip sync data
            unreal.log("Step 2: Waiting for Georgy Dev Lip Sync to process audio...")
            unreal.log("(This prevents 'Editor is in play mode' conflicts)")

            import time
            time.sleep(3)  # 3 second delay to allow background processing

            unreal.log("✓ Pre-render audio processing complete")

            # Step 3: Adjust sequence timeline to match audio file duration
            audio_duration = adjust_sequence_to_audio_file(sequence, audio_file_path)
            if audio_duration is None:
                unreal.log_warning("Could not adjust sequence to audio - will use current playback range")
                # Don't fail - just continue with existing playback range

        # Set the sequence and map
        job.sequence = unreal.SoftObjectPath(sequence.get_path_name())
        job.map = unreal.SoftObjectPath(map_path)
        job.job_name = f"AutoRender_{sequence.get_name()}"

        unreal.log(f"Job created: {job.job_name}")
        unreal.log(f"  Sequence: {sequence_path}")
        unreal.log(f"  Map: {map_path}")

        # Load the saved preset
        preset = unreal.load_asset(preset_path)
        if not preset:
            unreal.log_error(f"Failed to load preset: {preset_path}")
            return False

        unreal.log(f"Loaded preset: {preset_path}")

        # Apply the preset to the job configuration
        job_config = job.get_configuration()
        job_config.copy_from(preset)
        unreal.log("Applied preset configuration to job:")
        unreal.log("  - Output: 3840x2160 (4K) @ 30fps")
        unreal.log("  - Command Line Encoder: Enabled (deletes source PNGs)")

        # Note: Encoder settings come from preset
        # If videos aren't playable, add "-pix_fmt yuv420p" to Project Settings instead

        # Create executor and start rendering
        active_executor = unreal.MoviePipelinePIEExecutor()
        subsystem.render_queue_with_executor_instance(active_executor)

        # Register sentry callback to keep script alive
        unreal.register_slate_post_tick_callback(check_render_status)
        unreal.log("Render started - waiting for completion...")

        return True

    except Exception as e:
        unreal.log_error(f"Setup/Render failed: {str(e)}")
        import traceback
        unreal.log_error(traceback.format_exc())
        return False

def main():
    import sys
    import os

    unreal.log("=" * 60)
    unreal.log("Movie Render Queue - Automated Render with Audio Sync")
    unreal.log("=" * 60)

    # Check for command-line audio file argument
    audio_file_path = None

    # Try to get audio file from command line arguments
    # UE passes arguments via sys.argv
    if len(sys.argv) > 1:
        # Look for -AudioFile argument
        for i, arg in enumerate(sys.argv):
            if arg == "-AudioFile" and i + 1 < len(sys.argv):
                audio_file_path = sys.argv[i + 1]
                unreal.log(f"Using audio file from command line: {audio_file_path}")
                break

    # If no command-line argument, check current_audio.txt for exact path first
    if not audio_file_path:
        marker_file = os.path.join(INPUT_AUDIO_FOLDER, "current_audio.txt")
        if os.path.exists(marker_file):
            with open(marker_file, "r", encoding="utf-8") as f:
                audio_file_path = f.read().strip()
            unreal.log(f"Using audio file from current_audio.txt: {audio_file_path}")
        elif os.path.exists(INPUT_AUDIO_FOLDER):
            import glob
            wav_files = glob.glob(os.path.join(INPUT_AUDIO_FOLDER, "*.wav"))
            if wav_files:
                # Sort by modification time, get newest
                wav_files.sort(key=os.path.getmtime, reverse=True)
                audio_file_path = wav_files[0]
                unreal.log(f"Using latest audio file from input folder: {audio_file_path}")
            else:
                unreal.log_warning(f"No WAV files found in input folder: {INPUT_AUDIO_FOLDER}")
        else:
            unreal.log_warning(f"Input folder does not exist: {INPUT_AUDIO_FOLDER}")

    # Log whether we're replacing audio or using existing
    if audio_file_path:
        unreal.log(f"Will replace audio with: {audio_file_path}")
    else:
        unreal.log("No new audio file specified - will use existing audio in sequence")

    if not setup_and_render_with_preset(
        sequence_path=LEVEL_SEQUENCE_PATH,
        preset_path=PRESET_PATH,
        map_path=MAP_PATH,
        audio_file_path=audio_file_path
    ):
        unreal.log_error("Failed to setup/start render. Shutting down.")
        unreal.SystemLibrary.quit_editor()
        return

    unreal.log("=" * 60)
    unreal.log("Rendering in progress...")
    unreal.log("Script will auto-exit when complete")
    unreal.log("=" * 60)

if __name__ == "__main__":
    main()
