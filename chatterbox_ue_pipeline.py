#!/usr/bin/env python3
"""
ChatterBox-UE Pipeline - Voice Asset Generation for Unreal Engine

Enhanced ChatterBox TTS pipeline with:
- Apple Silicon MPS support
- OpenAI analysis for gender/emotion detection
- 12 emotion support with TTS parameter mapping
- Unreal Engine metadata generation
- UE-friendly file naming convention

Main Features:
    - Automatic language detection (23 languages)
    - OpenAI-powered gender & emotion detection
    - Prosody-optimized text rewriting
    - Apple Silicon MPS GPU acceleration
    - Unreal Engine metadata JSON output
    - File naming: <character>_<gender>_<emotion>_<id>.wav
"""

import os
import sys
import torch
import torchaudio as ta
import numpy as np
import uuid
from pathlib import Path
from typing import Optional, Dict, Tuple
from langdetect import detect
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))
from language_config import (
    LANGDETECT_TO_CHATTERBOX,
    get_language_name,
    is_supported
)

# Import ChatterBox TTS
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Import new modules
from openai_analyzer import OpenAIVoiceAnalyzer
from emotion_config import get_emotion_params, validate_emotion
from ue_metadata import UEMetadataGenerator


class ChatterBoxUEPipeline:
    """
    Enhanced ChatterBox pipeline for Unreal Engine voice asset generation.

    Workflow:
        1. Input: Raw text
        2. OpenAI Analysis: Detect gender, emotion, rewrite for prosody
        3. TTS Generation: ChatterBox with emotion parameters
        4. Output: WAV file + metadata JSON
        5. Naming: <character>_<gender>_<emotion>_<uuid>.wav
    """

    def __init__(
        self,
        device: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        enable_openai: bool = True
    ):
        """
        Initialize ChatterBox-UE pipeline.

        Args:
            device: Computing device ('cuda', 'mps', 'cpu'). Auto-detects if None.
            openai_api_key: OpenAI API key. Uses OPENAI_API_KEY env var if None.
            enable_openai: Enable OpenAI analysis (default: True)
        """
        # Device detection with MPS support
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
                print("[Device] Using Apple Silicon MPS GPU")
            elif torch.cuda.is_available():
                self.device = "cuda"  # NVIDIA GPU
                print("[Device] Using CUDA GPU")
            else:
                self.device = "cpu"
                print("[Device] Using CPU")
        else:
            self.device = device
            print(f"[Device] Using {self.device}")

        # Initialize TTS models
        print("[TTS] Loading ChatterBox models...")
        try:
            # Patch torch.load to force CPU mapping for ChatterBox models
            original_torch_load = torch.load
            def patched_torch_load(f, map_location=None, **kwargs):
                # Force CPU mapping for Mac compatibility
                return original_torch_load(f, map_location='cpu', **kwargs)
            torch.load = patched_torch_load

            # Load to CPU first for compatibility
            self.english_model = ChatterboxTTS.from_pretrained(device="cpu")
            self.multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")

            # Restore original torch.load
            torch.load = original_torch_load

            # Move to target device if not CPU
            if self.device != "cpu":
                print(f"[TTS] Moving models to {self.device}...")
                # Note: MPS may have compatibility issues, use CPU as fallback
                if self.device == "mps":
                    print("[Warning] MPS support is experimental. Using CPU for stability.")
                    self.device = "cpu"
                else:
                    # For CUDA, move models
                    self.english_model = ChatterboxTTS.from_pretrained(device=self.device)
                    self.multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)

            # Fix for transformers attention implementation compatibility
            # Set attn_implementation to 'eager' for all transformer models
            try:
                if hasattr(self.multilingual_model, 't3') and hasattr(self.multilingual_model.t3, 'text'):
                    if hasattr(self.multilingual_model.t3.text, 'config'):
                        self.multilingual_model.t3.text.config._attn_implementation = 'eager'
                if hasattr(self.multilingual_model, 't3') and hasattr(self.multilingual_model.t3, 'speech'):
                    if hasattr(self.multilingual_model.t3.speech, 'config'):
                        self.multilingual_model.t3.speech.config._attn_implementation = 'eager'
            except Exception as patch_error:
                print(f"[Warning] Could not patch attention implementation: {patch_error}")

            print("[TTS] Models loaded successfully")
        except Exception as e:
            print(f"[Error] Failed to load models: {e}")
            raise

        # Initialize OpenAI analyzer
        self.enable_openai = enable_openai
        if self.enable_openai:
            try:
                self.openai_analyzer = OpenAIVoiceAnalyzer(api_key=openai_api_key)
                print("[OpenAI] Analyzer initialized")
            except Exception as e:
                print(f"[Warning] OpenAI initialization failed: {e}")
                print("[Warning] Continuing without OpenAI analysis")
                self.enable_openai = False
                self.openai_analyzer = None
        else:
            self.openai_analyzer = None
            print("[OpenAI] Analysis disabled")

        # Initialize metadata generator
        self.metadata_generator = UEMetadataGenerator()

    def detect_language(self, text: str) -> Tuple[str, str]:
        """
        Detect language from text.

        Returns:
            (language_code, language_name)
        """
        try:
            detected = detect(text)
            chatterbox_code = LANGDETECT_TO_CHATTERBOX.get(detected, detected)

            if not is_supported(chatterbox_code):
                print(f"[Language] Detected '{detected}' not supported, using English")
                return "en", "English"

            lang_name = get_language_name(chatterbox_code)
            print(f"[Language] Detected: {lang_name} ({chatterbox_code})")
            return chatterbox_code, lang_name
        except Exception as e:
            print(f"[Language] Detection failed: {e}, using English")
            return "en", "English"

    def analyze_text(self, text: str, language: str = "en") -> Dict[str, str]:
        """
        Analyze text with OpenAI to get gender, emotion, and optimized text.
        Works for all supported languages via a single GPT-4o call.

        Returns:
            {
                "gender": "male|female",
                "emotion": "emotion_name",
                "spoken_text": "optimized text"
            }
        """
        if not self.enable_openai or self.openai_analyzer is None:
            # Return defaults without OpenAI
            return {
                "gender": "female",
                "emotion": "neutral",
                "spoken_text": text
            }

        print(f"[OpenAI] Analyzing text (language={language})...")
        try:
            result = self.openai_analyzer.analyze_and_optimize(text, language)
            print(f"[OpenAI] Gender: {result['gender']}, Emotion: {result['emotion']}")
            return result
        except Exception as e:
            print(f"[Error] OpenAI analysis failed: {e}")
            return {
                "gender": "female",
                "emotion": "neutral",
                "spoken_text": text
            }

    def generate_audio(
        self,
        text: str,
        language: Optional[str] = None,
        emotion_params: Optional[Dict[str, float]] = None,
        audio_prompt_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text using ChatterBox TTS.

        Args:
            text: Text to synthesize (should be prosody-optimized)
            language: Language code (auto-detect if None)
            emotion_params: Dict with exaggeration, temperature, cfg_weight

        Returns:
            (audio_array, sample_rate)
        """
        # Detect language if not provided
        if language is None:
            language, _ = self.detect_language(text)

        # Get emotion parameters
        if emotion_params is None:
            emotion_params = {
                "exaggeration": 0.5,
                "temperature": 0.8,
                "cfg_weight": 0.5
            }

        print(f"[TTS] Generating audio...")
        print(f"[TTS] Parameters: exag={emotion_params['exaggeration']:.2f}, "
              f"temp={emotion_params['temperature']:.2f}, cfg={emotion_params['cfg_weight']:.2f}")

        try:
            # Use English model for English, multilingual for others
            if language == "en":
                # English model doesn't take language parameter
                audio = self.english_model.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=emotion_params["exaggeration"],
                    cfg_weight=emotion_params["cfg_weight"],
                    temperature=emotion_params["temperature"]
                )
            else:
                # Multilingual model requires language_id
                audio = self.multilingual_model.generate(
                    text=text,
                    language_id=language,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=emotion_params["exaggeration"],
                    cfg_weight=emotion_params["cfg_weight"],
                    temperature=emotion_params["temperature"]
                )

            # Convert to numpy array and get sample rate
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = np.array(audio)

            sample_rate = 24000  # ChatterBox default

            print(f"[TTS] Audio generated: {len(audio_np)} samples @ {sample_rate} Hz")
            return audio_np, sample_rate

        except Exception as e:
            print(f"[Error] TTS generation failed: {e}")
            raise

    def save_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str
    ) -> None:
        """
        Save audio to WAV file.

        Args:
            audio: Audio array
            sample_rate: Sample rate
            output_path: Output file path (must end with .wav)
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio)

            # Ensure 2D tensor (channels x samples)
            if audio_tensor.ndim == 1:
                # Mono audio: add channel dimension
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.ndim == 3:
                # Remove batch dimension if present
                audio_tensor = audio_tensor.squeeze(0)

            ta.save(output_path, audio_tensor, sample_rate)

            print(f"[Audio] Saved: {output_path}")
        except Exception as e:
            print(f"[Error] Failed to save audio: {e}")
            raise

    def generate_ue_filename(
        self,
        character_name: str,
        gender: str,
        emotion: str,
        unique_id: Optional[str] = None
    ) -> str:
        """
        Generate Unreal Engine friendly filename.

        Format: <character>_<gender>_<emotion>_<id>.wav

        Args:
            character_name: Character identifier (e.g., "npc01", "narrator")
            gender: "male" or "female"
            emotion: One of 12 emotions
            unique_id: Optional ID (generates UUID if None)

        Returns:
            Filename (without directory path)

        Example:
            >>> generate_ue_filename("npc01", "female", "happy")
            "npc01_female_happy_a3f8e1b2.wav"
        """
        if unique_id is None:
            unique_id = str(uuid.uuid4())[:8]

        filename = f"{character_name}_{gender}_{emotion}_{unique_id}.wav"
        return filename

    def generate(
        self,
        text: str,
        character_name: str = "character",
        output_dir: str = "output",
        language: Optional[str] = None,
        force_gender: Optional[str] = None,
        force_emotion: Optional[str] = None,
        enable_analysis: Optional[bool] = None,
        unique_id: Optional[str] = None,
        audio_prompt_path: Optional[str] = None
    ) -> Dict:
        """
        Complete pipeline: Text → Analysis → TTS → WAV + Metadata.

        Args:
            text: Input text to synthesize
            character_name: Character identifier for filename
            output_dir: Output directory for files
            language: Language code (auto-detect if None)
            force_gender: Override OpenAI gender detection
            force_emotion: Override OpenAI emotion detection
            enable_analysis: Override default OpenAI enable setting
            unique_id: Custom ID for filename (UUID if None)

        Returns:
            Dictionary with:
                - audio_file: Path to WAV file
                - metadata_file: Path to JSON metadata
                - gender: Detected/forced gender
                - emotion: Detected/forced emotion
                - spoken_text: Final text used for TTS
        """
        print("=" * 70)
        print(f"ChatterBox-UE Pipeline: {text[:50]}...")
        print("=" * 70)

        # Step 1: Detect language (needed before analysis)
        detected_language = language or self.detect_language(text)[0]

        # Step 2: OpenAI Analysis (if enabled)
        use_analysis = enable_analysis if enable_analysis is not None else self.enable_openai

        if use_analysis:
            analysis = self.analyze_text(text, language=detected_language)
            gender = force_gender or analysis["gender"]
            emotion = force_emotion or analysis["emotion"]
            spoken_text = analysis["spoken_text"]
        else:
            gender = force_gender or "female"
            emotion = force_emotion or "neutral"
            spoken_text = text
            print(f"[Analysis] Skipped (gender={gender}, emotion={emotion})")

        # Validate emotion
        if not validate_emotion(emotion):
            print(f"[Warning] Invalid emotion '{emotion}', using 'neutral'")
            emotion = "neutral"

        # Step 3: Get emotion parameters
        emotion_params = get_emotion_params(emotion)

        # Step 4: Generate audio
        audio, sample_rate = self.generate_audio(
            text=spoken_text,
            language=detected_language,
            emotion_params=emotion_params,
            audio_prompt_path=audio_prompt_path
        )

        # Step 5: Generate filename and paths
        filename = self.generate_ue_filename(character_name, gender, emotion, unique_id)
        audio_path = os.path.join(output_dir, filename)
        metadata_path = os.path.join(output_dir, filename.replace(".wav", "_metadata.json"))

        # Step 6: Save audio
        self.save_audio(audio, sample_rate, audio_path)

        # Step 7: Generate and save metadata
        metadata = self.metadata_generator.generate_and_save(
            audio_path=audio_path,
            metadata_path=metadata_path,
            gender=gender,
            emotion=emotion,
            language=detected_language,
            emotion_params=emotion_params,
            character_name=character_name,
            original_text=text,
            spoken_text=spoken_text
        )

        print("=" * 70)
        print(f"[SUCCESS] Generation Complete")
        print(f"  Audio: {audio_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Gender: {gender} | Emotion: {emotion}")
        print("=" * 70)

        return {
            "audio_file": audio_path,
            "metadata_file": metadata_path,
            "gender": gender,
            "emotion": emotion,
            "spoken_text": spoken_text,
            "metadata": metadata
        }


def main():
    """Command-line interface for ChatterBox-UE."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ChatterBox-UE: Voice Asset Generation for Unreal Engine"
    )

    # Required arguments
    parser.add_argument(
        "--text",
        type=str,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--text-file",
        type=str,
        help="Path to text file"
    )

    # Character and output
    parser.add_argument(
        "--character",
        type=str,
        default="character",
        help="Character name for filename (default: character)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Custom ID for filename (default: auto-generate UUID)"
    )

    # Override options
    parser.add_argument(
        "--language",
        type=str,
        help="Language code (auto-detect if not specified)"
    )
    parser.add_argument(
        "--gender",
        type=str,
        choices=["male", "female"],
        help="Force gender (overrides OpenAI detection)"
    )
    parser.add_argument(
        "--emotion",
        type=str,
        choices=[
            "neutral", "happy", "sad", "angry", "fear", "surprise",
            "disgust", "excited", "calm", "confident", "nervous", "serious"
        ],
        help="Force emotion (overrides OpenAI detection)"
    )
    parser.add_argument(
        "--voice",
        type=str,
        help="Path to reference audio file for voice cloning (WAV format, 5+ seconds)"
    )

    # Control flags
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Computing device (auto-detect if not specified)"
    )
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Disable OpenAI analysis"
    )

    args = parser.parse_args()

    # Validate input
    if not args.text and not args.text_file:
        parser.error("Either --text or --text-file must be provided")

    # Read text
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        text = args.text

    # Initialize pipeline
    pipeline = ChatterBoxUEPipeline(
        device=args.device,
        enable_openai=not args.no_openai
    )

    # Generate
    result = pipeline.generate(
        text=text,
        character_name=args.character,
        output_dir=args.output_dir,
        language=args.language,
        force_gender=args.gender,
        force_emotion=args.emotion,
        unique_id=args.id,
        audio_prompt_path=args.voice
    )

    print("\n[SUCCESS] Success!")
    return result


if __name__ == "__main__":
    main()
