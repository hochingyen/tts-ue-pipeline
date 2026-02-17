#!/usr/bin/env python3
"""
Unreal Engine Metadata Generator

Generates JSON metadata files for voice assets to be used in Unreal Engine.
Includes voice gender, emotion, audio info, and plugin configuration.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import wave


# Unreal Engine Plugin Mapping
# Maps (gender, emotion) → UE plugin identifier
UE_PLUGIN_MAPPING = {
    # Male voices
    ("male", "neutral"): "UE_TTS_Male_Neutral",
    ("male", "calm"): "UE_TTS_Male_Neutral",
    ("male", "serious"): "UE_TTS_Male_Neutral",
    ("male", "happy"): "UE_TTS_Male_Emotional",
    ("male", "excited"): "UE_TTS_Male_Emotional",
    ("male", "confident"): "UE_TTS_Male_Emotional",
    ("male", "sad"): "UE_TTS_Male_Emotional",
    ("male", "angry"): "UE_TTS_Male_Emotional",
    ("male", "fear"): "UE_TTS_Male_Emotional",
    ("male", "nervous"): "UE_TTS_Male_Emotional",
    ("male", "surprise"): "UE_TTS_Male_Emotional",
    ("male", "disgust"): "UE_TTS_Male_Emotional",

    # Female voices
    ("female", "neutral"): "UE_TTS_Female_Neutral",
    ("female", "calm"): "UE_TTS_Female_Neutral",
    ("female", "serious"): "UE_TTS_Female_Neutral",
    ("female", "happy"): "UE_TTS_Female_Emotional",
    ("female", "excited"): "UE_TTS_Female_Emotional",
    ("female", "confident"): "UE_TTS_Female_Emotional",
    ("female", "sad"): "UE_TTS_Female_Emotional",
    ("female", "angry"): "UE_TTS_Female_Emotional",
    ("female", "fear"): "UE_TTS_Female_Emotional",
    ("female", "nervous"): "UE_TTS_Female_Emotional",
    ("female", "surprise"): "UE_TTS_Female_Emotional",
    ("female", "disgust"): "UE_TTS_Female_Emotional",
}


# Voice profile mapping (for future expansion)
VOICE_PROFILES = {
    "male": {
        "neutral": "deep_male_01",
        "serious": "authoritative_male_01",
        "confident": "strong_male_01",
        "angry": "aggressive_male_01",
    },
    "female": {
        "neutral": "clear_female_01",
        "happy": "bright_female_01",
        "confident": "strong_female_01",
        "sad": "soft_female_01",
    }
}


class UEMetadataGenerator:
    """
    Generate Unreal Engine compatible metadata for voice assets.
    """

    def __init__(self, metadata_version: str = "1.0"):
        """
        Initialize metadata generator.

        Args:
            metadata_version: Version string for metadata schema
        """
        self.version = metadata_version

    def generate(
        self,
        audio_path: str,
        gender: str,
        emotion: str,
        language: str,
        emotion_params: Dict[str, float],
        character_name: Optional[str] = None,
        original_text: Optional[str] = None,
        spoken_text: Optional[str] = None
    ) -> Dict:
        """
        Generate complete metadata for a voice asset.

        Args:
            audio_path: Path to the WAV file
            gender: "male" or "female"
            emotion: One of 12 emotions
            language: Language code (e.g., "en", "es", "fr")
            emotion_params: Dict with exaggeration, temperature, cfg_weight
            character_name: Optional character identifier
            original_text: Original input text
            spoken_text: Prosody-optimized text used for TTS

        Returns:
            Dictionary containing all metadata
        """
        audio_info = self._extract_audio_info(audio_path)
        plugin_config = self._get_plugin_config(gender, emotion)

        metadata = {
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "character_name": character_name or "unknown",
            "voice_gender": gender,
            "emotion": emotion,
            "language": language,

            "unreal_plugin_config": plugin_config,
            "audio_info": audio_info,
            "emotion_parameters": emotion_params,

            "text_data": {
                "original_text": original_text or "",
                "spoken_text": spoken_text or "",
            }
        }

        return metadata

    def _extract_audio_info(self, audio_path: str) -> Dict:
        """Extract audio file information."""
        try:
            with wave.open(audio_path, 'rb') as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                n_frames = wav.getnframes()
                duration = n_frames / sample_rate

            file_size = os.path.getsize(audio_path)

            return {
                "file_path": str(Path(audio_path).name),  # Just filename for portability
                "sample_rate": sample_rate,
                "bit_depth": sample_width * 8,
                "channels": channels,
                "duration_seconds": round(duration, 2),
                "format": "wav",
                "file_size_bytes": file_size
            }

        except Exception as e:
            print(f"[Warning] Could not extract audio info: {e}")
            return {
                "file_path": str(Path(audio_path).name),
                "sample_rate": 24000,
                "bit_depth": 16,
                "channels": 1,
                "duration_seconds": 0.0,
                "format": "wav",
                "file_size_bytes": 0
            }

    def _get_plugin_config(self, gender: str, emotion: str) -> Dict:
        """Get Unreal Engine plugin configuration."""
        plugin_id = UE_PLUGIN_MAPPING.get(
            (gender, emotion),
            "UE_TTS_Default"
        )

        voice_profile = VOICE_PROFILES.get(gender, {}).get(emotion, "default")

        # Emotion blend (for future multi-emotion support)
        emotion_blend = {emotion: 1.0}

        return {
            "plugin_id": plugin_id,
            "voice_profile": voice_profile,
            "emotion_blend": emotion_blend
        }

    def save_metadata(self, metadata: Dict, output_path: str) -> None:
        """
        Save metadata to JSON file.

        Args:
            metadata: Metadata dictionary
            output_path: Path for JSON file (e.g., "output/file_metadata.json")
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"[Metadata] Saved to {output_path}")

        except Exception as e:
            print(f"[Error] Failed to save metadata: {e}")

    def generate_and_save(
        self,
        audio_path: str,
        metadata_path: str,
        gender: str,
        emotion: str,
        language: str,
        emotion_params: Dict[str, float],
        character_name: Optional[str] = None,
        original_text: Optional[str] = None,
        spoken_text: Optional[str] = None
    ) -> Dict:
        """
        Generate metadata and save to file in one call.

        Returns:
            Generated metadata dictionary
        """
        metadata = self.generate(
            audio_path=audio_path,
            gender=gender,
            emotion=emotion,
            language=language,
            emotion_params=emotion_params,
            character_name=character_name,
            original_text=original_text,
            spoken_text=spoken_text
        )

        self.save_metadata(metadata, metadata_path)
        return metadata


def test_metadata_generator():
    """Test metadata generation."""
    generator = UEMetadataGenerator()

    # Create dummy WAV info
    metadata = generator.generate(
        audio_path="output/test.wav",
        gender="female",
        emotion="confident",
        language="en",
        emotion_params={
            "exaggeration": 0.45,
            "temperature": 0.85,
            "cfg_weight": 0.55
        },
        character_name="npc01",
        original_text="This is critical!",
        spoken_text="This is, CRITICAL!"
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    test_metadata_generator()
