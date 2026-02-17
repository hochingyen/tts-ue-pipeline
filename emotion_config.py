#!/usr/bin/env python3
"""
Emotion Configuration for ChatterBox-UE

Defines the 12 standard emotions and their TTS parameter mappings.
Designed for Unreal Engine voice asset generation.
"""

from typing import Dict


# 12 Standard Emotions
EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
    "excited",
    "calm",
    "confident",
    "nervous",
    "serious"
]


# Emotion → Exaggeration Mapping
# Exaggeration controls speech expressiveness in ChatterBox TTS
# Range: 0.25 - 2.0
# - Lower values (0.25-0.4): More subdued, serious tones
# - Medium values (0.4-0.6): Balanced, natural speech
# - Higher values (0.6-2.0): More expressive, energetic
EMOTION_EXAGGERATION = {
    "neutral": 0.40,
    "calm": 0.35,
    "sad": 0.30,
    "serious": 0.35,
    "confident": 0.45,
    "happy": 0.60,
    "excited": 0.75,
    "angry": 0.70,
    "fear": 0.60,
    "nervous": 0.55,
    "surprise": 0.65,
    "disgust": 0.60
}


# Emotion → Temperature Mapping (optional fine-tuning)
# Temperature controls randomness/creativity
# Range: 0.05 - 5.0
EMOTION_TEMPERATURE = {
    "neutral": 0.8,
    "calm": 0.7,
    "sad": 0.75,
    "serious": 0.75,
    "confident": 0.85,
    "happy": 0.9,
    "excited": 0.95,
    "angry": 0.9,
    "fear": 0.85,
    "nervous": 0.85,
    "surprise": 0.9,
    "disgust": 0.85
}


# Emotion → CFG Weight Mapping (optional fine-tuning)
# CFG Weight controls pace/generation quality
# Range: 0.0 - 1.0
EMOTION_CFG_WEIGHT = {
    "neutral": 0.5,
    "calm": 0.45,
    "sad": 0.4,
    "serious": 0.45,
    "confident": 0.55,
    "happy": 0.6,
    "excited": 0.65,
    "angry": 0.65,
    "fear": 0.55,
    "nervous": 0.5,
    "surprise": 0.6,
    "disgust": 0.55
}


def get_emotion_params(emotion: str) -> Dict[str, float]:
    """
    Get all TTS parameters for a given emotion.

    Args:
        emotion: One of the 12 supported emotions

    Returns:
        Dictionary with exaggeration, temperature, cfg_weight

    Example:
        >>> params = get_emotion_params("happy")
        >>> print(params)
        {'exaggeration': 0.6, 'temperature': 0.9, 'cfg_weight': 0.6}
    """
    if emotion not in EMOTIONS:
        print(f"[Warning] Unknown emotion '{emotion}', using 'neutral'")
        emotion = "neutral"

    return {
        "exaggeration": EMOTION_EXAGGERATION[emotion],
        "temperature": EMOTION_TEMPERATURE[emotion],
        "cfg_weight": EMOTION_CFG_WEIGHT[emotion]
    }


def validate_emotion(emotion: str) -> bool:
    """Check if emotion is supported."""
    return emotion in EMOTIONS


def list_emotions() -> list[str]:
    """Get list of all supported emotions."""
    return EMOTIONS.copy()


# Emotion → Description (for documentation/UI)
EMOTION_DESCRIPTIONS = {
    "neutral": "Balanced, natural speech without strong emotion",
    "calm": "Relaxed, peaceful, composed delivery",
    "sad": "Melancholic, somber, low-energy tone",
    "serious": "Professional, grave, matter-of-fact",
    "confident": "Assured, self-assured, strong presence",
    "happy": "Cheerful, positive, upbeat energy",
    "excited": "Enthusiastic, energetic, high-spirited",
    "angry": "Intense, forceful, aggressive tone",
    "fear": "Anxious, worried, tense delivery",
    "nervous": "Uncertain, hesitant, slightly anxious",
    "surprise": "Shocked, astonished, taken aback",
    "disgust": "Revulsion, distaste, disapproval"
}


def get_emotion_description(emotion: str) -> str:
    """Get human-readable description of emotion."""
    return EMOTION_DESCRIPTIONS.get(emotion, "Unknown emotion")


if __name__ == "__main__":
    # Print emotion configuration table
    print("=" * 70)
    print("ChatterBox-UE Emotion Configuration")
    print("=" * 70)
    print(f"\n{'Emotion':<12} {'Exag':<6} {'Temp':<6} {'CFG':<6} {'Description'}")
    print("-" * 70)

    for emotion in EMOTIONS:
        params = get_emotion_params(emotion)
        desc = get_emotion_description(emotion)[:40]
        print(
            f"{emotion:<12} "
            f"{params['exaggeration']:<6.2f} "
            f"{params['temperature']:<6.2f} "
            f"{params['cfg_weight']:<6.2f} "
            f"{desc}"
        )

    print("\n" + "=" * 70)
