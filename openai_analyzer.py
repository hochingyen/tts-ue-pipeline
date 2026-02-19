#!/usr/bin/env python3
"""
OpenAI Analyzer for Voice Asset Generation System

This module analyzes text to determine:
1. Speaker gender (male/female)
2. Dominant emotion (12 emotions)
3. Prosody-optimized spoken text for TTS

Works for all supported languages via a single GPT-4o call.
Designed for Unreal Engine voice asset generation.
"""

import os
import re
import json
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# 12 Standard Emotions for Unreal Engine
SUPPORTED_EMOTIONS = [
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


class OpenAIVoiceAnalyzer:
    """
    Analyzes text using OpenAI GPT-4o for voice asset generation.
    Works for all supported languages in a single API call.

    Returns:
        - gender: male or female
        - emotion: one of 12 supported emotions
        - spoken_text: normalized and prosody-optimized text for TTS
    """

    _LANGUAGE_INFO = {
        "en": {"name": "English",             "punctuation": ". ! ?"},
        "zh": {"name": "Chinese (Simplified)", "punctuation": "。！？，、"},
        "ja": {"name": "Japanese",             "punctuation": "。！？、"},
        "ar": {"name": "Arabic",               "punctuation": "؟ ! ."},
        "es": {"name": "Spanish",              "punctuation": ". ! ? ¿ ¡"},
        "fr": {"name": "French",               "punctuation": ". ! ?"},
        "de": {"name": "German",               "punctuation": ". ! ?"},
        "ko": {"name": "Korean",               "punctuation": ". ! ?"},
        "pt": {"name": "Portuguese",           "punctuation": ". ! ?"},
        "ru": {"name": "Russian",              "punctuation": ". ! ?"},
        "it": {"name": "Italian",              "punctuation": ". ! ?"},
        "nl": {"name": "Dutch",                "punctuation": ". ! ?"},
        "pl": {"name": "Polish",               "punctuation": ". ! ?"},
        "sv": {"name": "Swedish",              "punctuation": ". ! ?"},
        "no": {"name": "Norwegian",            "punctuation": ". ! ?"},
        "da": {"name": "Danish",               "punctuation": ". ! ?"},
        "fi": {"name": "Finnish",              "punctuation": ". ! ?"},
        "tr": {"name": "Turkish",              "punctuation": ". ! ?"},
        "hi": {"name": "Hindi",                "punctuation": "। ! ?"},
        "ms": {"name": "Malay",                "punctuation": ". ! ?"},
        "sw": {"name": "Swahili",              "punctuation": ". ! ?"},
        "he": {"name": "Hebrew",               "punctuation": "? ! ."},
        "el": {"name": "Greek",                "punctuation": ". ! ?"},
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize OpenAI analyzer.

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o)
                   Options:
                   - GPT-5: gpt-5.2, gpt-5-mini, gpt-5-nano
                   - GPT-4: gpt-4o, gpt-4o-mini
                   - O1: o1-preview, o1-mini
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        print(f"[OpenAI] Using model: {self.model}")

    def analyze_and_optimize(self, text: str, language: str = "en") -> Dict[str, str]:
        """
        Single GPT-4o call for all languages: detect gender + emotion,
        normalize text for TTS, and add prosody punctuation.

        Args:
            text: Input text in any supported language
            language: Language code ('en', 'zh', 'ja', 'ar', 'es', etc.)

        Returns:
            Dictionary with keys: gender, emotion, spoken_text

        Example:
            >>> analyzer = OpenAIVoiceAnalyzer()
            >>> result = analyzer.analyze_and_optimize("Dr. Smith costs $99.99!", "en")
            >>> print(result)
            {"gender": "male", "emotion": "confident",
             "spoken_text": "Doctor Smith costs ninety-nine dollars and ninety-nine cents!"}
        """
        lang_info = self._LANGUAGE_INFO.get(language, {"name": language, "punctuation": ". ! ?"})
        lang_name = lang_info["name"]
        punctuation = lang_info["punctuation"]

        system_prompt = f"""#Role: You are a voice director for a game dialogue system.

#Task: Analyze input text and prepare it for text-to-speech (TTS) delivery.

#Topic: Voice asset generation for Unreal Engine.

#Format: Return JSON ONLY in this exact format:
{{
  "gender": "male|female",
  "emotion": "emotion_name",
  "spoken_text": "..."
}}

#Context: The input text is in {lang_name}.

#Goal: Generate natural-sounding TTS audio with appropriate gender voice and emotion.

#Requirements/Constraints:

1. Detect the speaker gender: male or female
   - Infer from pronouns, character names, or content context
   - If ambiguous, choose the most contextually appropriate gender

2. Detect the dominant emotion — choose exactly ONE from this list:
   neutral, happy, sad, angry, fear, surprise, disgust, excited, calm, confident, nervous, serious

3. Write spoken_text: the text rewritten for natural TTS delivery
   - Must be in {lang_name} (same language as input — do NOT translate)
   - Apply normalization rules (see below)
   - Apply prosody rules (see below)

NORMALIZATION — convert to spoken {lang_name} words:

Numbers:
- Integers: write as words in {lang_name} (e.g. "42", "1,234")
- Years: use natural spoken form (e.g. "1999" → "nineteen ninety-nine" in English)
- Decimals: "3.14" → "three point one four" (in {lang_name})
- Currency: "$99.99" → spoken currency in {lang_name}
- Percentages: "15%" → spoken percent in {lang_name}
- Time: "3:45 PM" → spoken time in {lang_name}

Abbreviations & Acronyms:
- Titles: expand fully (Dr. → Doctor, Prof. → Professor, Mr. → Mister, etc.)
- "etc." → et cetera, "vs." → versus, "e.g." → for example, "i.e." → that is
- Acronyms spelled letter-by-letter: add spaces (FBI → F B I, CEO → C E O)
- Widely-known pronounceable acronyms: keep as-is (NASA, UNICEF)

Internet slang (expand to full words in {lang_name}):
- LOL → laughing out loud, OMG → oh my goodness, IDK → I don't know
- BTW → by the way, FYI → for your information, ASAP → as soon as possible
- BRB → be right back, IMO → in my opinion, TBH → to be honest
- SMH → shaking my head, NGL → not gonna lie

Symbols:
- "+" → plus, "=" → equals, "&" → and, "@" → at
- "-" (standalone, not hyphen) → minus

URLs:
- Short URLs: spell out (www.example.com → W W W dot example dot com)
- Long URLs: "the web address on screen"

PROSODY — add punctuation for natural ChatterBox TTS delivery:

CRITICAL REQUIREMENT - spoken_text MUST BE DIFFERENT from input:
- You MUST add (..) pauses within EVERY long sentence (15+ words)
- spoken_text that is identical to input text is INCORRECT and will be rejected
- The (..) pauses are MANDATORY for TTS breathing and preventing audio glitches

Prosody rules:
- Add (..) for breathing pauses every 15-20 words within long sentences (REQUIRED)
- Add (..) after introductory phrases, before conjunctions, between clauses
- Minimum: at least 2-3 (..) pauses per paragraph
- Keep existing sentence boundaries (. ! ?) intact — do NOT artificially break sentences
- Keep all existing ! and ? marks
- For emphasis on 1-2 key words: use CAPITALIZATION if appropriate for {lang_name}
- Use proper {lang_name} punctuation: {punctuation}
- Preserve ALL content — do NOT remove or omit any words or sentences
- Do NOT add rhetorical questions unless in original text

Example (MANDATORY format - note the (..) pauses added):
- Input:  "الدُّكْتُورُ أُوبْرَايِنُ قَرَأَ الْمَقَالَ الرَّئِيسِيَّ فِي السَّاعَةِ الثَّالِثَةِ وَخَمْسٍ وَأَرْبَعِينَ مَسَاءً عَنْ صَيْدِ السَّمَكِ"
- Output: "الدُّكْتُورُ أُوبْرَايِنُ قَرَأَ الْمَقَالَ الرَّئِيسِيَّ.. فِي السَّاعَةِ الثَّالِثَةِ وَخَمْسٍ وَأَرْبَعِينَ مَسَاءً.. عَنْ صَيْدِ السَّمَكِ." """

        user_prompt = f"""Text to analyze and optimize:

<<<
{text}
>>>

REMINDER: You MUST add (..) pauses to spoken_text. Do NOT return identical text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,  # Higher temp to encourage prosody changes
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            self._validate_result(result)

            # Clean up spacing artifacts
            result["spoken_text"] = self._cleanup_text_spacing(result["spoken_text"])

            return result

        except Exception as e:
            print(f"[OpenAI Analysis Error] {e}")
            return {
                "gender": "female",
                "emotion": "neutral",
                "spoken_text": text
            }

    def _validate_result(self, result: Dict[str, str]) -> None:
        """Validate OpenAI response format."""
        required_keys = {"gender", "emotion", "spoken_text"}
        if not required_keys.issubset(result.keys()):
            raise ValueError(f"Missing required keys. Got: {result.keys()}")

        if result["gender"] not in ["male", "female"]:
            raise ValueError(f"Invalid gender: {result['gender']}")

        if result["emotion"] not in SUPPORTED_EMOTIONS:
            print(f"[Warning] Unexpected emotion '{result['emotion']}', defaulting to 'neutral'")
            result["emotion"] = "neutral"

    def _cleanup_text_spacing(self, text: str) -> str:
        """
        Clean up unnecessary spaces in text for better TTS delivery.
        """
        # Remove multiple consecutive spaces → single space
        text = re.sub(r' {2,}', ' ', text)

        # Remove spaces before punctuation marks (English/Latin)
        text = re.sub(r'\s+([.,!?;:\)\]\}])', r'\1', text)

        # Chinese/Japanese punctuation
        text = re.sub(r'\s+([。！？，、])', r'\1', text)

        # Spanish opening punctuation (¿ ¡ should have no space after)
        text = re.sub(r'([¿¡])\s+', r'\1', text)

        # Arabic question mark
        text = re.sub(r'\s+؟', '؟', text)

        # Remove spaces after opening punctuation
        text = re.sub(r'([\(\[\{])\s+', r'\1', text)

        # Normalize double periods (..)
        text = re.sub(r'\s*\.\.\s*', '..', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Normalize multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def translate(self, text: str, target_language: str) -> str:
        """
        Translate English text to target language and normalize for TTS.

        Args:
            text: English text to translate
            target_language: Target language code (ar, es, ja, zh, etc.)

        Returns:
            Translated and TTS-normalized text

        Example:
            >>> analyzer = OpenAIVoiceAnalyzer()
            >>> result = analyzer.translate("Meet at 3:45 PM", "zh")
            >>> print(result)
            "在下午三点四十五分见面"
        """
        LANGUAGE_NAMES = {
            "ar": "Arabic",
            "da": "Danish",
            "de": "German",
            "el": "Greek",
            "en": "English",
            "es": "Spanish",
            "fi": "Finnish",
            "fr": "French",
            "he": "Hebrew",
            "hi": "Hindi",
            "it": "Italian",
            "ja": "Japanese",
            "ko": "Korean",
            "ms": "Malay",
            "nl": "Dutch",
            "no": "Norwegian",
            "pl": "Polish",
            "pt": "Portuguese",
            "ru": "Russian",
            "sv": "Swedish",
            "sw": "Swahili",
            "tr": "Turkish",
            "zh": "Chinese (Simplified)",
        }

        language_name = LANGUAGE_NAMES.get(target_language, target_language)

        system_prompt = f"""#Role: You are a professional translator specializing in text-to-speech (TTS) preparation.

#Task: Translate English text to {language_name} and prepare it for natural TTS delivery.

#Format: Return ONLY the translated, normalized, and prosody-optimized text. No explanations, no JSON, just the text.

#Context: The input text is in English and needs to be translated to {language_name}.

#Goal: Produce natural-sounding {language_name} text optimized for TTS pronunciation and pacing.

#Requirements/Constraints:

1. Translate English text to {language_name}
2. Normalize ALL numbers, dates, times, symbols, and abbreviations for natural TTS pronunciation
3. Add prosody optimization (pauses, emphasis, pacing) for natural speech delivery

CRITICAL TTS NORMALIZATION RULES for {language_name}:

Numbers:
- Write ALL numbers as words in {language_name}
- Years: "1999" → write in {language_name} words
- Times: "3:45 PM" → write in {language_name} words
- Money: "$99.99" → write amount in {language_name} words
- Percentages: "15%" → write in {language_name} words
- Decimals: "99.99" → write with decimal point in {language_name} words

Abbreviations:
- Expand ALL abbreviations: "Dr." → full word in {language_name}
- "WWW" → spell out in {language_name}
- "PM" → use appropriate {language_name} equivalent

Special symbols:
- "$" → write currency name in {language_name}
- "%" → write "percent" equivalent in {language_name}
- Keep punctuation appropriate for {language_name}

PROSODY OPTIMIZATION for natural speech:
- Add commas for natural pauses (after clauses, before conjunctions, between phrases)
- Keep sentences under 25 words for natural breathing
- Use periods to create definitive breaks
- Add ellipses (...) for contemplative pauses if appropriate
- For strong emphasis on 1-2 key words: use appropriate {language_name} emphasis markers or CAPITALIZATION if culturally appropriate
- Ensure proper sentence pacing and rhythm"""

        user_prompt = f"Translate to {language_name} and normalize for TTS:\n\n{text}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            translated_text = response.choices[0].message.content.strip()
            return translated_text

        except Exception as e:
            print(f"[Translation Error] {e}")
            return text  # Fallback to original text


def test_analyzer():
    """Test the OpenAI analyzer with sample inputs across multiple languages."""
    analyzer = OpenAIVoiceAnalyzer()

    test_cases = [
        ("en", "Dr. Smith will discuss the $99.99 budget at 3:45 PM. LOL!"),
        ("zh", "奥布莱恩博士在下午三点四十五分阅读了头条文章。哈哈！"),
        ("es", "El Doctor O'Brien leyó el artículo a las tres y cuarenta y cinco. ¡JAJA!"),
        ("en", "We need to act NOW or it's over..."),
        ("en", "I scored 87% on the test! That's amazing!"),
    ]

    print("=" * 60)
    print("OpenAI Voice Analyzer Test — analyze_and_optimize()")
    print("=" * 60)

    for i, (lang, text) in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Language: {lang}")
        print(f"  Input:   {text}")
        result = analyzer.analyze_and_optimize(text, lang)
        print(f"  Gender:  {result['gender']}")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Spoken:  {result['spoken_text']}")


if __name__ == "__main__":
    test_analyzer()
