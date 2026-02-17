#!/usr/bin/env python3
"""
OpenAI Analyzer for Voice Asset Generation System

This module analyzes text to determine:
1. Speaker gender (male/female)
2. Dominant emotion (12 emotions)
3. Prosody-optimized spoken text

Designed for Unreal Engine voice asset generation.
"""

import os
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

    Returns:
        - gender: male or female
        - emotion: one of 12 supported emotions
        - spoken_text: prosody-optimized text for TTS
    """

    SYSTEM_PROMPT = """You are a voice director for a game dialogue system.

Tasks:
1. Decide the speaker gender: male or female
2. Decide the dominant emotion (choose exactly ONE from the list)
3. Rewrite the text for natural spoken delivery with prosody

Emotion list:
neutral, happy, sad, angry, fear, surprise, disgust,
excited, calm, confident, nervous, serious

TEXT TRANSFORMATION RULES

Punctuation & Pacing:
- Sentence length: Keep sentences under 25 words for natural breathing
- Commas: Add for natural pauses (after clauses, before conjunctions, after introductory phrases)
- Periods: Use to create definitive breaks; avoid run-on sentences
- Ellipses (...): Use sparingly for longer contemplative pauses
- Em dashes (—): Replace with commas or split into sentences
- Semicolons: Replace with periods (semicolons create awkward pauses)

Numbers & Measurements:
- Years: "1984" → "nineteen eighty-four"
- Large numbers: "1,234,567" → "one million, two hundred thirty-four thousand"
- Decimals: "3.14159" → "three point one four" (limit to 2-3 digits)
- Percentages: "45%" → "forty-five percent"
- Money: "$1,250" → "one thousand two hundred fifty dollars" OR "twelve fifty dollars" (contextual)
- Money with cents: "$99.99" → "ninety-nine dollars, ninety-nine cents" (use comma, not "and")
- Measurements: "5'3"" → "five feet three inches", "25kg" → "twenty-five kilograms"
- Dates: "12/16/2025" → "December sixteenth, twenty twenty-five"
- Time: "3:45 PM" → "three forty-five PM" (formal expression)
- Phone numbers: "(555) 123-4567" → speak each digit individually ("five five five, one two three, four five six seven")
- Addresses: "123 Main St" → "one twenty-three Main Street"

Abbreviations & Acronyms:
- Common abbreviations: Expand all (Dr. → Doctor, Prof. → Professor, etc. → et cetera, vs. → versus)
- Acronyms pronounced as words: Keep as-is if widely known (NASA, UNICEF)
- Acronyms spelled out: "FBI" → "F B I" (with spaces for letter-by-letter reading)
- Units: "5 km" → "five kilometers", "100 mL" → "one hundred milliliters"
- e.g./i.e. → "for example" / "that is"

Technical Content Handling:
- Mathematical equations: Reference them: "As shown in equation 3" (don't read complex equations)
- Code: Reference instead: "See the code example" (don't read code aloud)
- Symbols: "+" → "plus", "-" → "minus", "=" → "equals", "&" → "and", "@" → "at"
- URLs: Short URLs → spell out ("www dot example dot com"), Long URLs → "the web address on screen"

Emphasis & Expression:
- Strong emphasis: Use CAPITALIZATION sparingly for 1-2 key words maximum
- Example: "This is, ABSOLUTELY, critical"
- Avoid overuse of filler words ("um," "uh," "like," "you know")

What NOT to Do:
- Do NOT add fake enthusiasm unless contextually appropriate
- Do NOT use quotation marks for emphasis
- Do NOT attempt to read complex mathematical notation or code blocks
- Do NOT add rhetorical questions unless in original text
- Do NOT delete or omit any sentence from the input

Return JSON ONLY in this exact format:
{
  "gender": "male|female",
  "emotion": "emotion_name",
  "spoken_text": "..."
}

Examples:

Input: "The meeting is at 3:45 PM. Dr. Smith will discuss the $99.99 budget."
Output: {
  "gender": "male",
  "emotion": "neutral",
  "spoken_text": "The meeting is at three forty-five PM. Doctor Smith will discuss the ninety-nine dollars, ninety-nine cents budget."
}

Input: "I scored 87% on the test! That's amazing!"
Output: {
  "gender": "female",
  "emotion": "excited",
  "spoken_text": "I scored eighty-seven percent on the test! That's, AMAZING!"
}

Input: "We need to act NOW or it's over..."
Output: {
  "gender": "male",
  "emotion": "nervous",
  "spoken_text": "We need to act, NOW, or it's over..."
}
"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI analyzer.

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"

    def analyze(self, text: str) -> Dict[str, str]:
        """
        Analyze text and return gender, emotion, and spoken text.

        DEPRECATED: This method is deprecated and will be removed in a future version.
        Use optimize_prosody() instead for multilingual prosody optimization.
        Gender detection has been removed from the pipeline.

        Args:
            text: Raw input text to analyze

        Returns:
            Dictionary with keys: gender, emotion, spoken_text

        Example:
            >>> analyzer = OpenAIVoiceAnalyzer()
            >>> result = analyzer.analyze("This is critical!")
            >>> print(result)
            {
                "gender": "male",
                "emotion": "confident",
                "spoken_text": "This is, CRITICAL!"
            }
        """
        user_prompt = f"Text:\n<<<\n{text}\n>>>"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Validate response
            self._validate_result(result)

            return result

        except Exception as e:
            print(f"[OpenAI Analysis Error] {e}")
            # Fallback to neutral defaults
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

    def analyze_batch(self, texts: list[str]) -> list[Dict[str, str]]:
        """
        Analyze multiple texts (processes sequentially).

        Args:
            texts: List of raw input texts

        Returns:
            List of analysis results
        """
        return [self.analyze(text) for text in texts]

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
            "es": "Spanish",
            "ja": "Japanese",
            "zh": "Chinese (Simplified)",
            "fr": "French",
            "de": "German",
            "ko": "Korean",
            "pt": "Portuguese",
            "ru": "Russian",
            "it": "Italian"
        }

        language_name = LANGUAGE_NAMES.get(target_language, target_language)

        system_prompt = f"""You are a professional translator specializing in text-to-speech (TTS) preparation.

Your task:
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
- Ensure proper sentence pacing and rhythm

Examples for {language_name}:
- If Chinese: "1999" → "一九九九", "3:45 PM" → "下午三点四十五分", "$99.99" → "九十九点九九美元"
- If Spanish: "1999" → "mil novecientos noventa y nueve", "3:45 PM" → "tres y cuarenta y cinco de la tarde"
- If Japanese: "1999" → "千九百九十九", "3:45 PM" → "午後三時四十五分"
- If Arabic: "1999" → "ألف وتسعمائة وتسعة وتسعون", "3:45 PM" → "الثالثة وخمسة وأربعون مساءً"

Return ONLY the translated, normalized, and prosody-optimized text. No explanations, no JSON, just the text."""

        user_prompt = f"Translate to {language_name} and normalize for TTS:\n\n{text}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3  # Lower temperature for more consistent translations
            )

            translated_text = response.choices[0].message.content.strip()
            return translated_text

        except Exception as e:
            print(f"[Translation Error] {e}")
            return text  # Fallback to original text

    def _cleanup_text_spacing(self, text: str) -> str:
        """
        Clean up unnecessary spaces in text for better TTS delivery.

        Args:
            text: Input text with potentially extra spaces

        Returns:
            Cleaned text with normalized spacing
        """
        import re

        # 1. Remove multiple consecutive spaces → single space
        text = re.sub(r' {2,}', ' ', text)

        # 2. Remove spaces before punctuation marks
        # English/Arabic: . ! ? , ; : ) ] }
        text = re.sub(r'\s+([.,!?;:\)\]\}])', r'\1', text)

        # Chinese/Japanese: 。！？，、
        text = re.sub(r'\s+([。！？，、])', r'\1', text)

        # Spanish: ¿ ¡ (opening punctuation should have space before, not after)
        text = re.sub(r'([¿¡])\s+', r'\1', text)

        # Arabic: ؟ (Arabic question mark)
        text = re.sub(r'\s+؟', '؟', text)

        # 3. Remove spaces after opening punctuation
        text = re.sub(r'([\(\[\{])\s+', r'\1', text)

        # 4. Normalize whitespace around double periods (..)
        # Remove spaces before/after ..
        text = re.sub(r'\s*\.\.\s*', '..', text)

        # 5. Remove leading/trailing whitespace
        text = text.strip()

        # 6. Normalize line breaks (remove multiple consecutive newlines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def optimize_prosody(
        self,
        text: str,
        language: str = "en"
    ) -> str:
        """
        Optimize text prosody for natural speech delivery using OpenAI.
        Improves punctuation, phrasing, and sentence structure for better TTS output.

        Args:
            text: Input text (already translated if multilingual)
            language: Language code ('en', 'zh', 'ja', 'ar', 'es')

        Returns:
            Prosody-optimized text with better punctuation and phrasing

        Example:
            >>> analyzer = OpenAIVoiceAnalyzer()
            >>> text = "你好世界"
            >>> result = analyzer.optimize_prosody(text, "zh")
            >>> print(result)
            "你好。世界！"
        """
        LANGUAGE_INFO = {
            "en": {"name": "English", "punctuation": ".!?"},
            "zh": {"name": "Chinese (Simplified)", "punctuation": "。！？"},
            "ja": {"name": "Japanese", "punctuation": "。！？"},
            "ar": {"name": "Arabic", "punctuation": ".!?؟!"},
            "es": {"name": "Spanish", "punctuation": ".!?¡¿"},
            "fr": {"name": "French", "punctuation": ".!?"},
            "de": {"name": "German", "punctuation": ".!?"},
            "ko": {"name": "Korean", "punctuation": ".!?"},
            "pt": {"name": "Portuguese", "punctuation": ".!?"},
            "ru": {"name": "Russian", "punctuation": ".!?"},
            "it": {"name": "Italian", "punctuation": ".!?"}
        }

        lang_info = LANGUAGE_INFO.get(language, LANGUAGE_INFO["en"])
        lang_name = lang_info["name"]

        system_prompt = f"""#Role: You are a speech prosody expert for text-to-speech systems.

#Task: Add punctuation for natural pauses in {lang_name} text.

#Format: Return ONLY the text with improved punctuation (no explanations, no JSON, just the text).

#Context: The input text is ALREADY in {lang_name} and ALREADY has all numbers/abbreviations normalized correctly. You should NOT translate, normalize, or modify the content. You should ONLY add commas and periods for better TTS delivery.

#Goal: Add punctuation (commas, periods) so the TTS system delivers the text naturally with proper pauses and breathing.

#Requirements/Constraints:

CRITICAL - DO NOT MODIFY CONTENT:
- The text is ALREADY correctly normalized - DO NOT change any words
- Numbers are ALREADY written as words - keep them EXACTLY as they are
- Abbreviations are ALREADY expanded - keep them EXACTLY as they are
- DO NOT translate anything
- DO NOT remove anything (especially "هَاهَا!" or "¡JAJA!" or any exclamations)
- DO NOT add new words or change meanings
- ONLY add punctuation marks: double periods (..) for pauses and periods (.) for sentence breaks

PUNCTUATION RULES for natural pauses:

1. Sentence Breaking (15-20 words maximum):
   - If a sentence is longer than 15-20 words, break it into shorter sentences
   - Use periods (.) to separate complete thoughts
   - Each sentence must be a complete thought (subject + verb)
   - Example: "Very long sentence with many clauses" → "First part here. Second part here."

2. Double Period Placement (natural pauses):
   - Add (..) after introductory phrases
   - Add (..) between independent clauses
   - Add (..) before conjunctions (and, but, or, etc.)
   - Add (..) after transitional words
   - Add (..) between items in a list

3. Preserve Existing Punctuation:
   - Keep all existing periods (.), question marks (?), exclamation marks (!)
   - Keep existing (..) or commas if they're already there
   - Use proper {lang_name} punctuation: {lang_info['punctuation']}

4. Goal:
   - Create natural breathing points for TTS delivery
   - Prevent sentences from running together
   - Ensure complete sentence delivery without cutoff

CRITICAL REMINDER:
- DO NOT change any words - text is already correct!
- ONLY add (..) and periods (.) for pauses
- Use (..) for natural breathing pauses instead of commas
- Keep ALL original content including "هَاهَا!" and "¡JAJA!"
- The text is perfect - just needs better punctuation for TTS"""

        user_prompt = f"""Add punctuation (double periods and periods) to this {lang_name} text for natural TTS delivery.

IMPORTANT:
- Text is already correct - DO NOT change any words
- ONLY add (..) and periods (.) for natural pauses
- Use (..) for breathing pauses (instead of commas)
- Break sentences longer than 15-20 words into shorter sentences using periods
- Keep ALL content including "هَاهَا!" and "¡JAJA!"

Input text:
{text}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3  # Lower temperature for consistent pause placement
            )

            optimized_text = response.choices[0].message.content.strip()

            # Clean up unnecessary spaces
            optimized_text = self._cleanup_text_spacing(optimized_text)

            return optimized_text

        except Exception as e:
            print(f"[Prosody Optimization Error] {e}")
            # Fallback: Return original text
            return text


def test_analyzer():
    """Test the OpenAI analyzer with sample inputs."""
    analyzer = OpenAIVoiceAnalyzer()

    test_cases = [
        "This is absolutely critical and we need to act now!",
        "I don't know if I can do this...",
        "What an amazing discovery!",
        "Get out of here right now!",
        "I'm so sorry for your loss.",
    ]

    print("=" * 60)
    print("OpenAI Voice Analyzer Test")
    print("=" * 60)

    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Input: {text}")
        result = analyzer.analyze(text)
        print(f"Gender: {result['gender']}")
        print(f"Emotion: {result['emotion']}")
        print(f"Spoken: {result['spoken_text']}")


if __name__ == "__main__":
    test_analyzer()
