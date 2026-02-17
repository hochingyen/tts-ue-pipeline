"""
Language Configuration Module for ChatterBox TTS Pipeline

This module provides language detection, mapping, and voice prompt configuration
for all 23 languages supported by ChatterBox TTS.

Functions:
    get_language_name(lang_code): Convert language code to full name
    get_default_voice(lang_code): Get default voice URL for a language
    is_supported(lang_code): Check if language is supported

Constants:
    SUPPORTED_LANGUAGES: Dict mapping language codes to full names
    DEFAULT_VOICE_PROMPTS: Dict mapping language codes to voice sample URLs
    LANGDETECT_TO_CHATTERBOX: Dict mapping langdetect codes to ChatterBox codes
"""

# ChatterBox supported languages (23 total)
# Maps ISO 639-1 language codes to full language names
SUPPORTED_LANGUAGES = {
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
    "zh": "Chinese"
}

# Default voice prompts for each language (from official ChatterBox demo)
# These are reference audio samples used for zero-shot voice cloning
# Each URL points to a 3-12 second FLAC audio file hosted by Resemble AI
DEFAULT_VOICE_PROMPTS = {
    "ar": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
    "da": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
    "de": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
    "el": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
    "en": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
    "es": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
    "fi": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
    "fr": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
    "he": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
    "hi": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
    "it": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
    "ja": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
    "ko": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
    "ms": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
    "nl": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
    "no": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
    "pl": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
    "pt": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
    "ru": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
    "sv": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
    "sw": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
    "tr": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
    "zh": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac"
}

# Language code mapping from langdetect library to ChatterBox TTS
# langdetect uses slightly different codes for some languages (e.g., zh-cn, zh-tw)
# This mapping ensures compatibility between the detection library and TTS engine
LANGDETECT_TO_CHATTERBOX = {
    "ar": "ar",
    "da": "da",
    "de": "de",
    "el": "el",
    "en": "en",
    "es": "es",
    "fi": "fi",
    "fr": "fr",
    "he": "he",
    "hi": "hi",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "ms": "ms",
    "nl": "nl",
    "no": "no",
    "pl": "pl",
    "pt": "pt",
    "ru": "ru",
    "sv": "sv",
    "sw": "sw",
    "tr": "tr",
    "zh-cn": "zh",
    "zh-tw": "zh"
}

def get_language_name(lang_code):
    """
    Get full language name from language code.

    Args:
        lang_code (str): ISO 639-1 language code (e.g., 'en', 'es', 'fr')

    Returns:
        str: Full language name (e.g., 'English', 'Spanish', 'French')
             Returns 'Unknown' if language code is not supported
    """
    return SUPPORTED_LANGUAGES.get(lang_code, "Unknown")

def get_default_voice(lang_code):
    """
    Get default voice prompt URL for a given language.

    Args:
        lang_code (str): ISO 639-1 language code (e.g., 'en', 'es', 'fr')

    Returns:
        str or None: URL to default voice sample FLAC file for the language,
                     or None if no default voice is available
    """
    return DEFAULT_VOICE_PROMPTS.get(lang_code)

def is_supported(lang_code):
    """
    Check if a language is supported by ChatterBox TTS.

    Args:
        lang_code (str): ISO 639-1 language code (e.g., 'en', 'es', 'fr')

    Returns:
        bool: True if language is supported, False otherwise
    """
    return lang_code in SUPPORTED_LANGUAGES
