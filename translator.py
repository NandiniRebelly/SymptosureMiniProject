# """
# Translation module for Symptoms Checker project.
# Uses deep-translator for stable text translation.
# Supports English, Hindi, Telugu, and Punjabi.
# """

# import re
# from functools import lru_cache
# from deep_translator import GoogleTranslator

# SUPPORTED_LANGS = {
#     "en": "english",
#     "hi": "hindi",
#     "te": "telugu",
#     "pa": "punjabi",
# }


# class Translator:
#     def __init__(self):
#         self.supported_langs = SUPPORTED_LANGS

#     def detect_language(self, text: str) -> str:
#         """
#         Detect language using Unicode ranges.
#         """
#         if not text or not text.strip():
#             return "en"

#         text = text.strip()

#         # Telugu
#         if re.search(r"[\u0C00-\u0C7F]", text):
#             return "te"

#         # Hindi / Devanagari
#         if re.search(r"[\u0900-\u097F]", text):
#             return "hi"

#         # Punjabi / Gurmukhi
#         if re.search(r"[\u0A00-\u0A7F]", text):
#             return "pa"

#         return "en"

#     @lru_cache(maxsize=512)
#     def _translate_cached(self, text: str, source_lang: str, target_lang: str) -> str:
#         """
#         Cached translation for repeated UI text and summaries.
#         """
#         if not text or not text.strip():
#             return text

#         if source_lang == target_lang:
#             return text

#         try:
#             src = "auto" if source_lang in ("auto", "", None) else source_lang
#             translated = GoogleTranslator(source=src, target=target_lang).translate(text)
#             return translated if translated else text
#         except Exception:
#             return text

#     def translate_to_english(self, text: str, source_lang: str = "auto") -> str:
#         """
#         Translate input text to English.
#         """
#         if not text or not text.strip():
#             return text

#         if source_lang == "en":
#             return text

#         return self._translate_cached(text, source_lang, "en")

#     def translate_from_english(self, text: str, target_lang: str) -> str:
#         """
#         Translate English text to target language.
#         """
#         if not text or not text.strip():
#             return text

#         if target_lang == "en":
#             return text

#         return self._translate_cached(text, "en", target_lang)

#     def get_supported_languages(self):
#         return self.supported_langs


# # Global singleton
# _translator_instance = None


# def get_translator() -> Translator:
#     global _translator_instance
#     if _translator_instance is None:
#         _translator_instance = Translator()
#         print("✓ deep-translator initialized")
#     return _translator_instance


# def detect_language(text: str) -> str:
#     return get_translator().detect_language(text)


# def translate_to_english(text: str, source_lang: str = "auto") -> str:
#     return get_translator().translate_to_english(text, source_lang)


# def translate_from_english(text: str, target_lang: str) -> str:
#     return get_translator().translate_from_english(text, target_lang)


# if __name__ == "__main__":
#     samples = [
#         "I have fever and headache",
#         "मुझे बुखार और सिर दर्द है",
#         "నాకు జ్వరం మరియు తలనొప్పి ఉంది",
#         "ਮੈਨੂੰ ਬੁਖਾਰ ਤੇ ਸਿਰ ਦਰਦ ਹੈ",
#     ]

#     tr = get_translator()
#     for s in samples:
#         lang = tr.detect_language(s)
#         eng = tr.translate_to_english(s, lang)
#         print(f"Original: {s}")
#         print(f"Detected: {lang}")
#         print(f"English: {eng}")
#         print("-" * 40)

"""
Translation module for Symptoms Checker project.
Uses deep-translator for stable text translation.
Supports English, Hindi, Telugu, Punjabi, Kannada, and Tamil.
"""

import re
from functools import lru_cache
from deep_translator import GoogleTranslator

SUPPORTED_LANGS = {
    "en": "english",
    "hi": "hindi",
    "te": "telugu",
    "pa": "punjabi",
    "kn": "kannada",
    "ta": "tamil",
}


class Translator:
    def __init__(self):
        self.supported_langs = SUPPORTED_LANGS

    def detect_language(self, text: str) -> str:
        if not text or not text.strip():
            return "en"

        text = text.strip()

        # Telugu
        if re.search(r"[\u0C00-\u0C7F]", text):
            return "te"

        # Hindi / Devanagari
        if re.search(r"[\u0900-\u097F]", text):
            return "hi"

        # Punjabi / Gurmukhi
        if re.search(r"[\u0A00-\u0A7F]", text):
            return "pa"

        # Kannada
        if re.search(r"[\u0C80-\u0CFF]", text):
            return "kn"

        # Tamil
        if re.search(r"[\u0B80-\u0BFF]", text):
            return "ta"

        return "en"

    @lru_cache(maxsize=1024)
    def _translate_cached(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text or not text.strip():
            return text

        if source_lang == target_lang:
            return text

        try:
            src = "auto" if source_lang in ("auto", "", None) else source_lang
            translated = GoogleTranslator(source=src, target=target_lang).translate(text)
            return translated if translated else text
        except Exception:
            return text

    def translate_to_english(self, text: str, source_lang: str = "auto") -> str:
        if not text or not text.strip():
            return text

        if source_lang == "en":
            return text

        return self._translate_cached(text, source_lang, "en")

    def translate_from_english(self, text: str, target_lang: str) -> str:
        if not text or not text.strip():
            return text

        if target_lang == "en":
            return text

        return self._translate_cached(text, "en", target_lang)

    def get_supported_languages(self):
        return self.supported_langs


_translator_instance = None


def get_translator() -> Translator:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = Translator()
        print("✓ deep-translator initialized")
    return _translator_instance


def detect_language(text: str) -> str:
    return get_translator().detect_language(text)


def translate_to_english(text: str, source_lang: str = "auto") -> str:
    return get_translator().translate_to_english(text, source_lang)


def translate_from_english(text: str, target_lang: str) -> str:
    return get_translator().translate_from_english(text, target_lang)

