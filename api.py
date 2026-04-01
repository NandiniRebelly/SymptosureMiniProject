import base64
import json
import os
import socket
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import asyncio
from cachetools import TTLCache, LRUCache

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from passlib.context import CryptContext

from translator import translate_to_english, translate_from_english, detect_language
from symptom_extractor import extract_symptoms
from feature_engineer import build_feature_vector
from stt_tts import (
    speech_to_text_cloud,
    speech_to_text_local,
    text_to_speech_cloud,
    text_to_speech_local,
)
from severity import load_severity, calculate_severity
from predictor import load_disease_map, predict_disease
from emergency import is_emergency
from database import users_collection, history_collection
from datetime import datetime

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# -----------------------------
# Optimized Caching Configuration
# -----------------------------
TRANSLATION_CACHE = LRUCache(maxsize=5000)
SYMPTOM_CACHE = LRUCache(maxsize=1000)
PREDICTION_CACHE = TTLCache(maxsize=200, ttl=300)
CHAT_CACHE = TTLCache(maxsize=500, ttl=600)
PREDICT_RESPONSE_CACHE = TTLCache(maxsize=500, ttl=600)
TTS_CACHE = TTLCache(maxsize=200, ttl=1800)

EXECUTOR = ThreadPoolExecutor(max_workers=12)
MAX_TRANSLATION_WORKERS = 12

import threading
_cache_lock = threading.Lock()

# -----------------------------
# Request / Response Models
# -----------------------------
class PredictRequest(BaseModel):
    input: str = Field(...)
    input_type: str = Field(...)
    language: str = Field(...)
    mode: str = Field(...)
    email: Optional[str] = "guest"


class DiseasePrediction(BaseModel):
    disease: str
    prob: float
    severity: str
    precautions: List[str]
    symptom_descriptions: Dict[str, str]
    matched_symptoms: List[str] = []
    emergency: bool = False


class PredictResponse(BaseModel):
    input_text: str
    input_text_user_lang: str
    symptoms: List[str]
    predictions: List[DiseasePrediction]
    predictions_translated: Optional[List[DiseasePrediction]] = None
    symptoms_translated: Optional[List[str]] = None
    language: str
    display_text: str
    tts_audio_base64: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    question: str
    language: str = "en"
    predicted_disease: str
    severity: str
    symptoms: List[str] = []
    precautions: List[str] = []
    symptom_descriptions: Dict[str, str] = {}
    emergency: bool = False


class ChatResponse(BaseModel):
    answer: str
    answer_localized: str
    disclaimer: str
    emergency: bool = False
    tts_audio_base64: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


class TTSResponse(BaseModel):
    tts_audio_base64: Optional[str] = None


# -----------------------------
# Globals
# -----------------------------
severity_dict = load_severity()
disease_map = load_disease_map()

model_data = None
meta_data = None
disease_precautions = {}
symptom_descriptions = {}

# Complete symptom labels with all languages
SYMPTOM_LABELS = {
    "headache": {"en": "headache", "hi": "सिरदर्द", "te": "తలనొప్పి", "pa": "ਸਿਰ ਦਰਦ", "kn": "ತಲೆನೋವು", "ta": "தலைவலி"},
    "high_fever": {"en": "high fever", "hi": "तेज बुखार", "te": "అధిక జ్వరం", "pa": "ਤੇਜ਼ ਬੁਖਾਰ", "kn": "ತೀವ್ರ ಜ್ವರ", "ta": "அதிக காய்ச்சல்"},
    "fever": {"en": "fever", "hi": "बुखार", "te": "జ్వరం", "pa": "ਬੁਖਾਰ", "kn": "ಜ್ವರ", "ta": "காய்ச்சல்"},
    "mild_fever": {"en": "mild fever", "hi": "हल्का बुखार", "te": "తేలికపాటి జ్వరం", "pa": "ਹਲਕਾ ਬੁਖਾਰ", "kn": "ಸ್ವಲ್ಪ ಜ್ವರ", "ta": "லேசான காய்ச்சல்"},
    "vomiting": {"en": "vomiting", "hi": "उल्टी", "te": "వాంతులు", "pa": "ਉਲਟੀ", "kn": "ವಾಂತಿ", "ta": "வாந்தி"},
    "nausea": {"en": "nausea", "hi": "मतली", "te": "వికారం", "pa": "ਮਤਲੀ", "kn": "ಛರ್ಡಿ ಭಾವನೆ", "ta": "குமட்டல்"},
    "cough": {"en": "cough", "hi": "खांसी", "te": "దగ్గు", "pa": "ਖੰਘ", "kn": "ಕೆಮ್ಮು", "ta": "இருமல்"},
    "dizziness": {"en": "dizziness", "hi": "चक्कर", "te": "తల తిరగడం", "pa": "ਚੱਕਰ", "kn": "ತಲೆ ಸುತ್ತು", "ta": "தலைசுற்றல்"},
    "skin_rash": {"en": "skin rash", "hi": "त्वचा पर चकत्ते", "te": "చర్మ దద్దుర్లు", "pa": "ਚਮੜੀ ਦੇ ਦਾਣੇ", "kn": "ಚರ್ಮದ ಉರಿ", "ta": "சருமச் சிரங்கு"},
    "itching": {"en": "itching", "hi": "खुजली", "te": "దురద", "pa": "ਖੁਜਲੀ", "kn": "ಖಜ್ಜಳಿ", "ta": "அரிப்பு"},
    "fatigue": {"en": "fatigue", "hi": "थकान", "te": "అలసట", "pa": "ਥਕਾਵਟ", "kn": "ದಣಿವು", "ta": "சோர்வு"},
    "back_pain": {"en": "back pain", "hi": "पीठ दर्द", "te": "నడుం నొప్పి", "pa": "ਪੀਠ ਦਰਦ", "kn": "ಬೆನ್ನು ನೋವು", "ta": "முதுகு வலி"},
    "neck_pain": {"en": "neck pain", "hi": "गर्दन दर्द", "te": "మెడ నొప్పి", "pa": "ਗਰਦਨ ਦਰਦ", "kn": "ಕತ್ತು ನೋವು", "ta": "கழுத்து வலி"},
    "chest_pain": {"en": "chest pain", "hi": "सीने में दर्द", "te": "ఛాతి నొప్పి", "pa": "ਸੀਨੇ ਵਿੱਚ ਦਰਦ", "kn": "ಛಾತಿ ನೋವು", "ta": "மார்பு வலி"},
    "breathlessness": {"en": "breathlessness", "hi": "सांस लेने में तकलीफ", "te": "శ్వాస ఇబ్బంది", "pa": "ਸਾਹ ਲੈਣ ਵਿੱਚ ਦਿੱਕਤ", "kn": "ಉಸಿರಾಟದ ತೊಂದರೆ", "ta": "மூச்சுத்திணறல்"},
    "runny_nose": {"en": "runny nose", "hi": "नाक बहना", "te": "ముక్కు కారడం", "pa": "ਨੱਕ ਵਗਣਾ", "kn": "ಮೂಗು ಹರಿಯುವುದು", "ta": "மூக்கு வடிதல்"},
    "congestion": {"en": "congestion", "hi": "नाक बंद", "te": "ముక్కు దిగబడటం", "pa": "ਨੱਕ ਬੰਦ", "kn": "ಮೂಗು ಮುಚ್ಚಿಕೊಳ್ಳುವುದು", "ta": "மூக்கு அடைப்பு"},
    "throat_irritation": {"en": "throat irritation", "hi": "गले में जलन", "te": "గొంతు రాపిడి", "pa": "ਗਲੇ ਵਿੱਚ ਜਲਣ", "kn": "ಗಂಟಲು ಕೆರಡು", "ta": "தொண்டை எரிச்சல்"},
    "sweating": {"en": "sweating", "hi": "पसीना", "te": "చెమటలు", "pa": "ਪਸੀਨਾ", "kn": "ಬೆವರು", "ta": "வியர்வை"},
    "weight_loss": {"en": "weight loss", "hi": "वजन कम होना", "te": "బరువు తగ్గడం", "pa": "ਵਜ਼ਨ ਘਟਣਾ", "kn": "ತೂಕ ಇಳಿಯುವುದು", "ta": "எடை குறைவு"},
    "loss_of_appetite": {"en": "loss of appetite", "hi": "भूख कम लगना", "te": "ఆకలి తగ్గడం", "pa": "ਭੁੱਖ ਘੱਟ ਲੱਗਣਾ", "kn": "ಭುಕ್ಕಿ ಕಡಿಮೆ", "ta": "பசியின்மை"},
    "abdominal_pain": {"en": "abdominal pain", "hi": "पेट दर्द", "te": "కడుపు నొప్పి", "pa": "ਪੇਟ ਦਰਦ", "kn": "ಹೊಟ್ಟೆ ನೋವು", "ta": "வயிற்று வலி"},
    "joint_pain": {"en": "joint pain", "hi": "जोड़ों में दर्द", "te": "సంధుల నొప్పి", "pa": "ਜੋੜ ਦਰਦ", "kn": "ಸಂಧಿ ನೋವು", "ta": "மூட்டு வலி"},
    "muscle_pain": {"en": "muscle pain", "hi": "मांसपेशियों में दर्द", "te": "కండరాల నొప్పి", "pa": "ਮਾਸਪੇਸ਼ੀ ਦਰਦ", "kn": "ಸ್ನಾಯು ನೋವು", "ta": "தசை வலி"},
    "chills": {"en": "chills", "hi": "कंपकंपी", "te": "చలి", "pa": "ਠੰਢ ਲੱਗਣਾ", "kn": "ಚಳಿ ಹಿಡಿಯುವುದು", "ta": "சளிச்சல்"},
    "insomnia": {"en": "insomnia", "hi": "अनिद्रा", "te": "నిద్రలేమి", "pa": "ਅਨੀਂਦਰਾ", "kn": "ನಿದ್ರಾಹೀನತೆ", "ta": "தூக்கமின்மை"},
    "weakness_of_one_body_side": {"en": "weakness of one body side", "hi": "शरीर के एक तरफ कमजोरी", "te": "ఒకవైపు బలహీనత", "pa": "ਸ਼ਰੀਰ ਦੇ ਇੱਕ ਪਾਸੇ ਕਮਜ਼ੋਰੀ", "kn": "ದೇಹದ ಒಂದು ಬದಿಯಲ್ಲಿ ದುರ್ಬಲತೆ", "ta": "உடலின் ஒரு பக்க பலவீனம்"},
}

SEVERITY_LABELS = {
    "en": {"Low": "Low", "Medium": "Medium", "High": "High"},
    "hi": {"Low": "कम", "Medium": "मध्यम", "High": "उच्च"},
    "te": {"Low": "తక్కువ", "Medium": "మధ్యస్థ", "High": "అధికం"},
    "pa": {"Low": "ਘੱਟ", "Medium": "ਦਰਮਿਆਨਾ", "High": "ਉੱਚ"},
    "kn": {"Low": "ಕಡಿಮೆ", "Medium": "ಮಧ್ಯಮ", "High": "ಹೆಚ್ಚು"},
    "ta": {"Low": "குறைவு", "Medium": "மிதமான", "High": "அதிகம்"},
}

STATIC_TEXT = {
    "en": {
        "no_symptoms": "No symptoms detected.",
        "no_match": "No matching disease found. Please add more symptoms.",
        "multiple": "Multiple diseases match your symptoms. Please add more symptoms for accurate prediction.",
        "you_may_have": "You may have",
        "severity": "Severity",
        "low_msg": "Home care and rest may help.",
        "medium_msg": "Monitor symptoms and consult a doctor if they worsen.",
        "high_msg": "Consult a doctor.",
        "emergency": "Seek immediate medical attention!",
        "chat_disclaimer": "This is general health guidance only and not a confirmed diagnosis or prescription."
    },
    "hi": {
        "no_symptoms": "कोई लक्षण नहीं मिले।",
        "no_match": "कोई मिलती-जुलती बीमारी नहीं मिली। कृपया और लक्षण जोड़ें।",
        "multiple": "आपके लक्षण कई बीमारियों से मेल खाते हैं। कृपया अधिक सटीक परिणाम के लिए और लक्षण जोड़ें।",
        "you_may_have": "आपको",
        "severity": "गंभीरता",
        "low_msg": "घर पर आराम और देखभाल मदद कर सकती है।",
        "medium_msg": "लक्षणों पर नज़र रखें और बढ़ने पर डॉक्टर से सलाह लें।",
        "high_msg": "एक डॉक्टर से परामर्श करें।",
        "emergency": "तुरंत चिकित्सकीय सहायता लें!",
        "chat_disclaimer": "यह केवल सामान्य स्वास्थ्य मार्गदर्शन है, पक्की जाँच या दवा पर्ची नहीं।"
    },
    "te": {
        "no_symptoms": "ఎలాంటి లక్షణాలు గుర్తించబడలేదు.",
        "no_match": "సరిపోయే వ్యాధి కనిపించలేదు. దయచేసి మరిన్ని లక్షణాలు ఇవ్వండి.",
        "multiple": "మీ లక్షణాలు అనేక వ్యాధులతో సరిపోతున్నాయి. ఖచ్చితమైన ఫలితాల కోసం మరిన్ని లక్షణాలు ఇవ్వండి.",
        "you_may_have": "మీకు",
        "severity": "తీవ్రత",
        "low_msg": "ఇంటివద్ద విశ్రాంతి మరియు జాగ్రత్తలు ఉపయోగపడవచ్చు.",
        "medium_msg": "లక్షణాలను గమనించండి. ఎక్కువైతే వైద్యుడిని సంప్రదించండి.",
        "high_msg": "వైద్యుడిని సంప్రదించండి.",
        "emergency": "వెంటనే వైద్య సహాయం పొందండి!",
        "chat_disclaimer": "ఇది సాధారణ ఆరోగ్య సూచన మాత్రమే. ఇది ఖచ్చిత నిర్ధారణ లేదా ప్రిస్క్రిప్షన్ కాదు."
    },
    "pa": {
        "no_symptoms": "ਕੋਈ ਲੱਛਣ ਨਹੀਂ ਮਿਲੇ।",
        "no_match": "ਕੋਈ ਮਿਲਦੀ ਬਿਮਾਰੀ ਨਹੀਂ ਮਿਲੀ। ਕਿਰਪਾ ਕਰਕੇ ਹੋਰ ਲੱਛਣ ਦਿਓ।",
        "multiple": "ਤੁਹਾਡੇ ਲੱਛਣ ਕਈ ਬਿਮਾਰੀਆਂ ਨਾਲ ਮਿਲਦੇ ਹਨ। ਕਿਰਪਾ ਕਰਕੇ ਹੋਰ ਲੱਛਣ ਦਿਓ।",
        "you_may_have": "ਤੁਹਾਨੂੰ",
        "severity": "ਤੀਬਰਤਾ",
        "low_msg": "ਘਰ ਵਿੱਚ ਆਰਾਮ ਅਤੇ ਦੇਖਭਾਲ ਮਦਦ ਕਰ ਸਕਦੀ ਹੈ।",
        "medium_msg": "ਲੱਛਣਾਂ 'ਤੇ ਨਜ਼ਰ ਰੱਖੋ ਅਤੇ ਵੱਧਣ 'ਤੇ ਡਾਕਟਰ ਨਾਲ ਸਲਾਹ ਕਰੋ।",
        "high_msg": "ਡਾਕਟਰ ਨਾਲ ਸਲਾਹ ਕਰੋ।",
        "emergency": "ਤੁਰੰਤ ਡਾਕਟਰੀ ਮਦਦ ਲਵੋ!",
        "chat_disclaimer": "ਇਹ ਸਿਰਫ਼ ਆਮ ਸਿਹਤ ਜਾਣਕਾਰੀ ਹੈ। ਇਹ ਪੱਕੀ ਜਾਂਚ ਜਾਂ ਦਵਾਈ ਦੀ ਪਰਚੀ ਨਹੀਂ ਹੈ।"
    },
    "kn": {
        "no_symptoms": "ಯಾವ ಲಕ್ಷಣಗಳೂ ಕಂಡುಬಂದಿಲ್ಲ.",
        "no_match": "ಹೊಂದುವ ರೋಗ ಕಂಡುಬಂದಿಲ್ಲ. ದಯವಿಟ್ಟು ಇನ್ನಷ್ಟು ಲಕ್ಷಣಗಳನ್ನು ಸೇರಿಸಿ.",
        "multiple": "ನಿಮ್ಮ ಲಕ್ಷಣಗಳು ಹಲವಾರು ರೋಗಗಳಿಗೆ ಹೊಂದಿಕೆಯಾಗುತ್ತಿವೆ. ಇನ್ನಷ್ಟು ಲಕ್ಷಣಗಳನ್ನು ನೀಡಿ.",
        "you_may_have": "ನಿಮಗೆ",
        "severity": "ತೀವ್ರತೆ",
        "low_msg": "ಮನೆ ವಿಶ್ರಾಂತಿ ಮತ್ತು ಜಾಗ್ರತೆ ಸಹಾಯವಾಗಬಹುದು.",
        "medium_msg": "ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ. ಹೆಚ್ಚಾದರೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "high_msg": "ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "emergency": "ತಕ್ಷಣ ವೈದ್ಯಕೀಯ ಸಹಾಯ ಪಡೆಯಿರಿ!",
        "chat_disclaimer": "ಇದು ಸಾಮಾನ್ಯ ಆರೋಗ್ಯ ಮಾರ್ಗದರ್ಶನ ಮಾತ್ರ. ಇದು ದೃಢೀಕೃತ ರೋಗನಿರ್ಣಯ ಅಥವಾ ಔಷಧ ಪಟ್ಟಿ ಅಲ್ಲ."
    },
    "ta": {
        "no_symptoms": "எந்த அறிகுறிகளும் கண்டறியப்படவில்லை.",
        "no_match": "பொருந்தும் நோய் எதுவும் கிடைக்கவில்லை. மேலும் அறிகுறிகள் அளிக்கவும்.",
        "multiple": "உங்கள் அறிகுறிகள் பல நோய்களுக்கு பொருந்துகின்றன. துல்லியமான முடிவுக்கு மேலும் அறிகுறிகள் அளிக்கவும்.",
        "you_may_have": "உங்களுக்கு",
        "severity": "தீவிரம்",
        "low_msg": "வீட்டில் ஓய்வு மற்றும் பராமரிப்பு உதவலாம்.",
        "medium_msg": "அறிகுறிகளை கவனிக்கவும். மோசமாயின் மருத்துவரை அணுகவும்.",
        "high_msg": "மருத்துவரை அணுகவும்.",
        "emergency": "உடனடி மருத்துவ உதவி பெறவும்!",
        "chat_disclaimer": "இது பொதுவான ஆரோக்கிய வழிகாட்டல் மட்டுமே. இது உறுதியான நோயறிதல் அல்லது மருந்து பரிந்துரை அல்ல."
    }
}


def t(lang: str, key: str) -> str:
    return STATIC_TEXT.get(lang, STATIC_TEXT["en"]).get(key, STATIC_TEXT["en"].get(key, key))


@lru_cache(maxsize=2000)
def cached_translate_from_english(text: str, lang: str) -> str:
    if not text or lang == "en":
        return text
    
    with _cache_lock:
        key = (text, lang)
        if key in TRANSLATION_CACHE:
            return TRANSLATION_CACHE[key]
    
    try:
        translated = translate_from_english(text, lang)
        result = translated if translated else text
        with _cache_lock:
            TRANSLATION_CACHE[key] = result
        return result
    except Exception:
        with _cache_lock:
            TRANSLATION_CACHE[key] = text
        return text


@lru_cache(maxsize=2000)
def cached_translate_to_english(text: str, lang: str) -> str:
    if not text or lang == "en":
        return text
    
    with _cache_lock:
        key = (text, lang)
        if key in TRANSLATION_CACHE:
            return TRANSLATION_CACHE[key]
    
    try:
        translated = translate_to_english(text, lang)
        result = translated if translated else text
        with _cache_lock:
            TRANSLATION_CACHE[key] = result
        return result
    except Exception:
        with _cache_lock:
            TRANSLATION_CACHE[key] = text
        return text


def translate_many_parallel(texts: List[str], lang: str) -> List[str]:
    if lang == "en" or not texts:
        return texts
    
    unique_texts = list(dict.fromkeys(texts))
    
    futures = {EXECUTOR.submit(cached_translate_from_english, text, lang): text for text in unique_texts}
    
    translation_map = {}
    for future in as_completed(futures):
        original = futures[future]
        try:
            translation_map[original] = future.result()
        except Exception:
            translation_map[original] = original
    
    return [translation_map.get(text, text) for text in texts]


def localize_symptom(symptom_key: str, lang: str) -> str:
    if lang == "en":
        return symptom_key
    if symptom_key in SYMPTOM_LABELS and lang in SYMPTOM_LABELS[symptom_key]:
        return SYMPTOM_LABELS[symptom_key][lang]
    return cached_translate_from_english(symptom_key.replace("_", " "), lang)


def localize_symptoms(symptoms: List[str], lang: str) -> List[str]:
    if lang == "en":
        return symptoms
    return [localize_symptom(s, lang) for s in symptoms]


def localize_severity(severity_en: str, lang: str) -> str:
    return SEVERITY_LABELS.get(lang, SEVERITY_LABELS["en"]).get(severity_en, severity_en)


def localize_disease_name(disease: str, lang: str) -> str:
    if lang == "en":
        return disease
    return cached_translate_from_english(disease, lang)


def localize_precautions(precautions: List[str], lang: str) -> List[str]:
    return translate_many_parallel(precautions, lang)


def build_summary(disease_name_en: str, severity_en: str, lang: str, emergency_flag: bool) -> str:
    sev = localize_severity(severity_en, lang)
    disease_local = localize_disease_name(disease_name_en, lang)
    msg = (
        t(lang, "low_msg")
        if severity_en == "Low"
        else (t(lang, "medium_msg") if severity_en == "Medium" else t(lang, "high_msg"))
    )
    summary = f"{t(lang, 'you_may_have')} {disease_local}. {t(lang, 'severity')}: {sev}. {msg}"
    if emergency_flag:
        summary += f" 🚨 {t(lang, 'emergency')}"
    return summary


def extract_audio_bytes(result) -> Optional[bytes]:
    if result is None:
        return None
    if isinstance(result, bytes):
        return result
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, bytes):
                return item
    return None


def load_artifacts():
    global model_data, meta_data, disease_precautions, symptom_descriptions
    artifacts_dir = Path("artifacts")
    model_path = artifacts_dir / "model.joblib"
    meta_path = artifacts_dir / "meta.json"
    precautions_path = artifacts_dir / "disease_precaution_map.json"
    descriptions_path = artifacts_dir / "symptom_description.json"

    if model_path.exists():
        model_data = joblib.load(model_path)
        print(f"✓ Loaded model from {model_path}")
    else:
        print("⚠️ model.joblib not found, using rule-based predictor")

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

    if precautions_path.exists():
        with open(precautions_path, "r", encoding="utf-8") as f:
            disease_precautions = json.load(f)

    if descriptions_path.exists():
        with open(descriptions_path, "r", encoding="utf-8") as f:
            symptom_descriptions = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    EXECUTOR.shutdown(wait=False)


app = FastAPI(
    title="Symptoms Checker API",
    description="AI-powered symptom analysis, disease prediction, and multilingual chatbot",
    version="8.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Audio Helpers
# -----------------------------
def process_audio_input(audio_base64: str, language_code: str) -> str:
    try:
        audio_bytes = base64.b64decode(audio_base64)
        try:
            cloud_lang_code = {
                "en": "en-US",
                "hi": "hi-IN",
                "te": "te-IN",
                "pa": "pa-IN",
                "kn": "kn-IN",
                "ta": "ta-IN",
            }.get(language_code, "en-US")
            return speech_to_text_cloud(audio_bytes, cloud_lang_code).strip()
        except Exception:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            try:
                return speech_to_text_local(temp_path).strip()
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")


def generate_tts_audio(text: str, language_code: str) -> Optional[str]:
    if not text:
        return None

    cache_key = (text.strip(), language_code)
    cached = TTS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cloud_lang_code = {
        "en": "en-US",
        "hi": "hi-IN",
        "te": "te-IN",
        "pa": "pa-IN",
        "kn": "kn-IN",
        "ta": "ta-IN",
    }.get(language_code, "en-US")

    try:
        result = text_to_speech_cloud(text, cloud_lang_code)
        audio_bytes = extract_audio_bytes(result)
        if audio_bytes:
            encoded = base64.b64encode(audio_bytes).decode("utf-8")
            TTS_CACHE[cache_key] = encoded
            return encoded
    except Exception:
        pass

    try:
        local_result = text_to_speech_local(text, language_code)
        audio_bytes = extract_audio_bytes(local_result)
        if audio_bytes:
            encoded = base64.b64encode(audio_bytes).decode("utf-8")
            TTS_CACHE[cache_key] = encoded
            return encoded
    except TypeError:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_path = temp_file.name
            try:
                local_result = text_to_speech_local(text, temp_path, language_code)
                audio_bytes = extract_audio_bytes(local_result)
                if not audio_bytes and os.path.exists(temp_path):
                    with open(temp_path, "rb") as f:
                        audio_bytes = f.read()
                if audio_bytes:
                    encoded = base64.b64encode(audio_bytes).decode("utf-8")
                    TTS_CACHE[cache_key] = encoded
                    return encoded
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception:
            pass
    except Exception:
        pass

    return None



# -----------------------------
# Prediction Helpers
# -----------------------------
def ml_predict(symptoms: List[str]) -> List[Dict[str, Any]]:
    if not model_data:
        return []
    
    cache_key = tuple(sorted(symptoms))
    if cache_key in PREDICTION_CACHE:
        return PREDICTION_CACHE[cache_key]
    
    try:
        model = model_data["model"] if isinstance(model_data, dict) and "model" in model_data else model_data
        class_names = []
        if isinstance(model_data, dict):
            class_names = model_data.get("class_names") or model_data.get("diseases") or []

        feature_vector = build_feature_vector(symptoms, "artifacts").reshape(1, -1)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_vector)[0]
            if not class_names:
                class_names = [str(i) for i in range(len(probs))]
            results = [{"disease": class_names[i], "score": float(probs[i])} for i in range(len(probs))]
            results.sort(key=lambda x: x["score"], reverse=True)
            result = results[:5]
            PREDICTION_CACHE[cache_key] = result
            return result
    except Exception:
        pass
    return []


def choose_predictions(symptoms: List[str], pred_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    if not pred_results:
        return [], "no_match"

    symptom_count = len(symptoms)
    top_score = float(pred_results[0].get("score", 0.0))
    second_score = float(pred_results[1].get("score", 0.0)) if len(pred_results) > 1 else 0.0

    if symptom_count < 3:
        if len(pred_results) > 1 and abs(top_score - second_score) <= 0.12:
            return [], "multiple"
        if top_score < 0.60:
            return [], "multiple"
        return [pred_results[0]], "single"

    selected = pred_results[:3]
    if not selected:
        return [], "no_match"

    return selected, "multiple_predictions" if len(selected) > 1 else "single"


# -----------------------------
# Chatbot Helpers
# -----------------------------
def normalize_text(s: str) -> str:
    return (s or "").strip().lower()


@lru_cache(maxsize=1000)
def detect_chat_intent(question_en: str) -> str:
    q = normalize_text(question_en)

    keywords = {
        "vitamin_c_drinks": ["vitamin c", "citrus", "rich drinks", "orange juice", "lemon water", "mosambi", "amla"],
        "diet": ["what food", "which food", "what should i eat", "diet", "foods", "drink", "eat", "meal"],
        "medicine": ["medicine", "medication", "tablet", "drug", "syrup", "ointment", "painkiller", "antibiotic"],
        "recovery": ["recovery", "recover", "how long", "days", "weeks", "time to heal", "cure time"],
        "doctor": ["doctor", "hospital", "consult", "when to see", "when should", "emergency"],
        "severity": ["severity", "serious", "dangerous", "risk", "how bad"],
        "symptoms": ["symptom", "symptoms", "meaning", "what does", "what is"],
        "precautions": ["precaution", "precautions", "why", "care", "avoid", "what should i do", "what to do"],
        "summary": ["summary", "explain", "about disease", "tell me about", "what is this disease"],
    }
    
    for intent, words in keywords.items():
        if any(k in q for k in words):
            return intent
    return "general"


def build_medicine_text(disease: str, severity: str) -> str:
    base = (
        f"For {disease}, the exact medicine should depend on a doctor's evaluation, age, allergies, medical history, "
        "and confirmation of the diagnosis. Common treatment may include supportive medicines for symptom relief, "
        "but you should avoid taking prescription medicines without medical advice."
    )
    sev = normalize_text(severity)
    if sev == "high":
        base += " Since the severity is high, please consult a doctor rather than self-medicating."
    elif sev == "medium":
        base += " Because the severity is medium, a doctor's advice is recommended if symptoms persist or worsen."
    else:
        base += " For mild symptoms, rest, hydration, and basic supportive care may help, but monitor your symptoms."
    return base


def build_recovery_text(severity: str) -> str:
    sev = normalize_text(severity)
    if sev == "low":
        return "Recovery may be faster in mild cases, often within a few days to a couple of weeks depending on the condition and self-care."
    if sev == "medium":
        return "Recovery time can vary. Many moderate cases may improve over days to a few weeks, but it depends on the disease and whether symptoms are getting better."
    return "High-severity cases may take longer and should be medically evaluated. Recovery time depends on the confirmed diagnosis and treatment."


def build_doctor_text(severity: str, emergency_flag: bool) -> str:
    if emergency_flag:
        return "Because red-flag symptoms are present, you should seek immediate medical attention or go to the nearest hospital."

    sev = normalize_text(severity)
    if sev == "high":
        return "You should consult a doctor as soon as possible because the severity is high."
    if sev == "medium":
        return "You should consult a doctor if symptoms worsen, do not improve, or interfere with daily activities."
    return "For low severity, home care may help, but consult a doctor if symptoms persist, worsen, or new symptoms appear."


def build_precaution_text(question_en: str, precautions: List[str], disease: str) -> str:
    if not precautions:
        return (
            f"The suggested precautions are meant to reduce discomfort, prevent worsening, and support recovery for {disease}. "
            "Follow general care advice and monitor symptoms."
        )
    
    explanations = {
        "consult nearest hospital": "This is suggested because the condition may become serious and needs timely medical evaluation.",
        "consult doctor": "This helps confirm the diagnosis and start the correct treatment.",
        "keep hydrated": "Hydration helps prevent weakness and supports recovery.",
        "avoid oily food": "Oily food may worsen stomach discomfort, nausea, or digestion problems.",
        "avoid non veg food": "This is sometimes suggested because light, simple food can be easier to digest during illness.",
        "keep mosquitos out": "This helps prevent further mosquito bites and reduces the spread of mosquito-borne illness.",
        "keep mosquitos away": "This helps prevent further mosquito bites and reduces the spread of mosquito-borne illness.",
        "drink papaya leaf juice": "Some people use this as supportive care in dengue, but it should not replace medical advice.",
        "drink vitamin c rich drinks": "Vitamin C rich drinks can include orange juice, lemon water, mosambi juice, and amla juice.",
        "take vapour": "Steam or vapour may help ease nasal congestion and make breathing feel more comfortable.",
        "avoid cold food": "Cold food may increase throat irritation or discomfort in some people.",
        "get away from trigger": "Avoiding triggers can reduce worsening of breathing or allergy-related symptoms.",
        "take deep breaths": "Slow deep breathing may help calm breathing discomfort and improve airflow.",
        "switch to loose cloothing": "Loose clothing may reduce chest tightness and make breathing easier.",
        "seek help": "This means you should get medical attention if symptoms are severe or worsening."
    }
    
    q = normalize_text(question_en)
    for item in precautions:
        item_norm = normalize_text(item)
        if item_norm in q:
            if item_norm in explanations:
                return explanations[item_norm]
            return f'"{item}" is suggested to help reduce symptoms, prevent worsening, or support recovery.'
    
    lines = [f"These precautions are suggested to help manage {disease} safely and support recovery:"]
    for item in precautions[:3]:
        lines.append(f"- {item}: this may help reduce symptoms, prevent worsening, or support healing.")
    lines.append("If symptoms continue, worsen, or new serious symptoms appear, please consult a doctor.")
    return "\n".join(lines)


def build_diet_text(question_en: str, severity: str, precautions: List[str]) -> str:
    q = normalize_text(question_en)

    if "vitamin c" in q or "rich drinks" in q:
        return "Vitamin C rich drinks include orange juice, lemon water, mosambi juice, amla juice, kiwi juice, and diluted guava juice. These can help with hydration, but do not replace medical treatment."

    if any(word in q for word in ["what food", "which food", "what should i eat"]):
        if any("avoid oily food" in normalize_text(p) for p in precautions):
            return "You can prefer light and easy-to-digest food such as soups, rice, khichdi, fruits, coconut water, lemon water, and plenty of fluids. Since oily food is being avoided, keep meals simple and less spicy."
        return "A light, balanced diet and good hydration are usually helpful. You can prefer soups, fruits, rice, khichdi, curd rice if tolerated, coconut water, and plenty of fluids."

    if "avoid" in q:
        return "Avoid very oily, very spicy, heavy, or irritating foods if they worsen your symptoms. Also avoid anything that triggers cough, nausea, or stomach discomfort."

    base = "A light, balanced diet and good hydration are usually helpful. Prefer simple meals, fruits, soups, coconut water, lemon water, and plenty of fluids."
    if normalize_text(severity) == "high":
        base += " Since your severity is high, doctor-guided advice is better than self-management alone."
    return base


def build_symptom_text(symptoms: List[str], symptom_desc: Dict[str, str]) -> str:
    if not symptoms:
        return "No specific symptoms were provided for explanation."

    lines = ["Here is what the detected symptoms indicate in general:"]
    for s in symptoms[:3]:
        key = s.replace("_", " ").title()
        desc = symptom_desc.get(s) or symptom_desc.get(key) or f"{key} is one of the symptoms detected in your input."
        lines.append(f"- {key}: {desc}")
    if len(symptoms) > 3:
        lines.append(f"... and {len(symptoms) - 3} more symptoms.")
    return "\n".join(lines)


def build_severity_text(severity: str, emergency_flag: bool) -> str:
    text = f"The current severity level is {severity}. "
    sev = normalize_text(severity)

    if sev == "low":
        text += "This suggests milder symptom intensity, but you should still monitor your condition."
    elif sev == "medium":
        text += "This suggests a moderate level of concern and symptoms should be monitored carefully."
    else:
        text += "This suggests a higher level of concern and medical consultation is recommended."

    if emergency_flag:
        text += " Red-flag symptoms are present, so seek immediate medical attention."

    return text


def build_general_text(disease: str, severity: str, precautions: List[str], symptoms: List[str], emergency_flag: bool) -> str:
    symptom_text = ", ".join([s.replace("_", " ") for s in symptoms[:3]]) if symptoms else "the detected symptoms"
    if len(symptoms) > 3:
        symptom_text += f" and {len(symptoms) - 3} more"
    
    response = (
        f"Based on {symptom_text}, the current likely condition shown is {disease} with {severity} severity. "
        f"General precautions include: {', '.join(precautions[:3]) if precautions else 'follow basic care, rest, hydration, and monitoring'}."
    )
    if emergency_flag:
        response += " Because emergency symptoms are present, please seek immediate medical attention."
    else:
        response += " If symptoms do not improve or become worse, consult a doctor."
    return response


def generate_chat_answer_en(request: ChatRequest, question_en: str) -> str:
    cache_key = (question_en, request.predicted_disease, request.severity, tuple(request.symptoms), request.emergency)
    if cache_key in CHAT_CACHE:
        return CHAT_CACHE[cache_key]
    
    disease = request.predicted_disease.strip()
    severity = request.severity.strip() or "Low"
    symptoms = request.symptoms or []
    precautions = request.precautions or []
    symptom_desc = request.symptom_descriptions or {}
    emergency_flag = bool(request.emergency)

    intent = detect_chat_intent(question_en)

    if intent == "medicine":
        answer = build_medicine_text(disease, severity)
    elif intent == "precautions":
        answer = build_precaution_text(question_en, precautions, disease)
    elif intent == "recovery":
        answer = f"For {disease}, {build_recovery_text(severity)} Recovery also depends on rest, hydration, treatment, and whether the diagnosis is confirmed by a doctor."
    elif intent in {"diet", "vitamin_c_drinks"}:
        answer = build_diet_text(question_en, severity, precautions)
    elif intent == "doctor":
        answer = build_doctor_text(severity, emergency_flag)
    elif intent == "symptoms":
        answer = build_symptom_text(symptoms, symptom_desc)
    elif intent == "severity":
        answer = build_severity_text(severity, emergency_flag)
    elif intent == "summary":
        answer = (
            f"{disease} is the current likely prediction from the entered symptoms. "
            f"The severity is {severity}. Suggested precautions are: "
            f"{', '.join(precautions[:3]) if precautions else 'general supportive care and monitoring'}. "
            f"{'Please seek immediate medical help.' if emergency_flag else 'Consult a doctor if symptoms worsen or continue.'}"
        )
    else:
        answer = build_general_text(disease, severity, precautions, symptoms, emergency_flag)
    
    CHAT_CACHE[cache_key] = answer
    return answer


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_data is not None}


@app.get("/users")
def get_users():
    users = list(users_collection.find({}, {"_id": 0, "password": 0}))
    return users


@app.post("/register")
def register(user: dict):
    name = (user.get("name") or "").strip()
    email = (user.get("email") or "").strip().lower()
    password = (user.get("password") or "").strip()

    if not name or not email or not password:
        raise HTTPException(status_code=400, detail="Name, email, and password are required")

    existing_user = users_collection.find_one({"email": email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    if len(password) > 72:
        password = password[:72]

    hashed_password = pwd_context.hash(password)

    users_collection.insert_one({
        "name": name,
        "email": email,
        "password": hashed_password
    })

    return {"message": "User registered successfully"}


@app.post("/login")
def login(user: dict):
    email = (user.get("email") or "").strip().lower()
    password = (user.get("password") or "").strip()

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    db_user = users_collection.find_one({"email": email})
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if len(password) > 72:
        password = password[:72]

    if not pwd_context.verify(password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"message": "Login successful", "name": db_user.get("name", "")}


@app.get("/history/{email}")
def get_history(email: str):
    try:
        data = list(history_collection.find({"email": email}, {"_id": 0}).sort("date", -1))
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


def build_predict_cache_key(input_text_user_lang: str, language: str, input_type: str):
    return ((input_text_user_lang or "").strip().lower(), language, input_type)


def base_predict_payload(input_text: str, input_text_user_lang: str, symptoms: List[str], request_language: str, display_text: str, detected_lang: str, reason: str = "") -> Dict[str, Any]:
    symptoms_translated = symptoms if request_language == "en" else localize_symptoms(symptoms, request_language)
    return {
        "input_text": input_text,
        "input_text_user_lang": input_text_user_lang,
        "symptoms": symptoms if request_language == "en" else symptoms_translated,
        "predictions": [],
        "predictions_translated": [],
        "symptoms_translated": symptoms_translated if request_language != "en" else None,
        "language": request_language,
        "display_text": display_text,
        "tts_audio_base64": None,
        "debug": {
            "detected_lang": detected_lang,
            "reason": reason,
            "symptoms_found": symptoms,
        },
    }


def build_chat_cache_key(request: ChatRequest, question_en: str):
    return (
        question_en.strip().lower(),
        request.language,
        request.predicted_disease,
        request.severity,
        tuple(request.symptoms or []),
        tuple(request.precautions or []),
        bool(request.emergency),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        if request.input_type not in ["text", "audio"]:
            raise HTTPException(status_code=400, detail="input_type must be 'text' or 'audio'")
        if request.mode not in ["text", "voice"]:
            raise HTTPException(status_code=400, detail="mode must be 'text' or 'voice'")

        input_text_user_lang = process_audio_input(request.input, request.language) if request.input_type == "audio" else request.input
        if not input_text_user_lang.strip():
            raise HTTPException(status_code=400, detail="No text found in input")

        detected_lang = request.language or "en"
        cache_key = build_predict_cache_key(input_text_user_lang, request.language, request.input_type)

        if request.mode == "text":
            cached_response = PREDICT_RESPONSE_CACHE.get(cache_key)
            if cached_response is not None:
                try:
                    history_collection.insert_one({
                        "email": (request.email or "guest").strip().lower(),
                        "symptoms": cached_response.get("debug", {}).get("symptoms_found", []),
                        "prediction": cached_response["predictions"][0].disease if cached_response.get("predictions") else "",
                        "severity": cached_response["predictions"][0].severity if cached_response.get("predictions") else "",
                        "display_text": cached_response.get("display_text", ""),
                        "date": datetime.now(),
                    })
                except Exception as history_error:
                    print(f"History save error: {history_error}")
                return PredictResponse(**cached_response)

        input_text = cached_translate_to_english(input_text_user_lang, request.language) if request.language != "en" else input_text_user_lang
        symptoms = extract_symptoms(input_text)

        if not symptoms:
            payload = base_predict_payload(input_text, input_text_user_lang, [], request.language, t(request.language, "no_symptoms"), detected_lang)
            if request.mode == "voice":
                payload["tts_audio_base64"] = generate_tts_audio(payload["display_text"], request.language)
            else:
                PREDICT_RESPONSE_CACHE[cache_key] = dict(payload)
            return PredictResponse(**payload)

        pred_results = predict_disease(symptoms, disease_map)
        selected, reason = choose_predictions(symptoms, pred_results)

        if reason in ["no_match", "multiple"]:
            payload = base_predict_payload(input_text, input_text_user_lang, symptoms, request.language, t(request.language, reason), detected_lang, reason)
            if request.mode == "voice":
                payload["tts_audio_base64"] = generate_tts_audio(payload["display_text"], request.language)
            else:
                PREDICT_RESPONSE_CACHE[cache_key] = dict(payload)
            return PredictResponse(**payload)

        emergency_flag = is_emergency(symptoms)
        disease_predictions = []
        predictions_translated = []
        symptoms_translated = symptoms if request.language == "en" else localize_symptoms(symptoms, request.language)

        for item in selected:
            disease = item["disease"]
            matched = item.get("matched", symptoms)
            precautions_en = disease_precautions.get(disease, [])
            severity_en = calculate_severity(matched, severity_dict)
            if emergency_flag and severity_en != "High":
                severity_en = "High"

            prob_percent = round(float(item.get("score", 0)) * 100, 1)
            symptom_desc_en = {s: symptom_descriptions.get(s, f"{s.replace('_', ' ').title()} detected.") for s in matched}
            disease_predictions.append(DiseasePrediction(
                disease=disease,
                prob=prob_percent,
                severity=severity_en,
                precautions=precautions_en,
                symptom_descriptions=symptom_desc_en,
                matched_symptoms=matched,
                emergency=emergency_flag,
            ))

            if request.language != "en":
                predictions_translated.append(DiseasePrediction(
                    disease=localize_disease_name(disease, request.language),
                    prob=prob_percent,
                    severity=localize_severity(severity_en, request.language),
                    precautions=localize_precautions(precautions_en, request.language),
                    symptom_descriptions={},
                    matched_symptoms=localize_symptoms(matched, request.language),
                    emergency=emergency_flag,
                ))

        display_text = build_summary(disease_predictions[0].disease, disease_predictions[0].severity, request.language, emergency_flag)
        response_payload = {
            "input_text": input_text,
            "input_text_user_lang": input_text_user_lang,
            "symptoms": symptoms if request.language == "en" else symptoms_translated,
            "predictions": disease_predictions,
            "predictions_translated": predictions_translated if request.language != "en" else None,
            "symptoms_translated": symptoms_translated if request.language != "en" else None,
            "language": request.language,
            "display_text": display_text,
            "tts_audio_base64": None,
            "debug": {
                "detected_lang": detected_lang,
                "reason": reason,
                "symptoms_found": symptoms,
            },
        }

        if request.mode == "voice":
            response_payload["tts_audio_base64"] = generate_tts_audio(display_text, request.language)
        else:
            PREDICT_RESPONSE_CACHE[cache_key] = dict(response_payload)

        try:
            history_collection.insert_one({
                "email": (request.email or "guest").strip().lower(),
                "symptoms": symptoms,
                "prediction": disease_predictions[0].disease if disease_predictions else "",
                "severity": disease_predictions[0].severity if disease_predictions else "",
                "display_text": display_text,
                "date": datetime.now(),
            })
        except Exception as history_error:
            print(f"History save error: {history_error}")

        return PredictResponse(**response_payload)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_followup(request: ChatRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question is required")

        question_raw = request.question.strip()
        question_en = question_raw if request.language == "en" else cached_translate_to_english(question_raw, request.language)
        cache_key = build_chat_cache_key(request, question_en)
        cached_response = CHAT_CACHE.get(cache_key)
        if cached_response is not None:
            return ChatResponse(**cached_response)

        answer_en = generate_chat_answer_en(request, question_en)
        answer_localized = answer_en if request.language == "en" else cached_translate_from_english(answer_en, request.language)
        response_payload = {
            "answer": answer_en,
            "answer_localized": answer_localized,
            "disclaimer": t(request.language, "chat_disclaimer"),
            "emergency": request.emergency,
            "tts_audio_base64": None,
        }
        CHAT_CACHE[cache_key] = dict(response_payload)
        return ChatResponse(**response_payload)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chatbot error: {e}")
        raise HTTPException(status_code=500, detail=f"Chatbot failed: {str(e)}")


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech_endpoint(request: TTSRequest):
    try:
        text = (request.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        return TTSResponse(tts_audio_base64=generate_tts_audio(text, request.language or "en"))
    except HTTPException:
        raise
    except Exception as e:
        print(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


def find_available_port(start_port=8000, max_port=8010):
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available port found between {start_port} and {max_port}")


if __name__ == "__main__":
    import uvicorn
    port = find_available_port()
    print(f"⚡ API running on http://localhost:{port}")
    print(f"📚 API Documentation: http://localhost:{port}/docs")
    print(f"❤️ Health Check: http://localhost:{port}/health")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=port)



