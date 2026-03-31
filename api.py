
import base64
import json
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from translator import translate_to_english, translate_from_english, detect_language
from symptom_extractor import extract_symptoms
from feature_engineer import build_feature_vector
from stt_tts import speech_to_text_cloud, speech_to_text_local, text_to_speech_cloud, text_to_speech_local
from severity import load_severity, calculate_severity
from predictor import load_disease_map, predict_disease
from emergency import is_emergency

from datetime import datetime

from database import users_collection, history_collection
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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


severity_dict = load_severity()
disease_map = load_disease_map()

model_data = None
meta_data = None
disease_precautions = {}
symptom_descriptions = {}

TRANSLATION_CACHE: Dict[Tuple[str, str], str] = {}
MAX_TRANSLATION_WORKERS = 8

SYMPTOM_LABELS = {
    "headache": {"en": "headache", "hi": "सिरदर्द", "te": "తలనొప్పి", "pa": "ਸਿਰ ਦਰਦ", "kn": "ತಲೆನೋವು", "ta": "தலைவலி"},
    "high_fever": {"en": "high fever", "hi": "तेज बुखार", "te": "అధిక జ్వరం", "pa": "ਤੇਜ਼ ਬੁਖਾਰ", "kn": "ತೀವ್ರ ಜ್ವರ", "ta": "அதிக காய்ச்சல்"},
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
    "en": {"no_symptoms": "No symptoms detected.", "no_match": "No matching disease found. Please add more symptoms.", "multiple": "Multiple diseases match your symptoms. Please add more symptoms for accurate prediction.", "you_may_have": "You may have", "severity": "Severity", "low_msg": "Home care and rest may help.", "medium_msg": "Monitor symptoms and consult a doctor if they worsen.", "high_msg": "Consult a doctor.", "emergency": "Seek immediate medical attention!"},
    "hi": {"no_symptoms": "कोई लक्षण नहीं मिले।", "no_match": "कोई मिलती-जुलती बीमारी नहीं मिली। कृपया और लक्षण जोड़ें।", "multiple": "आपके लक्षण कई बीमारियों से मेल खाते हैं। कृपया अधिक सटीक परिणाम के लिए और लक्षण जोड़ें।", "you_may_have": "आपको", "severity": "गंभीरता", "low_msg": "घर पर आराम और देखभाल मदद कर सकती है।", "medium_msg": "लक्षणों पर नज़र रखें और बढ़ने पर डॉक्टर से सलाह लें।", "high_msg": "एक डॉक्टर से परामर्श करें।", "emergency": "तुरंत चिकित्सकीय सहायता लें!"},
    "te": {"no_symptoms": "ఎలాంటి లక్షణాలు గుర్తించబడలేదు.", "no_match": "సరిపోయే వ్యాధి కనిపించలేదు. దయచేసి మరిన్ని లక్షణాలు ఇవ్వండి.", "multiple": "మీ లక్షణాలు అనేక వ్యాధులతో సరిపోతున్నాయి. ఖచ్చితమైన ఫలితాల కోసం మరిన్ని లక్షణాలు ఇవ్వండి.", "you_may_have": "మీకు", "severity": "తీవ్రత", "low_msg": "ఇంటివద్ద విశ్రాంతి మరియు జాగ్రత్తలు ఉపయోగపడవచ్చు.", "medium_msg": "లక్షణాలను గమనించండి. ఎక్కువైతే వైద్యుడిని సంప్రదించండి.", "high_msg": "వైద్యుడిని సంప్రదించండి.", "emergency": "వెంటనే వైద్య సహాయం పొందండి!"},
    "pa": {"no_symptoms": "ਕੋਈ ਲੱਛਣ ਨਹੀਂ ਮਿਲੇ।", "no_match": "ਕੋਈ ਮਿਲਦੀ ਬਿਮਾਰੀ ਨਹੀਂ ਮਿਲੀ। ਕਿਰਪਾ ਕਰਕੇ ਹੋਰ ਲੱਛਣ ਦਿਓ।", "multiple": "ਤੁਹਾਡੇ ਲੱਛਣ ਕਈ ਬਿਮਾਰੀਆਂ ਨਾਲ ਮਿਲਦੇ ਹਨ। ਕਿਰਪਾ ਕਰਕੇ ਹੋਰ ਲੱਛਣ ਦਿਓ।", "you_may_have": "ਤੁਹਾਨੂੰ", "severity": "ਤੀਬਰਤਾ", "low_msg": "ਘਰ ਵਿੱਚ ਆਰਾਮ ਅਤੇ ਦੇਖਭਾਲ ਮਦਦ ਕਰ ਸਕਦੀ ਹੈ।", "medium_msg": "ਲੱਛਣਾਂ 'ਤੇ ਨਜ਼ਰ ਰੱਖੋ ਅਤੇ ਵੱਧਣ 'ਤੇ ਡਾਕਟਰ ਨਾਲ ਸਲਾਹ ਕਰੋ।", "high_msg": "ਡਾਕਟਰ ਨਾਲ ਸਲਾਹ ਕਰੋ।", "emergency": "ਤੁਰੰਤ ਡਾਕਟਰੀ ਮਦਦ ਲਵੋ!"},
    "kn": {"no_symptoms": "ಯಾವ ಲಕ್ಷಣಗಳೂ ಕಂಡುಬಂದಿಲ್ಲ.", "no_match": "ಹೊಂದುವ ರೋಗ ಕಂಡುಬಂದಿಲ್ಲ. ದಯವಿಟ್ಟು ಇನ್ನಷ್ಟು ಲಕ್ಷಣಗಳನ್ನು ಸೇರಿಸಿ.", "multiple": "ನಿಮ್ಮ ಲಕ್ಷಣಗಳು ಹಲವಾರು ರೋಗಗಳಿಗೆ ಹೊಂದಿಕೆಯಾಗುತ್ತಿವೆ. ಇನ್ನಷ್ಟು ಲಕ್ಷಣಗಳನ್ನು ನೀಡಿ.", "you_may_have": "ನಿಮಗೆ", "severity": "ತೀವ್ರತೆ", "low_msg": "ಮನೆ ವಿಶ್ರಾಂತಿ ಮತ್ತು ಜಾಗ್ರತೆ ಸಹಾಯವಾಗಬಹುದು.", "medium_msg": "ಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ. ಹೆಚ್ಚಾದರೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.", "high_msg": "ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.", "emergency": "ತಕ್ಷಣ ವೈದ್ಯಕೀಯ ಸಹಾಯ ಪಡೆಯಿರಿ!"},
    "ta": {"no_symptoms": "எந்த அறிகுறிகளும் கண்டறியப்படவில்லை.", "no_match": "பொருந்தும் நோய் எதுவும் கிடைக்கவில்லை. மேலும் அறிகுறிகள் அளிக்கவும்.", "multiple": "உங்கள் அறிகுறிகள் பல நோய்களுடன் பொருந்துகின்றன. மேலும் அறிகுறிகள் அளிக்கவும்.", "you_may_have": "உங்களுக்கு", "severity": "தீவிரம்", "low_msg": "வீட்டு ஓய்வு மற்றும் கவனம் உதவலாம்.", "medium_msg": "அறிகுறிகளை கவனியுங்கள். அதிகரித்தால் மருத்துவரை அணுகவும்.", "high_msg": "மருத்துவரை அணுகவும்.", "emergency": "உடனடி மருத்துவ உதவி பெறுங்கள்!"},
}

def t(lang: str, key: str) -> str:
    return STATIC_TEXT.get(lang, STATIC_TEXT["en"]).get(key, STATIC_TEXT["en"][key])

def cached_translate(text: str, lang: str) -> str:
    if not text or lang == "en":
        return text
    key = (text, lang)
    if key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[key]
    try:
        translated = translate_from_english(text, lang)
        TRANSLATION_CACHE[key] = translated if translated else text
    except Exception:
        TRANSLATION_CACHE[key] = text
    return TRANSLATION_CACHE[key]

def translate_many(texts: List[str], lang: str) -> List[str]:
    if lang == "en":
        return texts
    unique = []
    seen = set()
    for x in texts:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    with ThreadPoolExecutor(max_workers=min(MAX_TRANSLATION_WORKERS, max(1, len(unique)))) as ex:
        translated_unique = list(ex.map(lambda x: cached_translate(x, lang), unique))
    mapping = dict(zip(unique, translated_unique))
    return [mapping.get(x, x) for x in texts]

def localize_symptom(symptom_key: str, lang: str) -> str:
    if lang == "en":
        return symptom_key
    if symptom_key in SYMPTOM_LABELS and lang in SYMPTOM_LABELS[symptom_key]:
        return SYMPTOM_LABELS[symptom_key][lang]
    return cached_translate(symptom_key.replace("_", " "), lang)

def localize_symptoms(symptoms: List[str], lang: str) -> List[str]:
    return [localize_symptom(s, lang) for s in symptoms]

def localize_severity(severity_en: str, lang: str) -> str:
    return SEVERITY_LABELS.get(lang, SEVERITY_LABELS["en"]).get(severity_en, severity_en)

def localize_disease_name(disease: str, lang: str) -> str:
    return disease if lang == "en" else cached_translate(disease, lang)

def localize_precautions(precautions: List[str], lang: str) -> List[str]:
    return translate_many(precautions, lang)

def build_summary(disease_name_en: str, severity_en: str, lang: str, emergency_flag: bool) -> str:
    sev = localize_severity(severity_en, lang)
    disease_local = localize_disease_name(disease_name_en, lang)
    msg = t(lang, "low_msg") if severity_en == "Low" else (t(lang, "medium_msg") if severity_en == "Medium" else t(lang, "high_msg"))
    summary = f"{t(lang, 'you_may_have')} {disease_local}. {t(lang, 'severity')}: {sev}. {msg}"
    if emergency_flag:
        summary += f" 🚨 {t(lang, 'emergency')}"
    return summary

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

app = FastAPI(title="Symptoms Checker API", description="AI-powered symptom analysis and disease prediction", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def process_audio_input(audio_base64: str, language_code: str) -> str:
    try:
        audio_bytes = base64.b64decode(audio_base64)
        try:
            cloud_lang_code = {"en": "en-US", "hi": "hi-IN", "te": "te-IN", "pa": "pa-IN", "kn": "kn-IN", "ta": "ta-IN"}.get(language_code, "en-US")
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

def ml_predict(symptoms: List[str]) -> List[Dict[str, Any]]:
    if not model_data or "model" not in model_data:
        return []
    try:
        model = model_data["model"]
        class_names = model_data.get("class_names") or model_data.get("diseases") or []
        feature_vector = build_feature_vector(symptoms, "artifacts").reshape(1, -1)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_vector)[0]
            results = []
            for disease, prob in zip(class_names, probs):
                matched = sorted(list(set(symptoms) & disease_map.get(disease, set())))
                results.append({
                    "disease": disease,
                    "score": float(prob),
                    "matched": matched,
                    "match_count": len(matched),
                    "missing_count": len(set(symptoms) - disease_map.get(disease, set()))
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
    except Exception:
        pass
    return []

def choose_predictions(symptoms: List[str], pred_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not pred_results:
        return [], "no_match"

    full_matches = [p for p in pred_results if p["missing_count"] == 0]

    if len(full_matches) == 1:
        return full_matches[:1], None

    if len(full_matches) > 1:
        full_matches.sort(key=lambda x: (x["score"], x["disease_coverage"]), reverse=True)
        if len(symptoms) <= 2:
            return [], "multiple"
        return full_matches[:3], None

    if len(symptoms) <= 2:
        return [], "multiple"

    if pred_results[0]["score"] <= 0:
        return [], "no_match"

    return pred_results[:3], None

def generate_tts_audio(display_text: str, lang: str) -> Optional[str]:
    try:
        cloud_lang_code = {"en": "en-US", "hi": "hi-IN", "te": "te-IN", "pa": "pa-IN", "kn": "kn-IN", "ta": "ta-IN"}.get(lang, "en-US")
        try:
            _, audio_bytes = text_to_speech_cloud(display_text, cloud_lang_code)
        except Exception:
            _, audio_bytes, temp_path = text_to_speech_local(display_text, lang)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception:
        return None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_data is not None}

@app.get("/users")
def get_users():
    users = list(users_collection.find({}, {"_id": 0, "password": 0}))
    return users

@app.post("/register")
def register(user: dict):
    # ✅ CHECK IF USER EXISTS
    existing_user = users_collection.find_one({"email": user["email"]})

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # ✅ CLEAN PASSWORD
    password = user["password"].strip()

    if len(password) > 72:
        password = password[:72]

    hashed_password = pwd_context.hash(password)

    users_collection.insert_one({
        "name": user["name"],
        "email": user["email"],
        "password": hashed_password
    })

    return {"message": "User registered successfully"}

@app.post("/login")
def login(user: dict):
    db_user = users_collection.find_one({"email": user["email"]})

    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    password = user["password"].strip()

    if len(password) > 72:
        password = password[:72]

    if not pwd_context.verify(password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"message": "Login successful"}

@app.get("/history/{email}")
def get_history(email: str):
    try:
        data = list(history_collection.find(
            {"email": email},
            {"_id": 0}   # remove MongoDB id
        ))
        return data
    except Exception as e:
        return {"error": str(e)}

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

        detected_lang = detect_language(input_text_user_lang)
        source_lang = request.language if request.language != "en" else detected_lang
        input_text = input_text_user_lang if source_lang == "en" else translate_to_english(input_text_user_lang, source_lang)

        symptoms = extract_symptoms(input_text)
        symptoms_translated = localize_symptoms(symptoms, request.language)

        if not symptoms:
            display_text = t(request.language, "no_symptoms")
            tts_audio = generate_tts_audio(display_text, request.language) if request.mode == "voice" else None
            return PredictResponse(
                input_text=input_text,
                input_text_user_lang=input_text_user_lang,
                symptoms=[],
                predictions=[],
                predictions_translated=[],
                symptoms_translated=[],
                language=request.language,
                display_text=display_text,
                tts_audio_base64=tts_audio,
                debug={"detected_lang": detected_lang},
            )

        pred_results = predict_disease(symptoms, disease_map)
        ml_results = ml_predict(symptoms)
        selected, reason = choose_predictions(symptoms, pred_results)

        if reason == "no_match":
            display_text = t(request.language, "no_match")
            tts_audio = generate_tts_audio(display_text, request.language) if request.mode == "voice" else None
            return PredictResponse(
                input_text=input_text,
                input_text_user_lang=input_text_user_lang,
                symptoms=symptoms if request.language == "en" else symptoms_translated,
                predictions=[],
                predictions_translated=[],
                symptoms_translated=symptoms_translated if request.language != "en" else None,
                language=request.language,
                display_text=display_text,
                tts_audio_base64=tts_audio,
                debug={"detected_lang": detected_lang, "rule_top": pred_results[:5], "ml_top": ml_results[:5]},
            )

        if reason == "multiple":
            display_text = t(request.language, "multiple")
            tts_audio = generate_tts_audio(display_text, request.language) if request.mode == "voice" else None
            return PredictResponse(
                input_text=input_text,
                input_text_user_lang=input_text_user_lang,
                symptoms=symptoms if request.language == "en" else symptoms_translated,
                predictions=[],
                predictions_translated=[],
                symptoms_translated=symptoms_translated if request.language != "en" else None,
                language=request.language,
                display_text=display_text,
                tts_audio_base64=tts_audio,
                debug={"detected_lang": detected_lang, "rule_top": pred_results[:5], "ml_top": ml_results[:5]},
            )

        emergency_flag = is_emergency(symptoms)
        disease_predictions = []
        predictions_translated = []

        for item in selected:
            disease = item["disease"]
            matched = item.get("matched", [])
            precautions = disease_precautions.get(disease, [])
            severity_en = calculate_severity(matched, severity_dict)
            if emergency_flag and severity_en != "High":
                severity_en = "High"

            english_pred = DiseasePrediction(
                disease=disease,
                prob=float(item["score"]),
                severity=severity_en,
                precautions=precautions,
                symptom_descriptions={s: symptom_descriptions.get(s, "") for s in matched},
            )
            disease_predictions.append(english_pred)

            translated_pred = DiseasePrediction(
                disease=localize_disease_name(disease, request.language),
                prob=float(item["score"]),
                severity=localize_severity(severity_en, request.language),
                precautions=localize_precautions(precautions, request.language),
                symptom_descriptions={localize_symptom(k, request.language): cached_translate(v, request.language) if v else "" for k, v in english_pred.symptom_descriptions.items()},
            )
            predictions_translated.append(translated_pred)

        top_prediction = disease_predictions[0]
        display_text = build_summary(top_prediction.disease, top_prediction.severity, request.language, emergency_flag)
        tts_audio_base64 = generate_tts_audio(display_text, request.language) if request.mode == "voice" else None

        # ✅ Save history
        try:
            history_collection.insert_one({
                "email": request.email or "guest",
                "symptoms": symptoms,
                "prediction": top_prediction.disease,
                "severity": top_prediction.severity,
                "date": datetime.now()
            })
        except Exception as e:
            print("History save error:", e)

        return PredictResponse(
            input_text=input_text,
            input_text_user_lang=input_text_user_lang,
            symptoms=symptoms if request.language == "en" else symptoms_translated,
            predictions=disease_predictions,
            predictions_translated=predictions_translated if request.language != "en" else None,
            symptoms_translated=symptoms_translated if request.language != "en" else None,
            language=request.language,
            display_text=display_text,
            tts_audio_base64=tts_audio_base64,
            debug={
                "detected_lang": detected_lang,
                "used_rule_based_primary": True,
                "rule_top": pred_results[:5],
                "ml_top": ml_results[:5],
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def find_available_port(start_port=8000, max_port=8010):
    import socket
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