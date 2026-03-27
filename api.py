"""
FastAPI application for symptoms checker project.

Sample curl request:
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "input": "I have a fever and headache",
       "input_type": "text",
       "language": "en",
       "mode": "text"
     }'

For audio input:
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "input": "base64_encoded_audio_data",
       "input_type": "audio",
       "language": "hi",
       "mode": "voice"
     }'
"""

import base64
import json
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import our custom modules
from translator import translate_to_english, translate_from_english, detect_language
from symptom_extractor import extract_symptoms
from feature_engineer import build_feature_vector
from stt_tts import speech_to_text_cloud, speech_to_text_local, text_to_speech_cloud, text_to_speech_local
from disease_matcher import DiseaseMatcher

from severity import load_severity, calculate_severity
from predictor import load_disease_map, predict_disease
from emergency import is_emergency

# Configuration flags
USE_SCORER_ONLY = False  # Set to True to use only rule-based scoring, False for ensemble

# Pydantic models for request/response
class PredictRequest(BaseModel):
    input: str = Field(..., description="User text or base64-encoded audio data")
    input_type: str = Field(..., description="Type of input: 'text' or 'audio'")
    language: str = Field(..., description="Language code: 'en', 'hi', or 'pa'")
    mode: str = Field(..., description="Output mode: 'text' or 'voice'")


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
    language: str
    display_text: str
    tts_audio_base64: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


# Global variables for loaded artifacts
model_data = None
meta_data = None
symptom_severity = None
disease_precautions = None
symptom_descriptions = None
disease_matcher = None
#NEW FIELD
severity_dict = load_severity()
disease_map = load_disease_map()

def load_artifacts():
    """
    Load all required artifacts on startup.
    """
    global model_data, meta_data, symptom_severity, disease_precautions, symptom_descriptions, disease_matcher
    
    artifacts_dir = Path("artifacts")
    
    try:
        # Load trained model
        model_path = artifacts_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model_data = joblib.load(model_path)
        print(f"✓ Loaded model from {model_path}")
        
        # Load metadata
        meta_path = artifacts_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta data not found: {meta_path}")
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        print(f"✓ Loaded metadata from {meta_path}")
        
        # Load symptom severity mapping
        severity_path = artifacts_dir / "symptom_severity.json"
        if not severity_path.exists():
            raise FileNotFoundError(f"Symptom severity not found: {severity_path}")
        with open(severity_path, 'r', encoding='utf-8') as f:
            symptom_severity = json.load(f)
        print(f"✓ Loaded symptom severity from {severity_path}")
        
        # Load disease precautions mapping
        precautions_path = artifacts_dir / "disease_precaution_map.json"
        if not precautions_path.exists():
            raise FileNotFoundError(f"Disease precautions not found: {precautions_path}")
        with open(precautions_path, 'r', encoding='utf-8') as f:
            disease_precautions = json.load(f)
        print(f"✓ Loaded disease precautions from {precautions_path}")
        
        # Load symptom descriptions
        descriptions_path = artifacts_dir / "symptom_description.json"
        if not descriptions_path.exists():
            print(f"⚠️ Symptom descriptions not found: {descriptions_path}")
            symptom_descriptions = {}
        else:
            with open(descriptions_path, 'r', encoding='utf-8') as f:
                symptom_descriptions = json.load(f)
            print(f"✓ Loaded symptom descriptions from {descriptions_path}")
        
        # Initialize disease matcher
        try:
            disease_matcher = DiseaseMatcher(str(artifacts_dir))
            print(f"✓ Disease matcher initialized successfully")
        except Exception as e:
            print(f"⚠️ Disease matcher initialization failed: {e}")
            disease_matcher = None
        
        print(f"✓ All artifacts loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        raise


def process_audio_input(audio_base64: str, language_code: str) -> str:
    """
    Process audio input and return transcribed text.
    
    Args:
        audio_base64: Base64-encoded audio data
        language_code: Language code for transcription
        
    Returns:
        Transcribed text in English
    """
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Try cloud STT first, fallback to local
        try:
            # Convert language code for Google Cloud (e.g., 'hi' -> 'hi-IN')
            cloud_lang_code = f"{language_code}-IN" if language_code in ['hi', 'pa'] else f"{language_code}-US"
            transcribed_text = speech_to_text_cloud(audio_bytes, cloud_lang_code)
        except Exception:
            # Fallback to local STT (requires saving to temp file)
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            try:
                transcribed_text = speech_to_text_local(temp_file_path)
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        return transcribed_text.strip()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")


def calculate_disease_severity(symptoms: List[str], disease: str) -> str:
    """
    Calculate disease severity based on symptom weights.
    
    Args:
        symptoms: List of symptoms for the disease
        disease: Disease name
        
    Returns:
        Severity label: 'Low', 'Medium', 'High', or 'Critical'
    """
    if not symptoms or not symptom_severity:
        return "Low"
    
    # Get weights for symptoms
    weights = []
    for symptom in symptoms:
        if symptom in symptom_severity:
            weights.append(symptom_severity[symptom])
    
    if not weights:
        return "Low"
    
    # Use max aggregation (most severe symptom determines overall severity)
    max_weight = max(weights)
    
    # Map to severity labels using thresholds from meta.json
    severity_thresholds = meta_data.get('severity_thresholds', {
        'low': 0.0,
        'medium': 0.3,
        'high': 0.6,
        'critical': 0.8
    })
    
    if max_weight >= severity_thresholds['critical']:
        return "Critical"
    elif max_weight >= severity_thresholds['high']:
        return "High"
    elif max_weight >= severity_thresholds['medium']:
        return "Medium"
    else:
        return "Low"


def generate_display_text(predictions: List[DiseasePrediction], language: str) -> str:
    """
    Generate natural language summary of predictions.
    
    Args:
        predictions: List of disease predictions
        language: Target language for translation
        
    Returns:
        Translated display text
    """
    if not predictions:
        return "No diseases predicted based on your symptoms."
    
    # Create English summary
    if len(predictions) == 1:
        disease = predictions[0].disease
        prob = predictions[0].prob
        severity = predictions[0].severity
        english_text = f"Based on your symptoms, you may have {disease} (probability: {prob:.1%}, severity: {severity})."
    else:
        top_disease = predictions[0].disease
        prob = predictions[0].prob
        severity = predictions[0].severity
        english_text = f"Based on your symptoms, the most likely condition is {top_disease} (probability: {prob:.1%}, severity: {severity})."
    
    # Translate to user's language
    try:
        if language != 'en':
            translated_text = translate_from_english(english_text, language)
            return translated_text
        else:
            return english_text
    except Exception:
        # Fallback to English if translation fails
        return english_text


# Create FastAPI app
app = FastAPI(
    title="Symptoms Checker API",
    description="AI-powered symptom analysis and disease prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Load artifacts on application startup.
    """
    print("Starting Symptoms Checker API...")
    load_artifacts()
    print("✓ API startup completed")


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Symptoms Checker API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "artifacts_loaded": all([
            model_data is not None,
            meta_data is not None,
            symptom_severity is not None,
            disease_precautions is not None
        ])
    }


# @app.post("/predict", response_model=PredictResponse)
# async def predict(request: PredictRequest):
#     """
#     Main prediction endpoint that processes text or audio input and returns disease predictions.
    
#     Args:
#         request: Prediction request with input, type, language, and mode
        
#     Returns:
#         Prediction response with diseases, probabilities, and additional information
#     """
#     try:
#         # Validate input
#         if request.input_type not in ['text', 'audio']:
#             raise HTTPException(status_code=400, detail="input_type must be 'text' or 'audio'")
        
#         if request.language not in ['en', 'hi', 'pa']:
#             raise HTTPException(status_code=400, detail="language must be 'en', 'hi', or 'pa'")
        
#         if request.mode not in ['text', 'voice']:
#             raise HTTPException(status_code=400, detail="mode must be 'text' or 'voice'")
        
#         # Step 1: Process input based on type
#         if request.input_type == 'audio':
#             # Process audio input
#             input_text = process_audio_input(request.input, request.language)
#             input_text_user_lang = input_text  # Keep original transcript
#         else:
#             # Use text input directly
#             input_text = request.input
#             input_text_user_lang = request.input
        
#         if not input_text.strip():
#             raise HTTPException(status_code=400, detail="No text found in input")
        
#         # Debug logging: Original user text and language
#         print(f"🔍 Original user text: '{input_text_user_lang}'")
#         print(f"🌐 Selected language: '{request.language}'")
        
#         # Step 2: Force translation to English if needed
#         if request.language != 'en':
#             try:
#                 print(f"🔄 Translating from {request.language} to English...")
#                 input_text = translate_to_english(input_text_user_lang, request.language)
#                 print(f"✅ Translated English text: '{input_text}'")
#             except Exception as e:
#                 print(f"❌ Translation error: {e}")
#                 print(f"⚠️ Continuing with original text: '{input_text}'")
#                 # Continue with original text if translation fails
#         else:
#             print(f"✅ Text already in English: '{input_text}'")
        
#         symptoms = extract_symptoms(input_text)
#         print(f"🎯 Extracted symptoms: {symptoms}")

#         if not symptoms:
#             return PredictResponse(
#                 input_text=input_text,
#                 input_text_user_lang=input_text_user_lang,
#                 symptoms=[],
#                 predictions=[],
#                 language=request.language,
#                 display_text="No symptoms detected.",
#                 tts_audio_base64=None
#             )
        
        
#                 # Step 4: Predict disease candidates
#         pred_results = predict_disease(symptoms, disease_map)

#         if not pred_results:
#             return PredictResponse(
#                 input_text=input_text,
#                 input_text_user_lang=input_text_user_lang,
#                 symptoms=symptoms,
#                 predictions=[],
#                 predictions_translated=None,
#                 language=request.language,
#                 display_text="No matching disease found. Please add more symptoms.",
#                 tts_audio_base64=None
#             )

#         # Prefer diseases that explain all extracted symptoms
#         full_matches = [p for p in pred_results if p["match_count"] == len(symptoms)]

#         if len(full_matches) == 1:
#             selected = full_matches[:1]

#         elif len(full_matches) > 1:
#             full_matches.sort(key=lambda x: x["score"], reverse=True)

#             if len(full_matches) >= 2 and (full_matches[0]["score"] - full_matches[1]["score"]) < 0.08:
#                 return PredictResponse(
#                     input_text=input_text,
#                     input_text_user_lang=input_text_user_lang,
#                     symptoms=symptoms,
#                     predictions=[],
#                     predictions_translated=None,
#                     language=request.language,
#                     display_text="Multiple diseases match your symptoms. Please add more symptoms for accurate prediction.",
#                     tts_audio_base64=None
#                 )

#             selected = full_matches[:3]

#         else:
#             if pred_results[0]["score"] < 0.35:
#                 return PredictResponse(
#                     input_text=input_text,
#                     input_text_user_lang=input_text_user_lang,
#                     symptoms=symptoms,
#                     predictions=[],
#                     predictions_translated=None,
#                     language=request.language,
#                     display_text="Symptoms are too general. Please add more symptoms for accurate prediction.",
#                     tts_audio_base64=None
#                 )

#             if len(pred_results) > 1 and abs(pred_results[0]["score"] - pred_results[1]["score"]) < 0.08:
#                 return PredictResponse(
#                     input_text=input_text,
#                     input_text_user_lang=input_text_user_lang,
#                     symptoms=symptoms,
#                     predictions=[],
#                     predictions_translated=None,
#                     language=request.language,
#                     display_text="Multiple diseases match your symptoms. Please add more symptoms for accurate prediction.",
#                     tts_audio_base64=None
#                 )

#             selected = pred_results[:3]

#         emergency_flag = is_emergency(symptoms)

#         disease_predictions = []
#         for item in selected:
#             disease = item["disease"]
#             prob = item["score"]
#             matched = item["matched"]
#             precautions = disease_precautions.get(disease, [])
#             severity = calculate_severity(matched, severity_dict)

#             disease_predictions.append(DiseasePrediction(
#                 disease=disease,
#                 prob=float(prob),
#                 severity=severity,
#                 precautions=precautions,
#                 symptom_descriptions={}
#             ))

#         top_disease = disease_predictions[0].disease
#         top_severity = disease_predictions[0].severity

#         summary = f"You may have {top_disease}. Severity: {top_severity}."

#         if top_severity == "Low":
#             summary += " Home care and rest may help."
#         elif top_severity == "Medium":
#             summary += " Monitor symptoms and consult a doctor if they worsen."
#         else:
#             summary += " Consult a doctor."

#         if emergency_flag:
#             summary += " 🚨 Seek immediate medical attention!"

#         # # Step 7: Generate display text
#         # display_text = generate_display_text(disease_predictions, request.language)
#         display_text = summary
        
#         # Step 7: Generate TTS audio if voice mode
#         tts_audio_base64 = None
#         if request.mode == 'voice':
#             try:
#                 # Try cloud TTS first, fallback to local
#                 try:
#                     cloud_lang_code = f"{request.language}-IN" if request.language in ['hi', 'pa'] else f"{request.language}-US"
#                     _, audio_bytes = text_to_speech_cloud(display_text, cloud_lang_code)
#                 except Exception:
#                     # Fallback to local TTS
#                     _, audio_bytes, temp_path = text_to_speech_local(display_text, request.language)
#                     # Clean up temp file
#                     import os
#                     if os.path.exists(temp_path):
#                         os.unlink(temp_path)
                
#                 # Encode audio as base64
#                 tts_audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
#             except Exception as e:
#                 print(f"TTS warning: {e}")
#                 # Continue without audio if TTS fails
#         predictions_translated = None
#         debug_info = None
        
#         # Return response
#         return PredictResponse(
#             input_text=input_text,
#             input_text_user_lang=input_text_user_lang,
#             symptoms=symptoms,
#             predictions=disease_predictions,
#             predictions_translated=predictions_translated,
#             language=request.language,
#             display_text=display_text,
#             tts_audio_base64=tts_audio_base64,
#             debug=debug_info if debug_info else None
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Main prediction endpoint that processes text or audio input and returns disease predictions.
    """
    try:
        # Validate input
        if request.input_type not in ['text', 'audio']:
            raise HTTPException(status_code=400, detail="input_type must be 'text' or 'audio'")

        if request.language not in ['en', 'hi', 'pa']:
            raise HTTPException(status_code=400, detail="language must be 'en', 'hi', or 'pa'")

        if request.mode not in ['text', 'voice']:
            raise HTTPException(status_code=400, detail="mode must be 'text' or 'voice'")

        # Step 1: Read input
        if request.input_type == 'audio':
            input_text = process_audio_input(request.input, request.language)
            input_text_user_lang = input_text
        else:
            input_text = request.input
            input_text_user_lang = request.input

        if not input_text.strip():
            raise HTTPException(status_code=400, detail="No text found in input")

        print(f"🔍 Original user text: '{input_text_user_lang}'")
        print(f"🌐 Selected language: '{request.language}'")

        # Step 2: Translate if needed
        if request.language != 'en':
            try:
                print(f"🔄 Translating from {request.language} to English...")
                input_text = translate_to_english(input_text_user_lang, request.language)
                print(f"✅ Translated English text: '{input_text}'")
            except Exception as e:
                print(f"❌ Translation error: {e}")
                print(f"⚠️ Continuing with original text: '{input_text}'")
        else:
            print(f"✅ Text already in English: '{input_text}'")

        # Step 3: Extract symptoms
        symptoms = extract_symptoms(input_text)
        print(f"🎯 Extracted symptoms: {symptoms}")

        if not symptoms:
            return PredictResponse(
                input_text=input_text,
                input_text_user_lang=input_text_user_lang,
                symptoms=[],
                predictions=[],
                predictions_translated=None,
                language=request.language,
                display_text="No symptoms detected.",
                tts_audio_base64=None
            )

        # Step 4: Predict disease candidates
        pred_results = predict_disease(symptoms, disease_map)

        if not pred_results:
            return PredictResponse(
                input_text=input_text,
                input_text_user_lang=input_text_user_lang,
                symptoms=symptoms,
                predictions=[],
                predictions_translated=None,
                language=request.language,
                display_text="No matching disease found. Please add more symptoms.",
                tts_audio_base64=None
            )

        #DELETE THIS 
        # If only 1 or 2 symptoms and many diseases are close, ask for more details
        if len(symptoms) <= 2:
            top_same_match_count = [
                p for p in pred_results
                if p["match_count"] == pred_results[0]["match_count"]
            ]
            if len(top_same_match_count) > 1:
                return PredictResponse(
                    input_text=input_text,
                    input_text_user_lang=input_text_user_lang,
                    symptoms=symptoms,
                    predictions=[],
                    predictions_translated=None,
                    language=request.language,
                    display_text="Multiple diseases match your symptoms. Please add more symptoms for accurate prediction.",
                    tts_audio_base64=None
                )
        #NEW FIELD
        # ✅ NEW FIX: if user gives 3+ symptoms → NEVER block
        
        if len(symptoms) >= 3:
            selected = pred_results[:3]
        else:
            selected = pred_results[:1]

                

        

        #IMP
        # # For 3+ symptoms, return top 3 instead of blocking too early
        # selected = pred_results[:3]

        # Step 5: Emergency check
        emergency_flag = is_emergency(symptoms)


        #DELETE THIS
        # Step 6: Build response predictions
        disease_predictions = []
        for item in selected:
            disease = item["disease"]
            prob = item["score"]
            matched = item["matched"]
            precautions = disease_precautions.get(disease, [])

            severity = calculate_severity(matched, severity_dict)

            disease_predictions.append(DiseasePrediction(
                disease=disease,
                prob=float(prob),
                severity=severity,
                precautions=precautions,
                symptom_descriptions={}
            ))
                
        # Step 7: Summary
        top_prediction = disease_predictions[0]
        summary = f"You may have {top_prediction.disease}. Severity: {top_prediction.severity}."

        if top_prediction.severity == "Low":
            summary += " Home care and rest may help."
        elif top_prediction.severity == "Medium":
            summary += " Monitor symptoms and consult a doctor if they worsen."
        else:
            summary += " Consult a doctor."

        if emergency_flag:
            summary += " 🚨 Seek immediate medical attention!"

        display_text = summary
        print(f"🧠 FINAL OUTPUT: {display_text}")

        # Step 8: TTS if voice mode
        tts_audio_base64 = None
        if request.mode == 'voice':
            try:
                try:
                    cloud_lang_code = f"{request.language}-IN" if request.language in ['hi', 'pa'] else f"{request.language}-US"
                    _, audio_bytes = text_to_speech_cloud(display_text, cloud_lang_code)
                except Exception:
                    _, audio_bytes, temp_path = text_to_speech_local(display_text, request.language)
                    import os
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

                tts_audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            except Exception as e:
                print(f"TTS warning: {e}")

        return PredictResponse(
            input_text=input_text,
            input_text_user_lang=input_text_user_lang,
            symptoms=symptoms,
            predictions=disease_predictions,
            predictions_translated=None,
            language=request.language,
            display_text=display_text,
            tts_audio_base64=tts_audio_base64,
            debug=None
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def find_available_port(start_port=8000, max_port=8010):
    """
    Find an available port starting from start_port up to max_port.
    
    Args:
        start_port: Starting port number
        max_port: Maximum port number to try
        
    Returns:
        Available port number
        
    Raises:
        RuntimeError: If no available port found
    """
    import socket
    
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"No available port found between {start_port} and {max_port}")


if __name__ == "__main__":
    import uvicorn
    
    # Find available port
    try:
        port = find_available_port()
        print(f"⚡ API running on http://localhost:{port}")
        print(f"📚 API Documentation: http://localhost:{port}/docs")
        print(f"❤️ Health Check: http://localhost:{port}/health")
        print("=" * 50)
        
        uvicorn.run(app, host="0.0.0.0", port=port)
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("Please free up some ports or modify the port range in the code.")
