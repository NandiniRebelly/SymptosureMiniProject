<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/FastAPI-API%20Backend-brightgreen.svg" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/ML-Model-orange.svg" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
  <img src="https://img.shields.io/github/last-commit/radha1805/symptom-checker-ml" alt="Last Commit"/>
  <img src="https://img.shields.io/github/repo-size/radha1805/symptom-checker-ml" alt="Repo Size"/>
</p>


# Symptom Checker ML Model

An AI-based symptom checker designed with a complete ML training pipeline, multilingual support, and a deployment-ready FastAPI backend.

## 📌 Overview

This project implements a machine learning model that predicts possible diseases based on user-provided symptoms.  
It includes preprocessing, feature engineering, model training, inferencing APIs, multilingual handling, and Docker-based deployment.  
The architecture is clean, modular, and suitable for real-world telemedicine workflows.

---

## 🚀 Features

- **Machine Learning Model** for symptom-based disease prediction  
- **Training Pipeline** with preprocessing, feature engineering, and evaluation  
- **Prediction API (FastAPI)** exposing inference endpoints  
- **Multilingual Support** (English, Hindi, Punjabi)  
- **Speech-to-Text & Text-to-Speech Support** (local + cloud options)  
- **Docker Deployment** for easy containerization  
- **Clean Modular Architecture** for scalability  

---

## 📁 Project Structure

```
symptoms_checker/
│
├── api.py                     # FastAPI prediction server
├── train_model.py             # Model training script
├── feature_engineer.py        # Feature extraction & preprocessing
├── symptom_extractor.py       # Extracts symptom keywords
├── translator.py              # Multilingual translation logic
├── stt_tts.py                 # Speech-to-text and text-to-speech
│
├── artifacts/                 # Saved ML models and encoders
├── tests/                     # Test suite
│
├── requirements.txt           # Dependencies
├── Dockerfile                 # Deployment configuration
└── README.md                  # Documentation
```

---

## 🔧 Installation

### 1. Clone the repository
```
git clone https://github.com/yourusername/symptom-checker-ml.git
cd symptom-checker-ml
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

---

## ▶️ Running the Model API

Start the FastAPI inference server:
```
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open in browser:  
**http://localhost:8000/docs**

---

## 📡 Example API Usage

### POST `/predict`

**Input**
```json
{
  "text": "I have fever and body pain"
}
```

**Output**
```json
{
  "predicted_disease": "Influenza"
}
```

---

## 🧠 Model

The model uses classical ML techniques trained on symptom–disease mappings.  
Pipeline includes:

- Text preprocessing  
- Symptom feature extraction  
- One-hot encodings  
- Model training  
- Artifact saving  

Artifacts are stored in the `/artifacts` directory.

---

## 🌍 Multilingual Support

Supported languages:
- English  
- Hindi  
- Punjabi  

Inputs are normalized internally before prediction.

---

## 🐳 Docker Deployment

Build image:
```
docker build -t symptom-checker .
```

Run container:
```
docker run -p 8000:8000 symptom-checker
```

---

## 🧪 Testing

Run the test suite:
```
pytest
```

---

## 👩‍💻 Author

**Radha Sarda**  
Machine Learning | AI | Deployment  
Open to collaborations and improvements.

---

## 📜 License

Released under the MIT License.


