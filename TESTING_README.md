# Testing and Running Guide

This guide explains how to run tests and execute the Symptoms Checker project.

## Prerequisites

### Required Environment Variables (Optional)

For full functionality, set these environment variables:

```bash
# Google Cloud credentials (optional - for cloud services)
set GOOGLE_APPLICATION_CREDENTIALS=path\to\your\google-credentials.json

# Or on Linux/Mac:
export GOOGLE_APPLICATION_CREDENTIALS=path/to/your/google-credentials.json
```

**Note**: The project works without Google Cloud credentials using local fallbacks, but cloud services provide better accuracy.

### Required Files

Before running tests or the application, ensure these files exist:
- `artifacts/symptom_vocab.json` (created by `data_prep.py`)
- `artifacts/disease_symptom_map.json` (created by `data_prep.py`)
- `artifacts/model.joblib` (created by `train_model.py`)
- `artifacts/meta.json` (created by `train_model.py`)

## Running Tests

### Install Test Dependencies

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install pytest if not already installed
pip install pytest
```

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_symptom_extractor.py -v
pytest tests/test_api_predict.py -v
```

### Test Descriptions

#### `test_symptom_extractor.py`
- Tests symptom extraction with known sample sentences
- Validates that expected symptoms are found
- Tests edge cases (empty input, typos, multiple symptoms)
- Tests fuzzy matching capabilities

#### `test_api_predict.py`
- Tests the `/predict` API endpoint using FastAPI TestClient
- Validates response JSON structure
- Tests different input types (text/voice modes)
- Tests different languages (en/hi/pa)
- Tests error handling

### Expected Test Output

```
tests/test_symptom_extractor.py::TestSymptomExtractor::test_extract_symptoms_basic PASSED
tests/test_symptom_extractor.py::TestSymptomExtractor::test_extract_symptoms_empty_input PASSED
tests/test_api_predict.py::TestAPIPredict::test_predict_text_input PASSED
tests/test_api_predict.py::TestAPIPredict::test_health_endpoint PASSED
```

## Running the Project

### Method 1: Using run.bat (Windows)

```bash
# Simply double-click run.bat or run from command line:
run.bat
```

This script will:
1. Activate the virtual environment
2. Run `data_prep.py` to process CSV files
3. Run `train_model.py` to train the ML model
4. Start the API server with `uvicorn api:app --reload --port 8000`

### Method 2: Manual Steps

```bash
# 1. Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# 2. Run data preparation
python data_prep.py

# 3. Train the model
python train_model.py

# 4. Start the API server
uvicorn api:app --reload --port 8000 --host 0.0.0.0
```

### Method 3: Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run specific services
docker-compose up symptom-api
```

## API Testing

### Manual API Testing

Once the server is running, test the API:

```bash
# Test with curl (Linux/Mac)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "input": "I have a fever and headache",
       "input_type": "text",
       "language": "en",
       "mode": "text"
     }'

# Test with PowerShell (Windows)
Invoke-WebRequest -Uri "http://localhost:8000/predict" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"input": "I have a fever and headache", "input_type": "text", "language": "en", "mode": "text"}'
```

### API Endpoints

- **POST** `/predict` - Main prediction endpoint
- **GET** `/health` - Health check endpoint
- **GET** `/` - API information
- **GET** `/docs` - Interactive API documentation

### Expected API Response

```json
{
  "input_text": "I have a fever and headache",
  "input_text_user_lang": "I have a fever and headache",
  "symptoms": ["headache"],
  "predictions": [
    {
      "disease": "allergy",
      "prob": 0.14,
      "severity": "Medium",
      "precautions": ["Avoid allergens", "Take antihistamines"],
      "symptom_descriptions": {
        "headache": "Pain in the head or neck area"
      }
    }
  ],
  "language": "en",
  "display_text": "Based on your symptoms, you may have allergy (probability: 14%, severity: Medium).",
  "tts_audio_base64": null
}
```

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'rapidfuzz'"**
   ```bash
   pip install rapidfuzz
   ```

2. **"FileNotFoundError: artifacts/symptom_vocab.json"**
   ```bash
   # Run data preparation first
   python data_prep.py
   ```

3. **"FileNotFoundError: artifacts/model.joblib"**
   ```bash
   # Run model training first
   python train_model.py
   ```

4. **"Connection refused" when testing API**
   ```bash
   # Make sure the server is running
   uvicorn api:app --reload --port 8000
   ```

5. **Tests skip with "Skipping test due to missing artifacts"**
   ```bash
   # Run the full pipeline first
   python data_prep.py
   python train_model.py
   ```

### Debug Mode

Run tests with debug output:

```bash
# Verbose test output
pytest tests/ -v -s

# Show print statements
pytest tests/ -v -s --capture=no

# Run specific test with debug
pytest tests/test_symptom_extractor.py::TestSymptomExtractor::test_extract_symptoms_basic -v -s
```

### Performance Testing

```bash
# Run tests with timing
pytest tests/ --durations=10

# Run API tests with performance monitoring
pytest tests/test_api_predict.py -v --durations=10
```

## Development Workflow

### Recommended Development Steps

1. **Setup Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Data Pipeline**
   ```bash
   python data_prep.py
   python train_model.py
   ```

3. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

4. **Start Development Server**
   ```bash
   uvicorn api:app --reload --port 8000
   ```

5. **Test Frontend**
   - Open `frontend_examples/index.html` in browser
   - Test with different languages and input modes

### Continuous Integration

For CI/CD pipelines:

```bash
# Install dependencies
pip install -r requirements.txt

# Run data preparation
python data_prep.py

# Run model training
python train_model.py

# Run tests
pytest tests/ -v --tb=short

# Start server for integration tests
uvicorn api:app --port 8000 &
sleep 5

# Run API tests
pytest tests/test_api_predict.py -v

# Stop server
pkill -f uvicorn
```

## Support

For issues and questions:

1. Check that all required files exist in `artifacts/`
2. Verify virtual environment is activated
3. Run tests to identify specific issues
4. Check server logs for API errors
5. Review this documentation
6. Check GitHub issues for known problems
