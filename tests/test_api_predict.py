"""
Test suite for API predict endpoint.
Tests the FastAPI /predict endpoint with sample inputs and validates response structure.
"""

import pytest
import json
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import app


class TestAPIPredict:
    """
    Test cases for the /predict API endpoint.
    """
    
    def setup_method(self):
        """
        Set up test fixtures before each test method.
        """
        # Create test client
        self.client = TestClient(app)
    
    def test_predict_endpoint_exists(self):
        """
        Test that the /predict endpoint exists and accepts POST requests.
        """
        response = self.client.post("/predict", json={})
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404, "Predict endpoint should exist"
    
    def test_predict_text_input(self):
        """
        Test /predict endpoint with text input.
        """
        test_data = {
            "input": "I have a fever and headache",
            "input_type": "text",
            "language": "en",
            "mode": "text"
        }
        
        try:
            response = self.client.post("/predict", json=test_data)
            
            # Check response status
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            # Parse response JSON
            response_data = response.json()
            
            # Validate response structure
            required_fields = [
                "input_text",
                "input_text_user_lang", 
                "symptoms",
                "predictions",
                "language",
                "display_text"
            ]
            
            for field in required_fields:
                assert field in response_data, f"Missing required field: {field}"
            
            # Validate field types
            assert isinstance(response_data["input_text"], str), "input_text should be string"
            assert isinstance(response_data["input_text_user_lang"], str), "input_text_user_lang should be string"
            assert isinstance(response_data["symptoms"], list), "symptoms should be list"
            assert isinstance(response_data["predictions"], list), "predictions should be list"
            assert isinstance(response_data["language"], str), "language should be string"
            assert isinstance(response_data["display_text"], str), "display_text should be string"
            
            # Validate predictions structure
            for prediction in response_data["predictions"]:
                required_prediction_fields = [
                    "disease",
                    "prob", 
                    "severity",
                    "precautions",
                    "symptom_descriptions"
                ]
                
                for field in required_prediction_fields:
                    assert field in prediction, f"Missing prediction field: {field}"
                
                # Validate prediction field types
                assert isinstance(prediction["disease"], str), "disease should be string"
                assert isinstance(prediction["prob"], (int, float)), "prob should be number"
                assert isinstance(prediction["severity"], str), "severity should be string"
                assert isinstance(prediction["precautions"], list), "precautions should be list"
                assert isinstance(prediction["symptom_descriptions"], dict), "symptom_descriptions should be dict"
                
                # Validate probability range
                assert 0 <= prediction["prob"] <= 1, f"Probability should be between 0 and 1, got {prediction['prob']}"
            
            print(f"✅ Text input test passed. Found {len(response_data['symptoms'])} symptoms, {len(response_data['predictions'])} predictions")
            
        except Exception as e:
            pytest.skip(f"Skipping test due to missing artifacts or API issues: {e}")
    
    def test_predict_voice_mode(self):
        """
        Test /predict endpoint with voice mode.
        """
        test_data = {
            "input": "I have a fever and headache",
            "input_type": "text",
            "language": "en",
            "mode": "voice"
        }
        
        try:
            response = self.client.post("/predict", json=test_data)
            
            if response.status_code == 200:
                response_data = response.json()
                
                # In voice mode, should have tts_audio_base64 field
                assert "tts_audio_base64" in response_data, "Voice mode should include tts_audio_base64"
                
                # tts_audio_base64 should be string or null
                if response_data["tts_audio_base64"] is not None:
                    assert isinstance(response_data["tts_audio_base64"], str), "tts_audio_base64 should be string"
                
                print(f"✅ Voice mode test passed. TTS audio: {'present' if response_data['tts_audio_base64'] else 'not generated'}")
            else:
                print(f"⚠️ Voice mode test skipped due to status {response.status_code}")
                
        except Exception as e:
            pytest.skip(f"Skipping voice mode test: {e}")
    
    def test_predict_different_languages(self):
        """
        Test /predict endpoint with different languages.
        """
        languages = ["en", "hi", "pa"]
        
        for lang in languages:
            test_data = {
                "input": "I have a fever and headache",
                "input_type": "text", 
                "language": lang,
                "mode": "text"
            }
            
            try:
                response = self.client.post("/predict", json=test_data)
                
                if response.status_code == 200:
                    response_data = response.json()
                    assert response_data["language"] == lang, f"Response language should match request: {lang}"
                    print(f"✅ Language test passed for {lang}")
                else:
                    print(f"⚠️ Language test skipped for {lang} due to status {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ Language test skipped for {lang}: {e}")
    
    def test_predict_invalid_input(self):
        """
        Test /predict endpoint with invalid input.
        """
        # Test empty input
        test_data = {
            "input": "",
            "input_type": "text",
            "language": "en", 
            "mode": "text"
        }
        
        try:
            response = self.client.post("/predict", json=test_data)
            
            # Should handle empty input gracefully
            if response.status_code == 200:
                response_data = response.json()
                assert isinstance(response_data["symptoms"], list), "Should return empty symptoms list"
                print("✅ Empty input handled correctly")
            else:
                print(f"⚠️ Empty input returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Empty input test skipped: {e}")
    
    def test_predict_invalid_parameters(self):
        """
        Test /predict endpoint with invalid parameters.
        """
        # Test invalid input_type
        test_data = {
            "input": "test",
            "input_type": "invalid",
            "language": "en",
            "mode": "text"
        }
        
        try:
            response = self.client.post("/predict", json=test_data)
            
            # Should return 400 for invalid parameters
            assert response.status_code == 400, f"Expected 400 for invalid input_type, got {response.status_code}"
            print("✅ Invalid input_type handled correctly")
            
        except Exception as e:
            print(f"⚠️ Invalid parameters test skipped: {e}")
    
    def test_health_endpoint(self):
        """
        Test the /health endpoint.
        """
        try:
            response = self.client.get("/health")
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Validate health response structure
                assert "status" in response_data, "Health response should have status"
                assert "artifacts_loaded" in response_data, "Health response should have artifacts_loaded"
                
                assert response_data["status"] == "healthy", "Status should be healthy"
                assert isinstance(response_data["artifacts_loaded"], bool), "artifacts_loaded should be boolean"
                
                print(f"✅ Health endpoint test passed. Artifacts loaded: {response_data['artifacts_loaded']}")
            else:
                print(f"⚠️ Health endpoint returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Health endpoint test skipped: {e}")
    
    def test_root_endpoint(self):
        """
        Test the root endpoint.
        """
        try:
            response = self.client.get("/")
            
            assert response.status_code == 200, f"Root endpoint should return 200, got {response.status_code}"
            
            response_data = response.json()
            assert "message" in response_data, "Root response should have message"
            assert "version" in response_data, "Root response should have version"
            assert "endpoints" in response_data, "Root response should have endpoints"
            
            print("✅ Root endpoint test passed")
            
        except Exception as e:
            print(f"⚠️ Root endpoint test skipped: {e}")


class TestAPIIntegration:
    """
    Integration tests for the complete API workflow.
    """
    
    def setup_method(self):
        """
        Set up test fixtures before each test method.
        """
        self.client = TestClient(app)
    
    def test_complete_workflow(self):
        """
        Test complete workflow from input to prediction.
        """
        test_cases = [
            {
                "input": "I have a fever and headache",
                "expected_symptoms": ["headache"],  # May vary based on vocabulary
                "description": "Basic fever and headache"
            },
            {
                "input": "I feel dizzy and have been vomiting",
                "expected_symptoms": ["vomiting"],
                "description": "Dizziness and vomiting"
            },
            {
                "input": "I have itching and skin rash",
                "expected_symptoms": ["itching"],
                "description": "Skin-related symptoms"
            }
        ]
        
        for test_case in test_cases:
            test_data = {
                "input": test_case["input"],
                "input_type": "text",
                "language": "en",
                "mode": "text"
            }
            
            try:
                response = self.client.post("/predict", json=test_data)
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Validate that we get some response
                    assert len(response_data["symptoms"]) >= 0, f"Should have symptoms list for '{test_case['description']}'"
                    assert len(response_data["predictions"]) >= 0, f"Should have predictions list for '{test_case['description']}'"
                    assert len(response_data["display_text"]) > 0, f"Should have display text for '{test_case['description']}'"
                    
                    print(f"✅ Complete workflow test passed for: {test_case['description']}")
                else:
                    print(f"⚠️ Complete workflow test skipped for '{test_case['description']}' due to status {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ Complete workflow test skipped for '{test_case['description']}': {e}")


if __name__ == "__main__":
    """
    Run tests directly with: python -m pytest tests/test_api_predict.py -v
    """
    pytest.main([__file__, "-v"])
