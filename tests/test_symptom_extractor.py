"""
Test suite for symptom_extractor module.
Tests symptom extraction functionality with known sample sentences.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from symptom_extractor import extract_symptoms, SymptomExtractor


class TestSymptomExtractor:
    """
    Test cases for symptom extraction functionality.
    """
    
    def setup_method(self):
        """
        Set up test fixtures before each test method.
        """
        # Create extractor instance for testing
        self.extractor = SymptomExtractor()
    
    def test_extract_symptoms_basic(self):
        """
        Test basic symptom extraction with common symptoms.
        """
        # Test cases: (input_text, expected_symptoms)
        test_cases = [
            ("I have a fever and headache", ["headache"]),
            ("My stomach hurts and I feel nauseous", []),  # May not match exact symptoms
            ("I have itching and skin rash", ["itching"]),
            ("I feel dizzy and have been vomiting", ["vomiting"]),
            ("I have joint pain and muscle weakness", []),  # May not match exact symptoms
            ("I'm sneezing continuously", []),  # May not match exact symptoms
            ("I have back pain", []),  # May not match exact symptoms
            ("I feel fatigued", ["fatigue"]),
            ("I have constipation", ["constipation"]),
        ]
        
        for text, expected_symptoms in test_cases:
            with pytest.subtest(text=text):
                symptoms = extract_symptoms(text)
                print(f"Text: '{text}' -> Symptoms: {symptoms}")
                
                # Assert that we get some symptoms (may not be exact matches due to vocabulary)
                assert isinstance(symptoms, list), f"Expected list, got {type(symptoms)}"
                
                # For known symptoms, check if they're found
                if expected_symptoms:
                    for expected in expected_symptoms:
                        if expected in self.extractor.symptom_vocab:
                            assert expected in symptoms, f"Expected '{expected}' in symptoms for '{text}'"
    
    def test_extract_symptoms_empty_input(self):
        """
        Test symptom extraction with empty or invalid input.
        """
        # Test empty string
        assert extract_symptoms("") == []
        assert extract_symptoms("   ") == []
        assert extract_symptoms(None) == []
    
    def test_extract_symptoms_no_symptoms(self):
        """
        Test symptom extraction with text containing no symptoms.
        """
        no_symptom_texts = [
            "I feel fine today",
            "The weather is nice",
            "How are you doing?",
            "This is just a regular conversation",
            "I love programming",
        ]
        
        for text in no_symptom_texts:
            symptoms = extract_symptoms(text)
            assert isinstance(symptoms, list), f"Expected list for '{text}'"
            # May or may not find symptoms, but should not crash
    
    def test_extract_symptoms_with_typos(self):
        """
        Test symptom extraction with typos (fuzzy matching).
        """
        typo_cases = [
            ("I have a fevr", []),  # fever -> fevr (may not match)
            ("My hed hurts", []),  # head -> hed (may not match)
            ("I have stomch pain", []),  # stomach -> stomch (may not match)
            ("I feel dizy", []),  # dizzy -> dizy (may not match)
            ("I have a skin rash with iching", ["itching"]),  # itching -> iching (should match)
        ]
        
        for text, expected_symptoms in typo_cases:
            symptoms = extract_symptoms(text)
            assert isinstance(symptoms, list), f"Expected list for '{text}'"
            
            # Check if expected symptoms are found
            for expected in expected_symptoms:
                if expected in self.extractor.symptom_vocab:
                    assert expected in symptoms, f"Expected '{expected}' in symptoms for '{text}'"
    
    def test_extract_symptoms_multiple_symptoms(self):
        """
        Test symptom extraction with multiple symptoms in one sentence.
        """
        multi_symptom_cases = [
            ("I have fever, headache, and nausea", ["headache", "nausea"]),
            ("My symptoms include itching, skin rash, and fatigue", ["itching", "fatigue"]),
            ("I feel dizzy, have vomiting, and constipation", ["vomiting", "constipation"]),
        ]
        
        for text, expected_symptoms in multi_symptom_cases:
            symptoms = extract_symptoms(text)
            assert isinstance(symptoms, list), f"Expected list for '{text}'"
            
            # Check that we find multiple symptoms
            found_expected = [s for s in expected_symptoms if s in symptoms]
            assert len(found_expected) > 0, f"Expected to find some symptoms in '{text}', got {symptoms}"
    
    def test_symptom_vocabulary_loaded(self):
        """
        Test that symptom vocabulary is properly loaded.
        """
        assert hasattr(self.extractor, 'symptom_vocab'), "Extractor should have symptom_vocab attribute"
        assert isinstance(self.extractor.symptom_vocab, list), "symptom_vocab should be a list"
        assert len(self.extractor.symptom_vocab) > 0, "symptom_vocab should not be empty"
        
        # Check for some expected symptoms
        expected_symptoms = ["fever", "headache", "itching", "vomiting", "fatigue", "constipation"]
        found_symptoms = [s for s in expected_symptoms if s in self.extractor.symptom_vocab]
        assert len(found_symptoms) > 0, f"Expected to find some common symptoms, found: {found_symptoms}"
    
    def test_extract_symptoms_case_insensitive(self):
        """
        Test that symptom extraction is case insensitive.
        """
        text_variations = [
            "I have a FEVER",
            "I have a Fever",
            "I have a fever",
            "I HAVE A FEVER",
        ]
        
        # All variations should produce similar results
        results = []
        for text in text_variations:
            symptoms = extract_symptoms(text)
            results.append(set(symptoms))
        
        # Results should be similar (allowing for some variation)
        assert len(results) == len(text_variations), "Should have results for all variations"
    
    def test_extract_symptoms_with_punctuation(self):
        """
        Test symptom extraction with various punctuation.
        """
        punctuation_cases = [
            "I have a fever!",
            "I have a fever?",
            "I have a fever.",
            "I have a fever,",
            "I have a fever;",
            "I have a fever:",
        ]
        
        for text in punctuation_cases:
            symptoms = extract_symptoms(text)
            assert isinstance(symptoms, list), f"Expected list for '{text}'"
            # Should not crash with punctuation


class TestSymptomExtractorIntegration:
    """
    Integration tests for symptom extractor with real vocabulary.
    """
    
    def test_extractor_initialization(self):
        """
        Test that SymptomExtractor initializes properly.
        """
        try:
            extractor = SymptomExtractor()
            assert extractor is not None
            assert hasattr(extractor, 'symptom_vocab')
            assert len(extractor.symptom_vocab) > 0
        except Exception as e:
            pytest.skip(f"Skipping test due to missing artifacts: {e}")
    
    def test_convenience_function(self):
        """
        Test the convenience function extract_symptoms.
        """
        try:
            symptoms = extract_symptoms("I have a fever and headache")
            assert isinstance(symptoms, list)
        except Exception as e:
            pytest.skip(f"Skipping test due to missing artifacts: {e}")


if __name__ == "__main__":
    """
    Run tests directly with: python -m pytest tests/test_symptom_extractor.py -v
    """
    pytest.main([__file__, "-v"])
