"""
Disease Matcher module for symptoms checker project.
Provides rule-based disease scoring and ensemble methods with ML models.
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


class DiseaseMatcher:
    """
    Rule-based disease matcher that scores diseases based on symptom overlap.
    Provides ensemble methods to combine with ML model predictions.
    """
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        """
        Initialize disease matcher with loaded artifacts.
        
        Args:
            artifacts_dir: Directory containing artifact files
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.disease_symptom_map = {}
        self.symptom_severity = {}
        self.meta_data = {}
        self.diseases = []
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """
        Load required artifacts with defensive error handling.
        """
        try:
            # Load disease-symptom mapping
            disease_map_path = self.artifacts_dir / "disease_symptom_map.json"
            if not disease_map_path.exists():
                raise FileNotFoundError(f"Disease-symptom map not found: {disease_map_path}")
            
            with open(disease_map_path, 'r', encoding='utf-8') as f:
                self.disease_symptom_map = json.load(f)
            print(f"✓ Loaded disease-symptom map: {len(self.disease_symptom_map)} diseases")
            
            # Load symptom severity weights (optional)
            severity_path = self.artifacts_dir / "symptom_severity.json"
            if severity_path.exists():
                with open(severity_path, 'r', encoding='utf-8') as f:
                    self.symptom_severity = json.load(f)
                print(f"✓ Loaded symptom severity: {len(self.symptom_severity)} symptoms")
            else:
                print("⚠️ Symptom severity not found, using default weight 1.0")
                self.symptom_severity = {}
            
            # Load metadata for diseases list and thresholds
            meta_path = self.artifacts_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.meta_data = json.load(f)
                self.diseases = self.meta_data.get('diseases', list(self.disease_symptom_map.keys()))
                print(f"✓ Loaded metadata: {len(self.diseases)} diseases")
            else:
                print("⚠️ Meta data not found, using disease-symptom map keys")
                self.diseases = list(self.disease_symptom_map.keys())
            
        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            raise
    
    def score_diseases(self, symptoms: List[str], top_k: int = 10, temperature: float = 1.0) -> List[Dict[str, Any]]:
        """
        Score diseases based on symptom overlap using rule-based heuristics.
        
        Args:
            symptoms: List of extracted symptoms
            top_k: Number of top diseases to return
            temperature: Temperature for softmax probability conversion
            
        Returns:
            List of dictionaries with disease, score, prob, and matched symptoms
        """
        if not symptoms:
            return []
        
        # Convert symptoms to lowercase for matching
        symptoms_lower = [s.lower().strip() for s in symptoms]
        symptoms_set = set(symptoms_lower)
        
        disease_scores = []
        
        for disease in self.diseases:
            # Get canonical symptoms for this disease
            disease_symptoms = self.disease_symptom_map.get(disease, [])
            if not disease_symptoms:
                continue
            
            # Convert to lowercase for matching
            disease_symptoms_lower = [s.lower().strip() for s in disease_symptoms]
            disease_symptoms_set = set(disease_symptoms_lower)
            
            # Find matched symptoms
            matched_symptoms = symptoms_set.intersection(disease_symptoms_set)
            matched_list = list(matched_symptoms)
            
            if not matched_symptoms:
                continue
            
            # Calculate weights
            matched_weight = sum(self.symptom_severity.get(s, 1.0) for s in matched_list)
            total_weight = sum(self.symptom_severity.get(s, 1.0) for s in disease_symptoms_lower) + 1e-6
            
            # Base score: weighted overlap ratio
            base_score = matched_weight / total_weight
            
            # Length factor: how many symptoms matched vs total disease symptoms
            length_factor = len(matched_symptoms) / max(1, len(disease_symptoms_set))
            
            # Combined score: weighted combination of base score and length factor
            score = base_score * 0.7 + length_factor * 0.3
            
            # Penalize extra unmatched input symptoms lightly
            extra_count = len(symptoms_set) - len(matched_symptoms)
            if extra_count > 0:
                score *= 1 / (1 + 0.05 * extra_count)
            
            disease_scores.append({
                "disease": disease,
                "score": score,
                "matched": matched_list,
                "matched_count": len(matched_symptoms),
                "total_disease_symptoms": len(disease_symptoms_set)
            })
        
        if not disease_scores:
            return []
        
        # Convert scores to probabilities using softmax
        scores = [item["score"] for item in disease_scores]
        exp_scores = [math.exp(s / temperature) for s in scores]
        sum_exp = sum(exp_scores)
        
        # Add probabilities to results
        for i, item in enumerate(disease_scores):
            item["prob"] = exp_scores[i] / sum_exp
        
        # Sort by probability (descending) and return top_k
        disease_scores.sort(key=lambda x: x["prob"], reverse=True)
        
        return disease_scores[:top_k]
    
    def ensemble_with_model(self, model_probabilities: List[float], scorer_probs: List[float], 
                           model_weight: float = 0.4) -> List[float]:
        """
        Combine ML model probabilities with rule-based scorer probabilities.
        
        Args:
            model_probabilities: Probabilities from ML model
            scorer_probs: Probabilities from rule-based scorer
            model_weight: Weight for ML model (0.0 to 1.0)
            
        Returns:
            Final ensemble probabilities
        """
        if not model_probabilities or not scorer_probs:
            return model_probabilities or scorer_probs
        
        # Ensure same length
        min_len = min(len(model_probabilities), len(scorer_probs))
        model_probs = model_probabilities[:min_len]
        scorer_probs = scorer_probs[:min_len]
        
        # Normalize both arrays to sum to 1
        model_sum = sum(model_probs)
        scorer_sum = sum(scorer_probs)
        
        if model_sum > 0:
            model_probs = [p / model_sum for p in model_probs]
        if scorer_sum > 0:
            scorer_probs = [p / scorer_sum for p in scorer_probs]
        
        # Weighted combination
        final_probs = []
        for i in range(min_len):
            final_prob = model_weight * model_probs[i] + (1 - model_weight) * scorer_probs[i]
            final_probs.append(final_prob)
        
        return final_probs
    
    def get_severity_label_from_meta(self, score_or_weight: float) -> str:
        """
        Get severity label based on score/weight using meta.json thresholds.
        
        Args:
            score_or_weight: Numeric score or weight value
            
        Returns:
            Severity label: "Low", "Medium", "High", or "Critical"
        """
        thresholds = self.meta_data.get('severity_thresholds', {
            'low': 0.0,
            'medium': 0.3,
            'high': 0.6,
            'critical': 0.8
        })
        
        # Convert to lowercase keys for consistency
        thresholds_lower = {k.lower(): v for k, v in thresholds.items()}
        
        if score_or_weight >= thresholds_lower.get('critical', 0.8):
            return "Critical"
        elif score_or_weight >= thresholds_lower.get('high', 0.6):
            return "High"
        elif score_or_weight >= thresholds_lower.get('medium', 0.3):
            return "Medium"
        else:
            return "Low"
    
    def get_disease_info(self, disease: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific disease.
        
        Args:
            disease: Disease name
            
        Returns:
            Dictionary with disease information
        """
        symptoms = self.disease_symptom_map.get(disease, [])
        severity_weights = [self.symptom_severity.get(s, 1.0) for s in symptoms]
        
        return {
            "disease": disease,
            "symptoms": symptoms,
            "symptom_count": len(symptoms),
            "total_severity_weight": sum(severity_weights),
            "avg_severity_weight": sum(severity_weights) / len(symptoms) if symptoms else 0
        }


def main():
    """
    CLI test function for disease matcher.
    """
    print("🔍 Disease Matcher CLI Test")
    print("=" * 50)
    
    try:
        # Initialize matcher
        matcher = DiseaseMatcher()
        
        # Sample symptoms for testing
        sample_text = "I have fever, headache, cough, and sore throat"
        print(f"📝 Sample text: '{sample_text}'")
        
        # Try to extract symptoms using symptom_extractor if available
        try:
            from symptom_extractor import extract_symptoms
            symptoms = extract_symptoms(sample_text)
            print(f"🎯 Extracted symptoms: {symptoms}")
        except ImportError:
            print("⚠️ symptom_extractor not available, using manual symptoms")
            symptoms = ["fever", "headache", "cough", "sore throat"]
        
        if not symptoms:
            print("❌ No symptoms found")
            return
        
        # Score diseases
        print(f"\n📊 Scoring diseases for {len(symptoms)} symptoms...")
        results = matcher.score_diseases(symptoms, top_k=5, temperature=1.0)
        
        if not results:
            print("❌ No diseases matched")
            return
        
        print(f"\n🏆 Top {len(results)} Disease Matches:")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            severity_label = matcher.get_severity_label_from_meta(result["score"])
            print(f"{i}. {result['disease']}")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Probability: {result['prob']:.4f} ({result['prob']*100:.1f}%)")
            print(f"   Severity: {severity_label}")
            print(f"   Matched symptoms: {result['matched']}")
            print(f"   Matched: {result['matched_count']}/{result['total_disease_symptoms']}")
            print()
        
        # Test ensemble method
        print("🔄 Testing ensemble method...")
        model_probs = [0.1, 0.3, 0.2, 0.4]  # Mock ML model probabilities
        scorer_probs = [result["prob"] for result in results[:4]]  # Top 4 scorer probabilities
        
        if len(scorer_probs) >= 4:
            ensemble_probs = matcher.ensemble_with_model(model_probs, scorer_probs, model_weight=0.4)
            print(f"Model probs: {[f'{p:.3f}' for p in model_probs]}")
            print(f"Scorer probs: {[f'{p:.3f}' for p in scorer_probs]}")
            print(f"Ensemble probs: {[f'{p:.3f}' for p in ensemble_probs]}")
        
        print("\n✅ Disease Matcher test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in disease matcher test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
