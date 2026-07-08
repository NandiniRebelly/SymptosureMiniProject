# """
# Feature engineering module for symptoms checker project.
# Creates feature vectors and training matrices for ML model training.
# """

# import json
# import numpy as np
# from pathlib import Path
# from typing import List, Dict, Tuple, Any


# class FeatureEngineer:
#     """
#     Handles feature vector creation and training matrix preparation.
#     Converts symptom data into numerical representations for ML models.
#     """
    
#     def __init__(self, artifacts_dir: str = "artifacts"):
#         """
#         Initialize the feature engineer with artifact paths.
        
#         Args:
#             artifacts_dir: Directory containing JSON artifacts
#         """
#         self.artifacts_dir = Path(artifacts_dir)
#         self.vocab_path = self.artifacts_dir / "symptom_vocab.json"
#         self.disease_symptom_path = self.artifacts_dir / "disease_symptom_map.json"
        
#         # Load vocabulary and disease mappings
#         self.symptom_vocab = self._load_vocabulary()
#         self.disease_symptom_map = self._load_disease_symptom_map()
        
#         # Create lookup dictionaries for efficient indexing
#         self.symptom_to_idx = {symptom: idx for idx, symptom in enumerate(self.symptom_vocab)}
#         self.disease_to_idx = {disease: idx for idx, disease in enumerate(self.disease_symptom_map.keys())}
        
#         print(f"✓ Loaded {len(self.symptom_vocab)} symptoms and {len(self.disease_symptom_map)} diseases")
    
#     def _load_vocabulary(self) -> List[str]:
#         """
#         Load symptom vocabulary from JSON file.
        
#         Returns:
#             List of canonical symptom strings
            
#         Raises:
#             FileNotFoundError: If vocabulary file doesn't exist
#         """
#         if not self.vocab_path.exists():
#             raise FileNotFoundError(f"Symptom vocabulary not found: {self.vocab_path}")
        
#         with open(self.vocab_path, 'r', encoding='utf-8') as f:
#             vocab = json.load(f)
        
#         if not isinstance(vocab, list) or len(vocab) == 0:
#             raise ValueError("Vocabulary must be a non-empty list of strings")
        
#         return vocab
    
#     def _load_disease_symptom_map(self) -> Dict[str, List[str]]:
#         """
#         Load disease-symptom mapping from JSON file.
        
#         Returns:
#             Dictionary mapping diseases to lists of symptoms
            
#         Raises:
#             FileNotFoundError: If mapping file doesn't exist
#         """
#         if not self.disease_symptom_path.exists():
#             raise FileNotFoundError(f"Disease-symptom mapping not found: {self.disease_symptom_path}")
        
#         with open(self.disease_symptom_path, 'r', encoding='utf-8') as f:
#             mapping = json.load(f)
        
#         if not isinstance(mapping, dict) or len(mapping) == 0:
#             raise ValueError("Disease-symptom mapping must be a non-empty dictionary")
        
#         return mapping
    
#     def build_feature_vector(self, symptoms: List[str]) -> np.ndarray:
#         """
#         Build binary feature vector from list of symptoms.
        
#         Args:
#             symptoms: List of symptom strings
            
#         Returns:
#             Binary numpy array of length |vocab| (1 where symptom exists, 0 otherwise)
#         """
#         if not symptoms:
#             return np.zeros(len(self.symptom_vocab), dtype=np.int8)
        
#         # Initialize binary vector
#         feature_vector = np.zeros(len(self.symptom_vocab), dtype=np.int8)
        
#         # Set 1 for each symptom that exists in vocabulary
#         for symptom in symptoms:
#             if symptom in self.symptom_to_idx:
#                 feature_vector[self.symptom_to_idx[symptom]] = 1
        
#         return feature_vector
    
#     def build_training_matrices(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
#         """
#         Build training matrices X and Y from disease-symptom mapping.
        
#         Returns:
#             Tuple of (X, Y, vocab_list, diseases_list) where:
#             - X: (N x V) binary matrix where each row represents a disease's symptoms
#             - Y: (N x D) multi-hot matrix where each row has one 1 indicating the disease
#             - vocab_list: List of symptom vocabulary
#             - diseases_list: List of disease names
            
#         Raises:
#             ValueError: If no diseases or symptoms are found
#         """
#         print("Building training matrices...")
        
#         # Get diseases and symptoms lists
#         diseases_list = list(self.disease_symptom_map.keys())
#         vocab_list = self.symptom_vocab.copy()
        
#         # Sanity checks
#         if len(diseases_list) == 0:
#             raise ValueError("No diseases found in disease-symptom mapping")
#         if len(vocab_list) == 0:
#             raise ValueError("No symptoms found in vocabulary")
        
#         N = len(diseases_list)  # Number of diseases
#         V = len(vocab_list)     # Vocabulary size
        
#         # Initialize matrices
#         X = np.zeros((N, V), dtype=np.int8)  # Disease-symptom matrix
#         Y = np.zeros((N, N), dtype=np.int8)  # Disease labels (one-hot)
        
#         # Build X matrix: each row represents a disease's symptoms
#         for disease_idx, disease in enumerate(diseases_list):
#             symptoms = self.disease_symptom_map[disease]
            
#             # Set 1 for each symptom this disease has
#             for symptom in symptoms:
#                 if symptom in self.symptom_to_idx:
#                     symptom_idx = self.symptom_to_idx[symptom]
#                     X[disease_idx, symptom_idx] = 1
            
#             # Set corresponding label in Y matrix (one-hot encoding)
#             Y[disease_idx, disease_idx] = 1
        
#         # Additional sanity checks
#         if X.sum() == 0:
#             raise ValueError("Training matrix X is empty - no symptom-disease relationships found")
        
#         if Y.sum() != N:
#             raise ValueError(f"Label matrix Y should have exactly {N} ones, but has {Y.sum()}")
        
#         print(f"✓ Built training matrices: X shape {X.shape}, Y shape {Y.shape}")
#         print(f"✓ X has {X.sum()} symptom-disease relationships")
#         print(f"✓ Y has {Y.sum()} disease labels")
        
#         return X, Y, vocab_list, diseases_list
    
#     def save_meta_artifacts(self, vocab_list: List[str], diseases_list: List[str]) -> None:
#         """
#         Save metadata artifacts including vocabulary, diseases, and configuration.
        
#         Args:
#             vocab_list: List of symptom vocabulary
#             diseases_list: List of disease names
#         """
#         print("Saving meta artifacts...")
        
#         # Create metadata dictionary
#         meta_data = {
#             "vocab": vocab_list,
#             "diseases": diseases_list,
#             "version": "v1",
#             "threshold": 0.3,
#             "severity_thresholds": {
#                 "low": 0.0,
#                 "medium": 0.3,
#                 "high": 0.6,
#                 "critical": 0.8
#             },
#             "stats": {
#                 "num_symptoms": len(vocab_list),
#                 "num_diseases": len(diseases_list),
#                 "total_symptom_disease_pairs": sum(len(symptoms) for symptoms in self.disease_symptom_map.values())
#             }
#         }
        
#         # Save to artifacts directory
#         meta_path = self.artifacts_dir / "meta.json"
#         with open(meta_path, 'w', encoding='utf-8') as f:
#             json.dump(meta_data, f, indent=2, ensure_ascii=False)
        
#         print(f"✓ Saved meta artifacts: {meta_path}")
    
#     def get_feature_stats(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
#         """
#         Get statistics about the feature matrices.
        
#         Args:
#             X: Feature matrix
#             Y: Label matrix
            
#         Returns:
#             Dictionary containing various statistics
#         """
#         stats = {
#             "matrix_shapes": {
#                 "X": list(X.shape),
#                 "Y": list(Y.shape)
#             },
#             "sparsity": {
#                 "X_density": float(X.sum() / X.size),
#                 "Y_density": float(Y.sum() / Y.size)
#             },
#             "symptoms_per_disease": {
#                 "mean": float(X.sum(axis=1).mean()),
#                 "min": int(X.sum(axis=1).min()),
#                 "max": int(X.sum(axis=1).max()),
#                 "std": float(X.sum(axis=1).std())
#             },
#             "diseases_per_symptom": {
#                 "mean": float(X.sum(axis=0).mean()),
#                 "min": int(X.sum(axis=0).min()),
#                 "max": int(X.sum(axis=0).max()),
#                 "std": float(X.sum(axis=0).std())
#             }
#         }
        
#         return stats


# def create_feature_engineer(artifacts_dir: str = "artifacts") -> FeatureEngineer:
#     """
#     Create and return a FeatureEngineer instance.
    
#     Args:
#         artifacts_dir: Directory containing JSON artifacts
        
#     Returns:
#         Configured FeatureEngineer instance
#     """
#     return FeatureEngineer(artifacts_dir)


# def build_feature_vector(symptoms: List[str], artifacts_dir: str = "artifacts") -> np.ndarray:
#     """
#     Convenience function to build feature vector from symptoms.
    
#     Args:
#         symptoms: List of symptom strings
#         artifacts_dir: Directory containing JSON artifacts
        
#     Returns:
#         Binary numpy array representing symptoms
#     """
#     engineer = create_feature_engineer(artifacts_dir)
#     return engineer.build_feature_vector(symptoms)


# def build_training_matrices(artifacts_dir: str = "artifacts") -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
#     """
#     Convenience function to build training matrices.
    
#     Args:
#         artifacts_dir: Directory containing JSON artifacts
        
#     Returns:
#         Tuple of (X, Y, vocab_list, diseases_list)
#     """
#     engineer = create_feature_engineer(artifacts_dir)
#     return engineer.build_training_matrices()


# if __name__ == "__main__":
#     """
#     Test examples demonstrating feature engineering functionality.
#     """
#     print("Testing Feature Engineer...")
#     print("=" * 50)
    
#     try:
#         # Create feature engineer
#         engineer = create_feature_engineer()
        
#         # Test feature vector creation
#         print("\nTesting feature vector creation:")
#         print("-" * 30)
        
#         test_symptoms = [
#             ["fever", "headache"],
#             ["stomach_pain", "nausea"],
#             ["itching", "skin_rash"],
#             [],  # Empty symptoms
#             ["nonexistent_symptom"]  # Symptom not in vocab
#         ]
        
#         for i, symptoms in enumerate(test_symptoms, 1):
#             vector = engineer.build_feature_vector(symptoms)
#             print(f"{i}. Symptoms: {symptoms}")
#             print(f"   Vector shape: {vector.shape}")
#             print(f"   Vector sum: {vector.sum()}")
#             print(f"   Non-zero indices: {np.where(vector == 1)[0].tolist()}")
#             print()
        
#         # Test training matrix creation
#         print("Testing training matrix creation:")
#         print("-" * 35)
        
#         X, Y, vocab_list, diseases_list = engineer.build_training_matrices()
        
#         print(f"X shape: {X.shape}")
#         print(f"Y shape: {Y.shape}")
#         print(f"Vocabulary size: {len(vocab_list)}")
#         print(f"Number of diseases: {len(diseases_list)}")
        
#         # Show some examples
#         print(f"\nFirst 5 diseases: {diseases_list[:5]}")
#         print(f"First 10 symptoms: {vocab_list[:10]}")
        
#         # Show matrix statistics
#         stats = engineer.get_feature_stats(X, Y)
#         print(f"\nMatrix Statistics:")
#         print(f"X density: {stats['sparsity']['X_density']:.3f}")
#         print(f"Y density: {stats['sparsity']['Y_density']:.3f}")
#         print(f"Mean symptoms per disease: {stats['symptoms_per_disease']['mean']:.2f}")
#         print(f"Mean diseases per symptom: {stats['diseases_per_symptom']['mean']:.2f}")
        
#         # Save meta artifacts
#         engineer.save_meta_artifacts(vocab_list, diseases_list)
        
#         # Verify saved artifacts
#         meta_path = Path("artifacts/meta.json")
#         if meta_path.exists():
#             with open(meta_path, 'r') as f:
#                 meta_data = json.load(f)
#             print(f"\n✓ Meta artifacts saved successfully")
#             print(f"Version: {meta_data['version']}")
#             print(f"Threshold: {meta_data['threshold']}")
#             print(f"Severity thresholds: {meta_data['severity_thresholds']}")
        
#         print("\n✓ All feature engineering tests completed successfully!")
        
#     except Exception as e:
#         print(f"❌ Error during testing: {str(e)}")
#         raise


import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple


class FeatureEngineer:
    def __init__(self, artifacts_dir: str = "artifacts", dataset_path: str = "dataset.csv"):
        self.artifacts_dir = Path(artifacts_dir)
        self.dataset_path = Path(dataset_path)

        self.vocab_path = self.artifacts_dir / "symptom_vocab.json"
        self.meta_path = self.artifacts_dir / "meta.json"

        self.symptom_vocab = self._load_vocab()
        self.symptom_to_idx = {symptom: i for i, symptom in enumerate(self.symptom_vocab)}

        print(f"✓ Loaded {len(self.symptom_vocab)} symptoms from vocabulary")

    def _load_vocab(self) -> List[str]:
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        if not isinstance(vocab, list) or not vocab:
            raise ValueError("symptom_vocab.json must contain a non-empty list")

        return vocab

    def _normalize_symptom(self, symptom: str) -> str:
        if pd.isna(symptom):
            return ""
        return str(symptom).strip().lower()

    def build_feature_vector(self, symptoms: List[str]) -> np.ndarray:
        x = np.zeros(len(self.symptom_vocab), dtype=np.int8)

        for symptom in symptoms:
            s = self._normalize_symptom(symptom)
            if s in self.symptom_to_idx:
                x[self.symptom_to_idx[s]] = 1

        return x

    def build_training_matrices(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)
        df.columns = [c.strip() for c in df.columns]

        if "Disease" not in df.columns:
            raise ValueError("dataset.csv must contain 'Disease' column")

        symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
        if not symptom_cols:
            raise ValueError("No Symptom_* columns found in dataset.csv")

        df["Disease"] = df["Disease"].astype(str).str.strip().str.lower()

        class_names = sorted(df["Disease"].unique().tolist())
        class_to_idx = {d: i for i, d in enumerate(class_names)}

        X = np.zeros((len(df), len(self.symptom_vocab)), dtype=np.int8)
        y = np.zeros(len(df), dtype=np.int32)

        for row_idx, row in df.iterrows():
            present = set()

            for col in symptom_cols:
                symptom = self._normalize_symptom(row[col])
                if symptom:
                    present.add(symptom)

            for symptom in present:
                if symptom in self.symptom_to_idx:
                    X[row_idx, self.symptom_to_idx[symptom]] = 1

            y[row_idx] = class_to_idx[row["Disease"]]

        print(f"✓ Built row-level training matrices: X shape {X.shape}, y shape {y.shape}")
        print(f"✓ Number of classes: {len(class_names)}")

        return X, y, class_names


def create_feature_engineer(artifacts_dir: str = "artifacts", dataset_path: str = "dataset.csv") -> FeatureEngineer:
    return FeatureEngineer(artifacts_dir, dataset_path)


def build_feature_vector(symptoms: List[str], artifacts_dir: str = "artifacts") -> np.ndarray:
    engineer = create_feature_engineer(artifacts_dir)
    return engineer.build_feature_vector(symptoms)


def build_training_matrices(artifacts_dir: str = "artifacts", dataset_path: str = "dataset.csv"):
    engineer = create_feature_engineer(artifacts_dir, dataset_path)
    return engineer.build_training_matrices()