# """
# Model training script for symptoms checker project.
# Trains ML models and provides prediction functionality.
# """

# import json
# import joblib
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from typing import List, Tuple, Dict, Any
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.metrics import (
#     precision_score, recall_score, f1_score, 
#     hamming_loss, classification_report
# )
# from tabulate import tabulate

# # Import our custom modules
# from feature_engineer import build_training_matrices
# from symptom_extractor import extract_symptoms


# class SymptomPredictor:
#     """
#     Handles model training, evaluation, and prediction for symptom-based disease diagnosis.
#     """
    
#     def __init__(self, artifacts_dir: str = "artifacts"):
#         """
#         Initialize the symptom predictor.
        
#         Args:
#             artifacts_dir: Directory containing artifacts
#         """
#         self.artifacts_dir = Path(artifacts_dir)
#         self.model_path = self.artifacts_dir / "model.joblib"
#         self.meta_path = self.artifacts_dir / "meta.json"
        
#         # Load metadata
#         self.meta_data = self._load_meta_data()
#         self.vocab = self.meta_data.get("vocab", [])
#         self.diseases = self.meta_data.get("diseases", [])
#         self.threshold = self.meta_data.get("threshold", 0.3)
        
#         # Model and feature engineer
#         self.model = None
#         self.feature_engineer = None
        
#         print(f"✓ Loaded metadata: {len(self.vocab)} symptoms, {len(self.diseases)} diseases")
    
#     def _load_meta_data(self) -> Dict[str, Any]:
#         """
#         Load metadata from JSON file.
        
#         Returns:
#             Dictionary containing metadata
#         """
#         if not self.meta_path.exists():
#             raise FileNotFoundError(f"Meta data not found: {self.meta_path}")
        
#         with open(self.meta_path, 'r', encoding='utf-8') as f:
#             return json.load(f)
    
#     def _create_model(self) -> OneVsRestClassifier:
#         """
#         Create the ML model pipeline.
        
#         Returns:
#             Configured OneVsRestClassifier with RandomForestClassifier
#         """
#         # Base classifier with balanced class weights and parallel processing
#         base_classifier = RandomForestClassifier(
#             n_estimators=200,
#             class_weight='balanced',
#             n_jobs=-1,
#             random_state=42
#         )
        
#         # Wrap in OneVsRest for multi-label classification
#         model = OneVsRestClassifier(base_classifier)
        
#         print("✓ Created OneVsRestClassifier with RandomForestClassifier")
#         return model
    
#     def train_model(self) -> Dict[str, Any]:
#         """
#         Train the ML model and evaluate performance.
        
#         Returns:
#             Dictionary containing training results and metrics
#         """
#         print("\n" + "="*60)
#         print("TRAINING MODEL")
#         print("="*60)
        
#         # Load training data
#         print("Loading training matrices...")
#         X, Y, vocab_list, diseases_list = build_training_matrices(str(self.artifacts_dir))
        
#         # Store feature engineer for later use
#         from feature_engineer import create_feature_engineer
#         self.feature_engineer = create_feature_engineer(str(self.artifacts_dir))
        
#         print(f"Training data: X shape {X.shape}, Y shape {Y.shape}")
        
#         # Split data into train/test sets (80/20 split with fixed seed)
#         print("Splitting data into train/test sets (80/20)...")
#         X_train, X_test, Y_train, Y_test = train_test_split(
#             X, Y, test_size=0.2, random_state=42, stratify=None
#         )
        
#         print(f"Train set: X_train {X_train.shape}, Y_train {Y_train.shape}")
#         print(f"Test set: X_test {X_test.shape}, Y_test {Y_test.shape}")
        
#         # Create and train model
#         print("\nTraining model...")
#         self.model = self._create_model()
        
#         # Train the model
#         self.model.fit(X_train, Y_train)
#         print("✓ Model training completed")
        
#         # Make predictions on test set
#         print("Making predictions on test set...")
#         Y_pred = self.model.predict(X_test)
#         Y_pred_proba = self.model.predict_proba(X_test)
        
#         # Compute evaluation metrics
#         print("Computing evaluation metrics...")
#         metrics = self._compute_metrics(Y_test, Y_pred, Y_pred_proba)
        
#         # Save trained model
#         self._save_model()
        
#         # Update metadata with training results
#         self._update_meta_data(metrics)
        
#         return metrics
    
#     def _compute_metrics(self, Y_true: np.ndarray, Y_pred: np.ndarray, Y_pred_proba: np.ndarray) -> Dict[str, Any]:
#         """
#         Compute comprehensive evaluation metrics.
        
#         Args:
#             Y_true: True labels
#             Y_pred: Predicted labels
#             Y_pred_proba: Predicted probabilities
            
#         Returns:
#             Dictionary containing all computed metrics
#         """
#         # Overall metrics
#         hamming_loss_score = hamming_loss(Y_true, Y_pred)
#         macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
#         micro_f1 = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
        
#         # Per-disease metrics
#         precision_per_disease = precision_score(Y_true, Y_pred, average=None, zero_division=0)
#         recall_per_disease = recall_score(Y_true, Y_pred, average=None, zero_division=0)
#         f1_per_disease = f1_score(Y_true, Y_pred, average=None, zero_division=0)
        
#         # Create metrics summary
#         metrics = {
#             "overall": {
#                 "hamming_loss": float(hamming_loss_score),
#                 "macro_f1": float(macro_f1),
#                 "micro_f1": float(micro_f1)
#             },
#             "per_disease": {
#                 "precision": precision_per_disease.tolist(),
#                 "recall": recall_per_disease.tolist(),
#                 "f1_score": f1_per_disease.tolist()
#             }
#         }
        
#         # Print metrics table
#         self._print_metrics_table(metrics)
        
#         return metrics
    
#     def _print_metrics_table(self, metrics: Dict[str, Any]) -> None:
#         """
#         Print a neat table of evaluation metrics.
        
#         Args:
#             metrics: Dictionary containing computed metrics
#         """
#         print("\n" + "="*80)
#         print("EVALUATION METRICS")
#         print("="*80)
        
#         # Overall metrics
#         overall = metrics["overall"]
#         print(f"Hamming Loss: {overall['hamming_loss']:.4f}")
#         print(f"Macro F1-Score: {overall['macro_f1']:.4f}")
#         print(f"Micro F1-Score: {overall['micro_f1']:.4f}")
        
#         # Per-disease metrics table
#         print(f"\nPer-Disease Metrics:")
#         print("-" * 80)
        
#         # Prepare data for table
#         table_data = []
#         for i, disease in enumerate(self.diseases):
#             precision = metrics["per_disease"]["precision"][i]
#             recall = metrics["per_disease"]["recall"][i]
#             f1 = metrics["per_disease"]["f1_score"][i]
            
#             table_data.append([
#                 disease[:30] + "..." if len(disease) > 30 else disease,
#                 f"{precision:.3f}",
#                 f"{recall:.3f}",
#                 f"{f1:.3f}"
#             ])
        
#         # Print table
#         headers = ["Disease", "Precision", "Recall", "F1-Score"]
#         print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
#         # Summary statistics
#         precision_scores = metrics["per_disease"]["precision"]
#         recall_scores = metrics["per_disease"]["recall"]
#         f1_scores = metrics["per_disease"]["f1_score"]
        
#         print(f"\nSummary Statistics:")
#         print(f"Mean Precision: {np.mean(precision_scores):.3f}")
#         print(f"Mean Recall: {np.mean(recall_scores):.3f}")
#         print(f"Mean F1-Score: {np.mean(f1_scores):.3f}")
#         print("="*80)
    
#     def _save_model(self) -> None:
#         """
#         Save the trained model and metadata to artifacts directory.
#         """
#         print("Saving trained model...")
        
#         # Prepare model data for saving
#         model_data = {
#             "model": self.model,
#             "feature_engineer": self.feature_engineer,
#             "vocab": self.vocab,
#             "diseases": self.diseases,
#             "threshold": self.threshold,
#             "meta_data": self.meta_data
#         }
        
#         # Save to joblib file
#         joblib.dump(model_data, self.model_path)
#         print(f"✓ Model saved to: {self.model_path}")
    
#     def _update_meta_data(self, metrics: Dict[str, Any]) -> None:
#         """
#         Update metadata with training results.
        
#         Args:
#             metrics: Dictionary containing training metrics
#         """
#         print("Updating metadata...")
        
#         # Update metadata with training results
#         self.meta_data.update({
#             "vocab": self.vocab,
#             "diseases": self.diseases,
#             "threshold": self.threshold,
#             "version": "v1",
#             "best_params": "RandomForestClassifier(n_estimators=200, class_weight='balanced')",
#             "training_metrics": {
#                 "hamming_loss": metrics["overall"]["hamming_loss"],
#                 "macro_f1": metrics["overall"]["macro_f1"],
#                 "micro_f1": metrics["overall"]["micro_f1"]
#             }
#         })
        
#         # Save updated metadata
#         with open(self.meta_path, 'w', encoding='utf-8') as f:
#             json.dump(self.meta_data, f, indent=2, ensure_ascii=False)
        
#         print(f"✓ Metadata updated: {self.meta_path}")
    
#     def predict_from_text(self, text: str, lang: str = "en") -> List[Tuple[str, float]]:
#         """
#         Predict diseases from text input.
        
#         Args:
#             text: Input text containing symptoms
#             lang: Language code (currently only English supported)
            
#         Returns:
#             List of tuples (disease_name, probability) sorted by probability (descending)
#         """
#         if not self.model:
#             raise ValueError("Model not trained. Call train_model() first.")
        
#         # Extract symptoms from text
#         symptoms = extract_symptoms(text)
        
#         if not symptoms:
#             return []
        
#         # Build feature vector
#         feature_vector = self.feature_engineer.build_feature_vector(symptoms)
        
#         # Reshape for prediction (sklearn expects 2D array)
#         feature_vector = feature_vector.reshape(1, -1)
        
#         # Get prediction probabilities
#         probabilities = self.model.predict_proba(feature_vector)[0]
        
#         # Create disease-probability pairs
#         disease_probs = [(disease, prob) for disease, prob in zip(self.diseases, probabilities)]
        
#         # Sort by probability (descending) and filter by threshold
#         disease_probs.sort(key=lambda x: x[1], reverse=True)
        
#         # Return top predictions above threshold
#         top_predictions = [(disease, prob) for disease, prob in disease_probs if prob >= self.threshold]
        
#         return top_predictions[:3]  # Return top 3 predictions
    
#     def load_model(self) -> None:
#         """
#         Load a previously trained model from artifacts directory.
#         """
#         if not self.model_path.exists():
#             raise FileNotFoundError(f"Model not found: {self.model_path}")
        
#         print("Loading trained model...")
#         model_data = joblib.load(self.model_path)
        
#         self.model = model_data["model"]
#         self.feature_engineer = model_data["feature_engineer"]
#         self.vocab = model_data["vocab"]
#         self.diseases = model_data["diseases"]
#         self.threshold = model_data["threshold"]
        
#         print("✓ Model loaded successfully")


# def train_and_evaluate_model(artifacts_dir: str = "artifacts") -> SymptomPredictor:
#     """
#     Train and evaluate the symptom prediction model.
    
#     Args:
#         artifacts_dir: Directory containing artifacts
        
#     Returns:
#         Trained SymptomPredictor instance
#     """
#     predictor = SymptomPredictor(artifacts_dir)
#     predictor.train_model()
#     return predictor


# def predict_from_text(text: str, lang: str = "en", artifacts_dir: str = "artifacts") -> List[Tuple[str, float]]:
#     """
#     Convenience function to predict diseases from text.
    
#     Args:
#         text: Input text containing symptoms
#         lang: Language code
#         artifacts_dir: Directory containing artifacts
        
#     Returns:
#         List of tuples (disease_name, probability)
#     """
#     predictor = SymptomPredictor(artifacts_dir)
#     predictor.load_model()
#     return predictor.predict_from_text(text, lang)


# if __name__ == "__main__":
#     """
#     Main execution: train model and test prediction.
#     """
#     print("Symptom Checker - Model Training")
#     print("=" * 50)
    
#     try:
#         # Train the model
#         predictor = train_and_evaluate_model()
        
#         # Test prediction with sample text
#         print("\n" + "="*60)
#         print("TESTING PREDICTION")
#         print("="*60)
        
#         test_text = "i have fever and sore throat and chills"
#         print(f"Input text: \"{test_text}\"")
        
#         # Extract symptoms first
#         symptoms = extract_symptoms(test_text)
#         print(f"Extracted symptoms: {symptoms}")
        
#         # Make prediction
#         predictions = predictor.predict_from_text(test_text)
        
#         print(f"\nTop-5 Disease Predictions:")
#         print("-" * 40)
        
#         if predictions:
#             for i, (disease, probability) in enumerate(predictions, 1):
#                 print(f"{i}. {disease}: {probability:.3f}")
#         else:
#             print("No diseases predicted above threshold.")
        
#         # Also show top 5 predictions regardless of threshold
#         print(f"\nTop-5 Disease Predictions (all probabilities):")
#         print("-" * 50)
        
#         # Get all predictions sorted by probability
#         symptoms = extract_symptoms(test_text)
#         feature_vector = predictor.feature_engineer.build_feature_vector(symptoms)
#         feature_vector = feature_vector.reshape(1, -1)
#         probabilities = predictor.model.predict_proba(feature_vector)[0]
        
#         all_predictions = [(disease, prob) for disease, prob in zip(predictor.diseases, probabilities)]
#         all_predictions.sort(key=lambda x: x[1], reverse=True)
        
#         for i, (disease, probability) in enumerate(all_predictions[:5], 1):
#             print(f"{i}. {disease}: {probability:.3f}")
        
#         print("\n✓ Model training and testing completed successfully!")
        
#     except Exception as e:
#         print(f"❌ Error during training/testing: {str(e)}")
#         raise



# #2nd one accuracy is 1
# import json
# import joblib
# from pathlib import Path
# from typing import Dict, Any

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, f1_score
# from sklearn.model_selection import train_test_split
# from tabulate import tabulate

# from feature_engineer import build_training_matrices


# class SymptomPredictor:
#     def __init__(self, artifacts_dir: str = "artifacts", dataset_path: str = "dataset.csv"):
#         self.artifacts_dir = Path(artifacts_dir)
#         self.dataset_path = dataset_path
#         self.model_path = self.artifacts_dir / "model.joblib"
#         self.meta_path = self.artifacts_dir / "meta.json"

#         self.meta_data = self._load_meta_data()
#         self.model = None

#     def _load_meta_data(self) -> Dict[str, Any]:
#         if not self.meta_path.exists():
#             raise FileNotFoundError(f"Meta file not found: {self.meta_path}")

#         with open(self.meta_path, "r", encoding="utf-8") as f:
#             return json.load(f)

#     def _save_meta_data(self):
#         with open(self.meta_path, "w", encoding="utf-8") as f:
#             json.dump(self.meta_data, f, indent=2, ensure_ascii=False)

#     def _create_model(self) -> RandomForestClassifier:
#         model = RandomForestClassifier(
#             n_estimators=300,
#             max_depth=None,
#             min_samples_split=2,
#             min_samples_leaf=1,
#             class_weight="balanced",
#             random_state=42,
#             n_jobs=-1
#         )
#         print("✓ Created RandomForestClassifier")
#         return model

#     def _save_model(self, class_names):
#         payload = {
#             "model": self.model,
#             "class_names": class_names
#         }
#         joblib.dump(payload, self.model_path)
#         print(f"✓ Saved model to {self.model_path}")

#     def train_model(self) -> Dict[str, Any]:
#         print("\n" + "=" * 60)
#         print("TRAINING MODEL")
#         print("=" * 60)

#         X, y, class_names = build_training_matrices(str(self.artifacts_dir), self.dataset_path)

#         print(f"Training data: X shape {X.shape}, y shape {y.shape}")

#         X_train, X_test, y_train, y_test = train_test_split(
#             X,
#             y,
#             test_size=0.2,
#             random_state=42,
#             stratify=y
#         )

#         print(f"Train set: X_train {X_train.shape}, y_train {y_train.shape}")
#         print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")

#         self.model = self._create_model()
#         self.model.fit(X_train, y_train)
#         print("✓ Model training completed")

#         y_pred = self.model.predict(X_test)

#         accuracy = accuracy_score(y_test, y_pred)
#         macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
#         weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

#         print("\n" + "=" * 60)
#         print("EVALUATION METRICS")
#         print("=" * 60)
#         print(f"Accuracy      : {accuracy:.4f}")
#         print(f"Macro F1      : {macro_f1:.4f}")
#         print(f"Weighted F1   : {weighted_f1:.4f}")

#         report = classification_report(
#             y_test,
#             y_pred,
#             target_names=class_names,
#             zero_division=0,
#             output_dict=True
#         )

#         rows = []
#         for disease in class_names:
#             if disease in report:
#                 rows.append([
#                     disease,
#                     round(report[disease]["precision"], 4),
#                     round(report[disease]["recall"], 4),
#                     round(report[disease]["f1-score"], 4),
#                     int(report[disease]["support"])
#                 ])

#         print("\nPer-Disease Metrics:")
#         print(tabulate(
#             rows,
#             headers=["Disease", "Precision", "Recall", "F1", "Support"],
#             tablefmt="grid"
#         ))

#         self._save_model(class_names)

#         self.meta_data["training_metrics"] = {
#             "accuracy": accuracy,
#             "macro_f1": macro_f1,
#             "weighted_f1": weighted_f1
#         }
#         self.meta_data["model_type"] = "RandomForestClassifier"
#         self.meta_data["num_classes"] = len(class_names)
#         self._save_meta_data()

#         return {
#             "accuracy": accuracy,
#             "macro_f1": macro_f1,
#             "weighted_f1": weighted_f1
#         }


# if __name__ == "__main__":
#     predictor = SymptomPredictor()
#     predictor.train_model()





# #USE THIS ALSO NO PRBM FIRST USE OF 2ND USE THIS OKK 
# """
# Improved train_model.py for SymptoSure.
# """

# import json
# import warnings
# import joblib
# import numpy as np
# from pathlib import Path
# from typing import Dict, Any, Tuple

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.exceptions import UndefinedMetricWarning
# from tabulate import tabulate

# from feature_engineer import build_training_matrices


# warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._classification")
# warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.multiclass")
# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# class SymptomPredictor:
#     def __init__(self, artifacts_dir: str = "artifacts", dataset_path: str = "dataset.csv"):
#         self.artifacts_dir = Path(artifacts_dir)
#         self.dataset_path = dataset_path
#         self.model_path = self.artifacts_dir / "model.joblib"
#         self.meta_path = self.artifacts_dir / "meta.json"
#         self.model = None
#         self.meta_data = self._load_meta()
#         self.vocab = self.meta_data.get("vocab", [])
#         self.diseases = self.meta_data.get("diseases", [])

#     def _load_meta(self) -> Dict[str, Any]:
#         if not self.meta_path.exists():
#             return {}
#         with open(self.meta_path, "r", encoding="utf-8") as f:
#             return json.load(f)

#     def _save_meta(self) -> None:
#         with open(self.meta_path, "w", encoding="utf-8") as f:
#             json.dump(self.meta_data, f, indent=2, ensure_ascii=False)

#     @staticmethod
#     def _deduplicate(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         seen = {}
#         keep = []
#         for i, row in enumerate(X):
#             key = (tuple(row.astype(int).tolist()), int(y[i]))
#             if key not in seen:
#                 seen[key] = True
#                 keep.append(i)
#         keep = np.array(keep)
#         return X[keep], y[keep]

#     @staticmethod
#     def _make_groups(X: np.ndarray) -> np.ndarray:
#         return np.array(["|".join(map(str, row.astype(int).tolist())) for row in X])

#     def _create_model(self) -> RandomForestClassifier:
#         return RandomForestClassifier(
#             n_estimators=300,
#             class_weight="balanced",
#             random_state=42,
#             n_jobs=-1,
#         )

#     def train_model(self) -> Dict[str, Any]:
#         print("\n" + "=" * 60)
#         print("TRAINING MODEL")
#         print("=" * 60)

#         X, y, class_names = build_training_matrices(str(self.artifacts_dir), self.dataset_path)
#         print(f"Raw training data: X shape {X.shape}, y shape {y.shape}")

#         X, y = self._deduplicate(X, y)
#         print(f"After deduplication: X shape {X.shape}, y shape {y.shape}")

#         groups = self._make_groups(X)
#         splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#         train_idx, test_idx = next(splitter.split(X, y, groups=groups))

#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]

#         print(f"Train set: X_train {X_train.shape}, y_train {y_train.shape}")
#         print(f"Test set : X_test  {X_test.shape}, y_test  {y_test.shape}")

#         self.model = self._create_model()
#         self.model.fit(X_train, y_train)
#         print("✓ Model training completed")

#         y_pred = self.model.predict(X_test)
#         y_prob = self.model.predict_proba(X_test)

#         accuracy = accuracy_score(y_test, y_pred)
#         macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
#         weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
#         top3_accuracy = top_k_accuracy_score(y_test, y_prob, k=min(3, len(class_names)), labels=list(range(len(class_names))))

#         labels = list(range(len(class_names)))
#         report = classification_report(
#             y_test,
#             y_pred,
#             labels=labels,
#             target_names=class_names,
#             zero_division=0,
#             output_dict=True,
#         )

#         print("\n" + "=" * 60)
#         print("EXACT-PATTERN HOLDOUT METRICS")
#         print("=" * 60)
#         print(f"Accuracy      : {accuracy:.4f}")
#         print(f"Macro F1      : {macro_f1:.4f}")
#         print(f"Weighted F1   : {weighted_f1:.4f}")
#         print(f"Top-3 Accuracy: {top3_accuracy:.4f}")

#         rows = []
#         for cname in class_names:
#             item = report.get(cname)
#             if item:
#                 rows.append([
#                     cname,
#                     f"{item['precision']:.3f}",
#                     f"{item['recall']:.3f}",
#                     f"{item['f1-score']:.3f}",
#                     int(item['support']),
#                 ])

#         print("\nPer-Disease Metrics:")
#         print(tabulate(rows, headers=["Disease", "Precision", "Recall", "F1", "Support"], tablefmt="grid"))

#         joblib.dump(
#             {
#                 "model": self.model,
#                 "class_names": class_names,
#                 "vocab": self.vocab,
#                 "diseases": class_names,
#                 "metrics": {
#                     "exact_holdout_accuracy": float(accuracy),
#                     "exact_holdout_macro_f1": float(macro_f1),
#                     "exact_holdout_weighted_f1": float(weighted_f1),
#                     "exact_holdout_top3_accuracy": float(top3_accuracy),
#                 },
#                 "split_info": {
#                     "method": "GroupShuffleSplit on exact symptom signature after deduplication",
#                     "train_size": int(len(train_idx)),
#                     "test_size": int(len(test_idx)),
#                 },
#             },
#             self.model_path,
#         )
#         print(f"✓ Model saved to {self.model_path}")

#         self.meta_data["training_metrics"] = {
#             "exact_holdout_accuracy": float(accuracy),
#             "exact_holdout_macro_f1": float(macro_f1),
#             "exact_holdout_weighted_f1": float(weighted_f1),
#             "exact_holdout_top3_accuracy": float(top3_accuracy),
#         }
#         self.meta_data["evaluation_note"] = (
#             "Exact-pattern holdout can remain optimistic on this dataset because many rows are deterministic symptom signatures. "
#             "Use evaluate_model.py masked-holdout metrics as the main reported performance."
#         )
#         self._save_meta()
#         print(f"✓ Metadata updated: {self.meta_path}")

#         return self.meta_data["training_metrics"]


# if __name__ == "__main__":
#     SymptomPredictor().train_model()


import json
import warnings
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.exceptions import UndefinedMetricWarning
from tabulate import tabulate
from feature_engineer import build_training_matrices

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class SymptomPredictor:
    def __init__(self, artifacts_dir: str = "artifacts", dataset_path: str = "dataset.csv"):
        self.artifacts_dir = Path(artifacts_dir)
        self.dataset_path = dataset_path
        self.model_path = self.artifacts_dir / "model.joblib"
        self.meta_path = self.artifacts_dir / "meta.json"
        self.model = None
        self.meta_data = self._load_meta()
        self.vocab = self.meta_data.get("vocab", [])
        self.diseases = self.meta_data.get("diseases", [])

    def _load_meta(self) -> Dict[str, Any]:
        if not self.meta_path.exists():
            return {}
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_meta(self) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _deduplicate(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seen = {}
        keep = []
        for i, row in enumerate(X):
            key = (tuple(row.astype(int).tolist()), int(y[i]))
            if key not in seen:
                seen[key] = True
                keep.append(i)
        keep = np.array(keep)
        return X[keep], y[keep]

    @staticmethod
    def _make_groups(X: np.ndarray) -> np.ndarray:
        return np.array(["|".join(map(str, row.astype(int).tolist())) for row in X])

    def _augment_training_only(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        augmented_X = [X]
        augmented_y = [y]
        masked_rows, masked_labels = [], []
        noisy_rows, noisy_labels = [], []
        for i in range(X.shape[0]):
            row = X[i].copy()
            label = y[i]
            ones = np.where(row == 1)[0]
            zeros = np.where(row == 0)[0]
            if len(ones) >= 3:
                row_masked = row.copy()
                drop_idx = rng.choice(ones, size=1, replace=False)
                row_masked[drop_idx] = 0
                masked_rows.append(row_masked)
                masked_labels.append(label)
            if len(zeros) > 0:
                row_noisy = row.copy()
                add_idx = rng.choice(zeros, size=1, replace=False)
                row_noisy[add_idx] = 1
                noisy_rows.append(row_noisy)
                noisy_labels.append(label)
        if masked_rows:
            augmented_X.append(np.array(masked_rows))
            augmented_y.append(np.array(masked_labels))
        if noisy_rows:
            augmented_X.append(np.array(noisy_rows))
            augmented_y.append(np.array(noisy_labels))
        return np.vstack(augmented_X), np.concatenate(augmented_y)

    def _create_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=350,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    def train_model(self) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        X, y, class_names = build_training_matrices(str(self.artifacts_dir), self.dataset_path)
        print(f"Raw training data: X shape {X.shape}, y shape {y.shape}")
        X, y = self._deduplicate(X, y)
        print(f"After deduplication: X shape {X.shape}, y shape {y.shape}")
        groups = self._make_groups(X)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"Train set before augmentation: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"Test set                  : X_test  {X_test.shape}, y_test  {y_test.shape}")
        X_train_aug, y_train_aug = self._augment_training_only(X_train, y_train, seed=42)
        print(f"Train set after augmentation : X_train {X_train_aug.shape}, y_train {y_train_aug.shape}")
        self.model = self._create_model()
        self.model.fit(X_train_aug, y_train_aug)
        print("✓ Model training completed")
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        top3_accuracy = top_k_accuracy_score(y_test, y_prob, k=min(3, len(class_names)), labels=list(range(len(class_names))))
        labels = list(range(len(class_names)))
        report = classification_report(y_test, y_pred, labels=labels, target_names=class_names, zero_division=0, output_dict=True)
        print("\n" + "=" * 60)
        print("EXACT-PATTERN HOLDOUT METRICS")
        print("=" * 60)
        print(f"Accuracy      : {accuracy:.4f}")
        print(f"Macro F1      : {macro_f1:.4f}")
        print(f"Weighted F1   : {weighted_f1:.4f}")
        print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
        rows = []
        for cname in class_names:
            item = report.get(cname)
            if item:
                rows.append([cname, f"{item['precision']:.3f}", f"{item['recall']:.3f}", f"{item['f1-score']:.3f}", int(item['support'])])
        print("\nPer-Disease Metrics:")
        print(tabulate(rows, headers=["Disease", "Precision", "Recall", "F1", "Support"], tablefmt="grid"))
        joblib.dump({
            "model": self.model,
            "class_names": class_names,
            "vocab": self.vocab,
            "diseases": class_names,
            "metrics": {
                "exact_holdout_accuracy": float(accuracy),
                "exact_holdout_macro_f1": float(macro_f1),
                "exact_holdout_weighted_f1": float(weighted_f1),
                "exact_holdout_top3_accuracy": float(top3_accuracy),
            },
            "split_info": {
                "method": "GroupShuffleSplit on exact symptom signature after deduplication",
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "augmentation": "training-only masked + noisy symptom variants",
            },
        }, self.model_path)
        print(f"✓ Model saved to {self.model_path}")
        self.meta_data["training_metrics"] = {
            "exact_holdout_accuracy": float(accuracy),
            "exact_holdout_macro_f1": float(macro_f1),
            "exact_holdout_weighted_f1": float(weighted_f1),
            "exact_holdout_top3_accuracy": float(top3_accuracy),
        }
        self.meta_data["evaluation_note"] = "Exact-pattern holdout can remain optimistic on this dataset. Use evaluate_model.py masked-holdout and grouped-CV metrics as the main reported performance."
        self._save_meta()
        print(f"✓ Metadata updated: {self.meta_path}")
        return self.meta_data["training_metrics"]

if __name__ == "__main__":
    SymptomPredictor().train_model()
