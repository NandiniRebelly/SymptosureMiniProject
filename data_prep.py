"""
Data preparation script for symptoms checker project.
Loads and processes CSV files to create normalized artifacts for the ML model.
"""

import pandas as pd
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Any


def normalize_text(text: str) -> str:
    """
    Normalize text by stripping whitespace, converting to lowercase, and removing extra spaces.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string, strip whitespace, lowercase, and remove extra spaces
    normalized = str(text).strip().lower()
    # Replace multiple whitespace with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def load_csv_with_validation(file_path: str, required_columns: List[str] = None) -> pd.DataFrame:
    """
    Load CSV file with defensive checks and helpful error messages.
    
    Args:
        file_path: Path to the CSV file
        required_columns: List of required column names (optional)
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {file_path}: {len(df)} rows, {len(df.columns)} columns")
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {file_path}: {missing_cols}")
        
        return df
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {str(e)}")


def expand_synonyms(symptom_text: str) -> List[str]:
    """
    Expand symptom synonyms by splitting on slashes and parentheses.
    Returns list of individual symptom terms.
    
    Args:
        symptom_text: Raw symptom text that may contain synonyms
        
    Returns:
        List of individual symptom terms
    """
    if pd.isna(symptom_text) or not symptom_text:
        return []
    
    # Normalize the text first
    normalized = normalize_text(symptom_text)
    if not normalized:
        return []
    
    # Split on slashes and parentheses, then clean each part
    synonyms = []
    
    # Split on slashes first
    slash_parts = normalized.split('/')
    for part in slash_parts:
        # Remove parentheses and their contents
        clean_part = re.sub(r'\([^)]*\)', '', part).strip()
        if clean_part:
            synonyms.append(clean_part)
    
    # Also handle parentheses directly (in case no slashes)
    if '/' not in normalized:
        clean_text = re.sub(r'\([^)]*\)', '', normalized).strip()
        if clean_text:
            synonyms.append(clean_text)
    
    return synonyms


def build_symptom_vocabulary(dataset_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build symptom vocabulary with synonym mapping.
    
    Args:
        dataset_df: Main dataset DataFrame
        
    Returns:
        Dictionary mapping synonyms to canonical symptom names
    """
    print("Building symptom vocabulary with synonym expansion...")
    
    symptom_synonyms = {}
    all_symptoms = set()
    
    # Get all symptom columns (assuming they start with "Symptom_")
    symptom_cols = [col for col in dataset_df.columns if col.startswith('Symptom_')]
    
    for _, row in dataset_df.iterrows():
        for col in symptom_cols:
            symptom_text = row[col]
            if pd.notna(symptom_text) and symptom_text:
                # Expand synonyms
                synonyms = expand_synonyms(symptom_text)
                
                for synonym in synonyms:
                    if synonym:  # Skip empty strings
                        all_symptoms.add(synonym)
                        # Map synonym to canonical form (first synonym becomes canonical)
                        if synonym not in symptom_synonyms:
                            symptom_synonyms[synonym] = synonym
    
    print(f"Found {len(all_symptoms)} unique symptoms after synonym expansion")
    return symptom_synonyms


def build_disease_symptom_map(dataset_df: pd.DataFrame, symptom_synonyms: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Build mapping from diseases to their symptoms.
    
    Args:
        dataset_df: Main dataset DataFrame
        symptom_synonyms: Synonym mapping dictionary
        
    Returns:
        Dictionary mapping diseases to lists of symptoms
    """
    print("Building disease-symptom mapping...")
    
    disease_symptoms = {}
    symptom_cols = [col for col in dataset_df.columns if col.startswith('Symptom_')]
    
    for _, row in dataset_df.iterrows():
        disease = normalize_text(row['Disease'])
        if not disease:
            continue
            
        symptoms = []
        for col in symptom_cols:
            symptom_text = row[col]
            if pd.notna(symptom_text) and symptom_text:
                # Expand synonyms and add canonical forms
                synonyms = expand_synonyms(symptom_text)
                for synonym in synonyms:
                    if synonym and synonym in symptom_synonyms:
                        canonical = symptom_synonyms[synonym]
                        if canonical not in symptoms:
                            symptoms.append(canonical)
        
        if symptoms:
            disease_symptoms[disease] = symptoms
    
    print(f"Built mapping for {len(disease_symptoms)} diseases")
    return disease_symptoms


def build_disease_precaution_map(precaution_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build mapping from diseases to their precautions.
    
    Args:
        precaution_df: Precautions DataFrame
        
    Returns:
        Dictionary mapping diseases to lists of precautions
    """
    print("Building disease-precaution mapping...")
    
    disease_precautions = {}
    precaution_cols = [col for col in precaution_df.columns if col.startswith('Precaution_')]
    
    for _, row in precaution_df.iterrows():
        disease = normalize_text(row['Disease'])
        if not disease:
            continue
            
        precautions = []
        for col in precaution_cols:
            precaution = normalize_text(row[col])
            if precaution:
                precautions.append(precaution)
        
        if precautions:
            disease_precautions[disease] = precautions
    
    print(f"Built mapping for {len(disease_precautions)} diseases with precautions")
    return disease_precautions


def build_symptom_severity_map(severity_df: pd.DataFrame, symptom_synonyms: Dict[str, str]) -> Dict[str, float]:
    """
    Build mapping from symptoms to their severity weights.
    
    Args:
        severity_df: Severity DataFrame
        symptom_synonyms: Synonym mapping dictionary
        
    Returns:
        Dictionary mapping symptoms to severity weights
    """
    print("Building symptom-severity mapping...")
    
    symptom_severity = {}
    
    for _, row in severity_df.iterrows():
        symptom_text = row['Symptom']
        weight = row['weight']
        
        if pd.notna(symptom_text) and pd.notna(weight):
            # Expand synonyms and map to canonical forms
            synonyms = expand_synonyms(symptom_text)
            for synonym in synonyms:
                if synonym and synonym in symptom_synonyms:
                    canonical = symptom_synonyms[synonym]
                    symptom_severity[canonical] = float(weight)
    
    print(f"Built severity mapping for {len(symptom_severity)} symptoms")
    return symptom_severity


def build_symptom_description_map(description_df: pd.DataFrame, symptom_synonyms: Dict[str, str]) -> Dict[str, str]:
    """
    Build mapping from symptoms to their descriptions.
    
    Args:
        description_df: Description DataFrame
        symptom_synonyms: Synonym mapping dictionary
        
    Returns:
        Dictionary mapping symptoms to descriptions
    """
    print("Building symptom-description mapping...")
    
    symptom_descriptions = {}
    
    for _, row in description_df.iterrows():
        symptom_text = row['Symptom']
        description = row['Description']
        
        if pd.notna(symptom_text) and pd.notna(description):
            # Expand synonyms and map to canonical forms
            synonyms = expand_synonyms(symptom_text)
            for synonym in synonyms:
                if synonym and synonym in symptom_synonyms:
                    canonical = symptom_synonyms[synonym]
                    symptom_descriptions[canonical] = str(description).strip()
    
    print(f"Built description mapping for {len(symptom_descriptions)} symptoms")
    return symptom_descriptions


def create_meta_info(dataset_df: pd.DataFrame, symptom_df: pd.DataFrame, 
                    precaution_df: pd.DataFrame, severity_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create metadata information about the input files.
    
    Args:
        dataset_df: Main dataset DataFrame
        symptom_df: Symptom description DataFrame
        precaution_df: Precautions DataFrame
        severity_df: Severity DataFrame
        
    Returns:
        Dictionary containing metadata information
    """
    meta_info = {
        "input_files": {
            "dataset.csv": {
                "rows": len(dataset_df),
                "columns": list(dataset_df.columns),
                "symptom_columns": [col for col in dataset_df.columns if col.startswith('Symptom_')]
            },
            "symptom_Description.csv": {
                "rows": len(symptom_df),
                "columns": list(symptom_df.columns)
            },
            "symptom_precaution.csv": {
                "rows": len(precaution_df),
                "columns": list(precaution_df.columns),
                "precaution_columns": [col for col in precaution_df.columns if col.startswith('Precaution_')]
            },
            "Symptom-severity.csv": {
                "rows": len(severity_df),
                "columns": list(severity_df.columns)
            }
        },
        "processing_notes": [
            "Text normalization: strip, lowercase, remove extra whitespace",
            "Synonym expansion: split on slashes and parentheses",
            "Empty/NaN values filtered out from mappings"
        ]
    }
    
    return meta_info


def save_json_artifact(data: Any, file_path: str, description: str) -> None:
    """
    Save data as JSON file with proper formatting.
    
    Args:
        data: Data to save
        file_path: Output file path
        description: Description of what's being saved
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {description}: {file_path}")


def print_summary(disease_symptoms: Dict[str, List[str]], symptom_vocab: Dict[str, str],
                 disease_precautions: Dict[str, List[str]]) -> None:
    """
    Print a summary of the processed data.
    
    Args:
        disease_symptoms: Disease to symptoms mapping
        symptom_vocab: Symptom vocabulary
        disease_precautions: Disease to precautions mapping
    """
    print("\n" + "="*60)
    print("DATA PROCESSING SUMMARY")
    print("="*60)
    
    print(f"Number of diseases: {len(disease_symptoms)}")
    print(f"Number of unique canonical symptoms: {len(symptom_vocab)}")
    print(f"Number of diseases with precautions: {len(disease_precautions)}")
    
    print(f"\nSample disease -> symptoms entries:")
    sample_diseases = list(disease_symptoms.keys())[:5]
    for disease in sample_diseases:
        symptoms = disease_symptoms[disease][:3]  # Show first 3 symptoms
        print(f"  {disease}: {symptoms}")
    
    print(f"\nTotal precautions loaded: {sum(len(precautions) for precautions in disease_precautions.values())}")
    print("="*60)


def main():
    """
    Main function that orchestrates the entire data preparation process.
    """
    print("Starting data preparation for symptoms checker...")
    
    # Define file paths
    base_path = Path(__file__).parent
    dataset_path = base_path / "dataset.csv"
    symptom_desc_path = base_path / "symptom_Description.csv"
    precaution_path = base_path / "symptom_precaution.csv"
    severity_path = base_path / "Symptom-severity.csv"
    
    try:
        # Load all CSV files with validation
        print("\nLoading CSV files...")
        dataset_df = load_csv_with_validation(str(dataset_path), ["Disease"])
        symptom_df = load_csv_with_validation(str(symptom_desc_path), ["Disease", "Description"])
        precaution_df = load_csv_with_validation(str(precaution_path), ["Disease"])
        severity_df = load_csv_with_validation(str(severity_path), ["Symptom", "weight"])
        
        # Fix: symptom_Description.csv uses "Disease" instead of "Symptom"
        if "Disease" in symptom_df.columns and "Symptom" not in symptom_df.columns:
            symptom_df = symptom_df.rename(columns={"Disease": "Symptom"})

        
        # Build symptom vocabulary with synonym expansion
        symptom_synonyms = build_symptom_vocabulary(dataset_df)
        
        # Build all mappings
        disease_symptoms = build_disease_symptom_map(dataset_df, symptom_synonyms)
        disease_precautions = build_disease_precaution_map(precaution_df)
        symptom_severity = build_symptom_severity_map(severity_df, symptom_synonyms)
        symptom_descriptions = build_symptom_description_map(symptom_df, symptom_synonyms)
        
        # Create metadata
        meta_info = create_meta_info(dataset_df, symptom_df, precaution_df, severity_df)
        
        # Save all artifacts
        print("\nSaving artifacts...")
        artifacts_dir = base_path / "artifacts"
        
        save_json_artifact(list(symptom_synonyms.keys()), 
                          artifacts_dir / "symptom_vocab.json", 
                          "symptom vocabulary")
        
        save_json_artifact(disease_symptoms, 
                          artifacts_dir / "disease_symptom_map.json", 
                          "disease-symptom mapping")
        
        save_json_artifact(disease_precautions, 
                          artifacts_dir / "disease_precaution_map.json", 
                          "disease-precaution mapping")
        
        save_json_artifact(symptom_severity, 
                          artifacts_dir / "symptom_severity.json", 
                          "symptom-severity mapping")
        
        save_json_artifact(symptom_descriptions, 
                          artifacts_dir / "symptom_description.json", 
                          "symptom-description mapping")
        
        save_json_artifact(meta_info, 
                          artifacts_dir / "meta_raw.csv_info.json", 
                          "metadata information")
        
        # Print summary
        print_summary(disease_symptoms, symptom_synonyms, disease_precautions)
        
        print("\n✓ Data preparation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during data preparation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
