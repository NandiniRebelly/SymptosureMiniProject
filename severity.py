import json

def load_severity(file_path="artifacts/symptom_severity.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_severity(symptoms, severity_dict):
    if not symptoms:
        return "Low"

    weights = [severity_dict.get(s, 0) for s in symptoms if s in severity_dict]
    if not weights:
        return "Low"

    avg = sum(weights) / len(weights)
    max_w = max(weights)

    if max_w >= 7 or avg >= 6:
        return "High"
    elif max_w >= 5 or avg >= 4:
        return "Medium"
    else:
        return "Low"