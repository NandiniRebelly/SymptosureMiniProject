import json


def load_disease_map(file_path="artifacts/disease_symptom_map.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {disease: set(symptoms) for disease, symptoms in data.items()}


def predict_disease(symptoms, disease_map):
    user_symptoms = set(symptoms)
    results = []

    if not user_symptoms:
        return results

    for disease, disease_symptoms in disease_map.items():
        matched = user_symptoms & disease_symptoms
        match_count = len(matched)

        # ignore weak matches
        if match_count < 2:
            continue

        missing_user = len(user_symptoms - disease_symptoms)

        user_coverage = match_count / len(user_symptoms)
        disease_coverage = match_count / len(disease_symptoms)

        # balanced score
        score = (
            (0.75 * user_coverage) +
            (0.25 * disease_coverage) -
            (0.25 * missing_user)
        )

        # small bonus for exact/full match
        if missing_user == 0:
            score += 0.10

        # keep score realistic
        score = max(0.0, min(score, 0.95))

        if score > 0:
            results.append({
                "disease": disease,
                "score": round(score, 4),
                "matched": sorted(matched),
                "match_count": match_count,
                "missing_count": missing_user,
                "user_coverage": round(user_coverage, 4),
                "disease_coverage": round(disease_coverage, 4),
            })

    results.sort(
        key=lambda x: (
            x["missing_count"] == 0,  # exact/full matches first
            x["match_count"],         # then more matched symptoms
            -x["missing_count"],      # then fewer missing symptoms
            x["score"]                # then higher score
        ),
        reverse=True
    )

    return results