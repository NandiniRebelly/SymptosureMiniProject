import json


def load_disease_map(file_path="artifacts/disease_symptom_map.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {disease: set(symptoms) for disease, symptoms in data.items()}


# def predict_disease(symptoms, disease_map):
#     user_symptoms = set(symptoms)
#     results = []

#     if not user_symptoms:
#         return results

#     for disease, disease_symptoms in disease_map.items():
#         matched = user_symptoms & disease_symptoms
#         match_count = len(matched)

#         if match_count == 0:
#             continue

#         user_coverage = match_count / len(user_symptoms)
#         disease_coverage = match_count / len(disease_symptoms)
#         missing_user = len(user_symptoms - disease_symptoms)

#         # Strongly prefer diseases that explain more of the user's symptoms
#         score = (0.8 * user_coverage) + (0.2 * disease_coverage) - (0.2 * missing_user)

#         if score > 0:
#             results.append({
#                 "disease": disease,
#                 "score": round(score, 4),
#                 "matched": sorted(matched),
#                 "match_count": match_count,
#                 "missing_count": missing_user
#             })

#     results.sort(
#         key=lambda x: (x["match_count"], -x["missing_count"], x["score"]),
#         reverse=True
#     )
#     return results 

def predict_disease(symptoms, disease_map):
    user_symptoms = set(symptoms)
    results = []

    if not user_symptoms:
        return results

    for disease, disease_symptoms in disease_map.items():
        matched = user_symptoms & disease_symptoms
        match_count = len(matched)

        if match_count == 0:
            continue

        user_coverage = match_count / len(user_symptoms)
        disease_coverage = match_count / len(disease_symptoms)
        missing_user = len(user_symptoms - disease_symptoms)

        # 🔥 NEW STRONG LOGIC
        score = (
            (0.85 * user_coverage) +     # prioritize user's symptoms
            (0.15 * disease_coverage)    # secondary importance
            - (0.25 * missing_user)      # strong penalty
        )

        if score > 0:
            results.append({
                "disease": disease,
                "score": round(score, 4),
                "matched": sorted(matched),
                "match_count": match_count,
                "missing_count": missing_user
            })

    results.sort(
        key=lambda x: (x["match_count"], -x["missing_count"], x["score"]),
        reverse=True
    )

    return results