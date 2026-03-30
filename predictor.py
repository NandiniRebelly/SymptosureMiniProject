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

#         # 🔥 NEW STRONG LOGIC
#         score = (
#             (0.85 * user_coverage) +     # prioritize user's symptoms
#             (0.15 * disease_coverage)    # secondary importance
#             - (0.25 * missing_user)      # strong penalty
#         )

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




# # #SEEE THE BELOW ONE IS CORRECT ONLY.
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

#         # softer penalty
#         score = (
#             (0.75 * user_coverage) +
#             (0.25 * disease_coverage) -
#             (0.08 * missing_user)
#         )

#         # keep diseases with at least 2 matched symptoms
#         if match_count >= 2 and score > 0:
#             results.append({
#                 "disease": disease,
#                 "score": round(score, 4),
#                 "matched": sorted(matched),
#                 "match_count": match_count,
#                 "missing_count": missing_user
#             })

#     results.sort(
#         key=lambda x: (x["match_count"], x["score"], -x["missing_count"]),
#         reverse=True
#     )

#     return results



# import json

# def load_disease_map(file_path="artifacts/disease_symptom_map.json"):
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     return {disease: set(symptoms) for disease, symptoms in data.items()}


# def predict_disease(symptoms, disease_map):
#     user_symptoms = set(symptoms)
#     results = []

#     if not user_symptoms:
#         return results

#     for disease, disease_symptoms in disease_map.items():
#         matched = user_symptoms & disease_symptoms
#         match_count = len(matched)

#         # Ignore weak matches
#         if match_count < 2:
#             continue

#         missing_user = len(user_symptoms - disease_symptoms)

#         user_coverage = match_count / len(user_symptoms)
#         disease_coverage = match_count / len(disease_symptoms)

#         # 🔥 Balanced scoring
#         score = (
#             (0.80 * user_coverage) +
#             (0.20 * disease_coverage) -
#             (0.35 * missing_user)
#         )

#         # Bonus for exact match
#         if missing_user == 0:
#             score += 0.25

#         # ✅ CRITICAL FIX → clamp score
#         score = max(0.0, min(score, 1.0))

#         if score > 0:
#             results.append({
#                 "disease": disease,
#                 "score": round(score, 4),
#                 "matched": sorted(matched),
#                 "match_count": match_count,
#                 "missing_count": missing_user,
#                 "user_coverage": round(user_coverage, 4),
#                 "disease_coverage": round(disease_coverage, 4),
#             })

#     # 🔥 Smart sorting
#     results.sort(
#         key=lambda x: (
#             x["missing_count"] == 0,  # exact match first
#             x["match_count"],         # more matches
#             -x["missing_count"],      # fewer missing
#             x["score"]                # higher score
#         ),
#         reverse=True
#     )

#     return results

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