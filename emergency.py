def is_emergency(symptoms):
    red_flags = {
        "chest_pain",
        "breathlessness",
        "coma",
        "altered_sensorium",
        "stomach_bleeding",
        "acute_liver_failure",
        "weakness_of_one_body_side" 
    }
    return any(symptom in red_flags for symptom in symptoms) 