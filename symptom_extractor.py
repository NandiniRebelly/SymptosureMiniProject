# import json
# import re
# from pathlib import Path
# from typing import List, Set
# from rapidfuzz import fuzz, process
# from symptom_mapping import SYMPTOM_MAP


# class SymptomExtractor:
#     def __init__(self, vocab_path: str = "artifacts/symptom_vocab.json"):
#         self.vocab_path = Path(vocab_path)
#         self.symptom_vocab = self._load_vocabulary()
#         self.fuzzy_threshold = 86

#     def _load_vocabulary(self) -> List[str]:
#         if not self.vocab_path.exists():
#             raise FileNotFoundError(f"Symptom vocabulary not found: {self.vocab_path}")

#         with open(self.vocab_path, "r", encoding="utf-8") as f:
#             vocab = json.load(f)

#         if not isinstance(vocab, list):
#             raise ValueError("Vocabulary must be a list")

#         print(f"✓ Loaded {len(vocab)} symptoms from vocabulary")
#         return vocab

#     def _normalize_text(self, text: str) -> str:
#         if not text:
#             return ""

#         text = text.lower().strip()

#         # unify separators
#         text = text.replace(",", " ")
#         text = text.replace(".", " ")
#         text = text.replace(";", " ")
#         text = text.replace("/", " ")

#         # keep only words/spaces
#         text = re.sub(r"[^\w\s]", " ", text)
#         text = re.sub(r"\s+", " ", text)

#         return text.strip()

#     def _map_user_text_to_symptoms(self, text: str) -> Set[str]:
#         found = set()

#         for canonical, phrases in SYMPTOM_MAP.items():
#             for phrase in phrases:
#                 phrase = phrase.lower().strip()
#                 pattern = r"\b" + re.escape(phrase) + r"\b"
#                 if re.search(pattern, text):
#                     found.add(canonical)

#         return found

#     def _exact_match_symptoms(self, text: str) -> Set[str]:
#         found = set()

#         for symptom in self.symptom_vocab:
#             phrase = symptom.replace("_", " ").strip()
#             pattern = r"\b" + re.escape(phrase) + r"\b"

#             if re.search(pattern, text):
#                 found.add(symptom)

#         return found

#     def _fuzzy_match_symptoms(self, text: str, exclude: Set[str]) -> Set[str]:
#         found = set()
#         words = text.split()

#         candidates = [s for s in self.symptom_vocab if s not in exclude]
#         candidate_phrases = {s: s.replace("_", " ") for s in candidates}

#         for word in words:
#             if len(word) < 4:
#                 continue

#             best = process.extractOne(
#                 word,
#                 list(candidate_phrases.values()),
#                 scorer=fuzz.ratio,
#                 score_cutoff=self.fuzzy_threshold
#             )

#             if best:
#                 matched_phrase = best[0]
#                 for canonical, phrase in candidate_phrases.items():
#                     if phrase == matched_phrase:
#                         found.add(canonical)
#                         break

#         return found

#     def extract_symptoms(self, text: str) -> List[str]:
#         if not text or not text.strip():
#             return []

#         normalized = self._normalize_text(text)
#         if not normalized:
#             return []

#         mapped = self._map_user_text_to_symptoms(normalized)
#         exact = self._exact_match_symptoms(normalized)
#         fuzzy = self._fuzzy_match_symptoms(normalized, mapped.union(exact))

#         all_matches = mapped.union(exact).union(fuzzy)
#         return sorted(all_matches)


# def create_extractor() -> SymptomExtractor:
#     return SymptomExtractor()


# def extract_symptoms(text: str) -> List[str]:
#     extractor = create_extractor()
#     return extractor.extract_symptoms(text)

import json
import re
from pathlib import Path
from typing import List, Set
from symptom_mapping import SYMPTOM_MAP


class SymptomExtractor:
    def __init__(self, vocab_path: str = "artifacts/symptom_vocab.json"):
        self.vocab_path = Path(vocab_path)
        self.symptom_vocab = self._load_vocabulary()

    def _load_vocabulary(self) -> List[str]:
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Symptom vocabulary not found: {self.vocab_path}")

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        if not isinstance(vocab, list):
            raise ValueError("Vocabulary must be a list")

        print(f"✓ Loaded {len(vocab)} symptoms from vocabulary")
        return vocab

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.lower().strip()

        # normalize separators
        text = text.replace(",", " ")
        text = text.replace(".", " ")
        text = text.replace(";", " ")
        text = text.replace("/", " ")
        text = text.replace("-", " ")

        # normalize common contractions
        text = text.replace("can't", "cant")
        text = text.replace("cannot", "cant")
        text = text.replace("dont", "do not")
        text = text.replace("didnt", "did not")

        # remove non-word chars
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _map_user_text_to_symptoms(self, text: str) -> Set[str]:
        found = set()

        for canonical, phrases in SYMPTOM_MAP.items():
            for phrase in phrases:
                phrase = self._normalize_text(phrase)
                pattern = r"\b" + re.escape(phrase) + r"\b"
                if re.search(pattern, text):
                    found.add(canonical)

        return found

    def _exact_match_symptoms(self, text: str) -> Set[str]:
        found = set()

        for symptom in self.symptom_vocab:
            phrase = symptom.replace("_", " ").strip().lower()
            pattern = r"\b" + re.escape(phrase) + r"\b"
            if re.search(pattern, text):
                found.add(symptom)

        return found

    def extract_symptoms(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        normalized = self._normalize_text(text)
        if not normalized:
            return []

        mapped = self._map_user_text_to_symptoms(normalized)
        exact = self._exact_match_symptoms(normalized)

        all_matches = mapped.union(exact)
        return sorted(all_matches)


def create_extractor() -> SymptomExtractor:
    return SymptomExtractor()


def extract_symptoms(text: str) -> List[str]:
    extractor = create_extractor()
    return extractor.extract_symptoms(text)