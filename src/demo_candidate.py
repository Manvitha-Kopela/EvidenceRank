from parser import load_candidates
from features import *
from scorer import calculate_scores
from explain import explain_candidate

TARGET = "CAND_0088025"

candidates = load_candidates("../data/raw/candidates.jsonl")

for candidate in candidates:
    if candidate["candidate_id"] == TARGET:
        q = qualification_features(candidate)
        b = behavioral_features(candidate)
        c = contradiction_features(candidate)
        conf = confidence_features(candidate)

        scores = calculate_scores(q, b, c, conf)

        explanation = explain_candidate(
            candidate,
            scores,
            c["contradiction_score"]
        )

        print(explanation)
        break