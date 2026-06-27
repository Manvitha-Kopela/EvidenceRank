import pandas as pd
from parser import load_candidates
from features import *
from scorer import calculate_scores

print("Loading candidates...")
candidates = load_candidates("../data/raw/candidates.jsonl")

results = []

print("Scoring candidates...")

for candidate in candidates:
    try:
        q = qualification_features(candidate)
        b = behavioral_features(candidate)
        c = contradiction_features(candidate)
        conf = confidence_features(candidate)

        scores = calculate_scores(q, b, c, conf)

        results.append({
            "candidate_id": candidate["candidate_id"],
            "fit_score": scores["fit_score"],
            "confidence": scores["confidence"],
            "contradictions": c["contradiction_score"]
        })

    except Exception as e:
        print(f"Error processing candidate {candidate['candidate_id']}: {e}")

df = pd.DataFrame(results)
print("\nContradiction Distribution:")
print(df["contradictions"].value_counts().sort_index())

df = df.sort_values("fit_score", ascending=False)

df.to_csv("../outputs/ranked_candidates.csv", index=False)

print(df.head(10))