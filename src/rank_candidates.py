import pandas as pd
from parser import load_candidates
from features import (
    qualification_features, behavioral_features,
    contradiction_features, confidence_features,
    precompute_embeddings, load_embedding_cache
)
from scorer import calculate_scores, calculate_risk

print("Loading candidates...")
candidates = load_candidates("../data/raw/candidates.jsonl")

# Precompute all embeddings once, or load from cache
load_embedding_cache()
if not __import__('os').path.exists("../outputs/candidate_embeddings.npy"):
    precompute_embeddings(candidates)

results = []
print("Scoring candidates...")

for candidate in candidates:
    try:
        q = qualification_features(candidate)
        b = behavioral_features(candidate)
        c = contradiction_features(candidate)
        conf = confidence_features(candidate)
        scores = calculate_scores(q, b, c, conf)
        risk = calculate_risk(candidate)

        results.append({
            "candidate_id": candidate["candidate_id"],
            "fit_score": round(scores["fit_score"], 4),
            "confidence": round(scores["confidence"], 4),
            "risk_score": round(risk, 4),
            "contradictions": c["contradiction_score"],
            "contradiction_reasons": "; ".join(c["reasons"])
        })
    except Exception as e:
        print(f"Error processing {candidate['candidate_id']}: {e}")

df = pd.DataFrame(results)
df = df.sort_values("fit_score", ascending=False).reset_index(drop=True)
df.insert(0, "rank", df.index + 1)

df.to_csv("../outputs/ranked_candidates.csv", index=False)

print("\nTop 10 candidates:")
print(df[["rank", "candidate_id", "fit_score", "confidence", "risk_score", "contradictions"]].head(10))
top_ids = df.head(5)["candidate_id"].tolist()

print("\n===== Manual Audit =====")

for candidate in candidates:
    if candidate["candidate_id"] in top_ids:
        profile = candidate["profile"]

        print("\n----------------------")
        print("Candidate:", candidate["candidate_id"])
        print("Title:", profile.get("current_title", "N/A"))
        print("Experience:", profile.get("years_of_experience", 0))

        skills = [s["name"] for s in candidate.get("skills", [])[:15]]
        print("Skills:", skills)

        summary = profile.get("summary", "")
        print("Summary:", summary[:300])
print(f"\nTotal scored: {len(df)}")
print("\nContradiction distribution:")
print(df["contradictions"].value_counts().sort_index())

print("\nRisk score distribution (top 100):")
print(df.head(100)["risk_score"].describe())