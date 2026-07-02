import sys
import os
import pandas as pd
from parser import load_candidates
from features import (
    initialize_features,
    qualification_features,
    behavioral_features,
    contradiction_features,
    confidence_features,
    precompute_embeddings,
    load_embedding_cache,
    compute_semantic_score
)
from scorer import calculate_scores, calculate_risk

if len(sys.argv) < 2:
    raise ValueError(
        "Usage: python rank_candidates.py <job_description.docx>"
    )

JD_PATH = sys.argv[1]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATE_PATH = os.path.join(BASE_DIR, "data", "raw", "candidates.jsonl")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "outputs", "candidate_embeddings.npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "ranked_candidates.csv")

print("Initializing features for JD:", JD_PATH)
initialize_features(JD_PATH)

print("Loading candidates...")
candidates = load_candidates(CANDIDATE_PATH)

load_embedding_cache()

if not os.path.exists(EMBEDDINGS_PATH):
    precompute_embeddings(candidates)

print("Prefiltering top 10k candidates...")
prefilter = []
for candidate in candidates:
    score = compute_semantic_score(candidate)
    prefilter.append((score, candidate))

prefilter.sort(key=lambda x: x[0], reverse=True)
candidates = [candidate for _, candidate in prefilter[:10000]]

results = []

print("Scoring candidates...")
total = len(candidates)
for i, candidate in enumerate(candidates, start=1):
    if i % 1000 == 0:
        print(f"Processed {i}/{total}")

    try:
        q = qualification_features(candidate)
        b = behavioral_features(candidate)
        c = contradiction_features(candidate)
        conf = confidence_features(candidate)

        scores = calculate_scores(q, b, c, conf)
        risk = calculate_risk(candidate)

        reasoning_parts = []

        if q["semantic_score"] >= 0.55:
            reasoning_parts.append("Strong semantic alignment with job description")
        elif q["semantic_score"] >= 0.45:
            reasoning_parts.append("Good semantic relevance to role")

        if q["skill_overlap"] >= 0.60:
            reasoning_parts.append("High required skill coverage")
        elif q["skill_overlap"] >= 0.30:
            reasoning_parts.append("Partial required skill coverage")

        if b["behavior_score"] >= 0.70:
            reasoning_parts.append("Strong recruiter engagement signals")

        if c["contradiction_score"] == 0:
            reasoning_parts.append("No profile contradictions detected")
        else:
            reasoning_parts.append(
                f'{c["contradiction_score"]} contradiction signals detected'
            )

        if risk <= 0.35:
            reasoning_parts.append("Low hiring risk")
        elif risk >= 0.65:
            reasoning_parts.append("Elevated hiring risk")

        reasoning_text = (
            "; ".join(reasoning_parts)
            if reasoning_parts
            else "Balanced candidate profile with moderate fit"
        )


        results.append({
            "candidate_id": candidate["candidate_id"],
            "fit_score": round(scores["fit_score"], 4),
            "confidence": round(scores["confidence"], 4),
            "risk_score": round(risk, 4),
            "contradictions": c["contradiction_score"],
            "contradiction_reasons": "; ".join(c["reasons"]),
            "reasoning": reasoning_text
        })
    except Exception as e:
        print(candidate["candidate_id"], e)

df = pd.DataFrame(results)
df = df.sort_values("fit_score", ascending=False).reset_index(drop=True)
df.insert(0, "rank", df.index + 1)

print(df.head(10))

os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(df)} ranked candidates to {OUTPUT_PATH}")

top_100_path = os.path.join(OUTPUT_DIR, "top_100_candidates.csv")
df.head(100).to_csv(top_100_path, index=False)
print(f"Saved top 100 shortlist to {top_100_path}")