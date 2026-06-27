import pandas as pd
from parser import load_candidates
from features import *
from scorer import calculate_scores
from explain import generate_reasoning

print("Loading candidates...")
candidates = load_candidates("../data/raw/candidates.jsonl")

results = []

print("Ranking candidates...")

for candidate in candidates:
    try:
        q = qualification_features(candidate)
        b = behavioral_features(candidate)
        c = contradiction_features(candidate)
        conf = confidence_features(candidate)

        scores = calculate_scores(q, b, c, conf)

        reasoning = generate_reasoning(candidate)
        candidate_num = int(candidate["candidate_id"].replace("CAND_", ""))
        adjusted_score = scores["fit_score"] - candidate_num * 1e-12

        results.append({
            "candidate_id": candidate["candidate_id"],
            "score": adjusted_score,   # no round here
            "reasoning": reasoning
        })

    except Exception as e:
        print("Error:", e)

df = pd.DataFrame(results)

df["candidate_num"] = df["candidate_id"].str.replace("CAND_", "").astype(int)

df = pd.DataFrame(results)

# Round score first
df["score"] = df["score"].round(4)

# Sort by score DESC, then candidate_id ASC (tie breaker)
df = df.sort_values(
    by=["score", "candidate_id"],
    ascending=[False, True]
).reset_index(drop=True)

# Keep only top 100 candidates
df = df.head(100).copy()

# Assign ranks 1–100
df["rank"] = range(1, len(df) + 1)

# Final submission columns
submission = df[["candidate_id", "rank", "score", "reasoning"]]

# Save CSV
submission.to_csv("../outputs/final_submission.csv", index=False)

print(submission.head(10))
print("\nSaved successfully!")