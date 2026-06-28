import pandas as pd
import json
from parser import load_candidates
from features import (
    qualification_features, behavioral_features,
    contradiction_features, confidence_features,
    load_embedding_cache, detect_honeypot
)
from scorer import calculate_scores, calculate_risk

def generate_reasoning(candidate, q, b, c):
    profile = candidate["profile"]
    title = profile.get("current_title", "")
    years = profile.get("years_of_experience", 0)
    skills = candidate.get("skills", [])
    top_skills = [s["name"] for s in skills if s.get("proficiency") == "advanced"][:3]
    response = candidate["redrob_signals"].get("recruiter_response_rate", 0)
    github = candidate["redrob_signals"].get("github_activity_score", 0)

    skill_str = ", ".join(top_skills) if top_skills else "technical skills"
    reasoning = (
        f"{title} with {years:.1f} yrs; "
        f"advanced in {skill_str}; "
        f"response rate {response:.2f}, GitHub {github:.0f}."
    )

    if c["contradiction_score"] > 0:
        reasoning += f" Concern: {c['reasons'][0].lower()}."

    return reasoning

print("Loading candidates...")
candidates = load_candidates("../data/raw/candidates.jsonl")

print("Loading embedding cache...")
load_embedding_cache()

results = []
honeypot_count = 0

print("Scoring candidates...")
for candidate in candidates:
    try:
        if detect_honeypot(candidate):
            honeypot_count += 1
            continue

        q = qualification_features(candidate)
        b = behavioral_features(candidate)
        c = contradiction_features(candidate)
        conf = confidence_features(candidate)
        scores = calculate_scores(q, b, c, conf)

        results.append({
            "candidate_id": candidate["candidate_id"],
            "fit_score": scores["fit_score"],
            "reasoning_data": (candidate, q, b, c)
        })
    except Exception as e:
        print(f"Error {candidate['candidate_id']}: {e}")

print(f"Honeypots filtered: {honeypot_count}")

# Sort and take top 100
results.sort(key=lambda x: (-x["fit_score"], x["candidate_id"]))
top100 = results[:100]

rows = []
for rank, item in enumerate(top100, 1):
    candidate, q, b, c = item["reasoning_data"]
    rows.append({
        "candidate_id": candidate["candidate_id"],
        "rank": rank,
        "score": round(item["fit_score"], 4),
        "reasoning": generate_reasoning(candidate, q, b, c)
    })

df = pd.DataFrame(rows)
df.to_csv("../outputs/final_submission.csv", index=False)

print("\nTop 10:")
print(df.head(10))
print(f"\nTotal rows: {len(df)}")
print("Saved to ../outputs/final_submission.csv")