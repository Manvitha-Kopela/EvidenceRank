def explain_candidate(candidate, scores, contradictions):
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    skills = candidate["skills"]

    strengths = []
    risks = []

    years = profile.get("years_of_experience", 0)

    if years > 5:
        strengths.append(f"{years} years of experience")

    if signals.get("github_activity_score", 0) > 60:
        strengths.append("Strong GitHub activity")

    if signals.get("recruiter_response_rate", 0) > 0.7:
        strengths.append("High recruiter responsiveness")

    if signals.get("saved_by_recruiters_30d", 0) > 10:
        strengths.append("High recruiter demand")

    skill_names = [s["name"] for s in skills[:5]]
    if skill_names:
        strengths.append("Key skills: " + ", ".join(skill_names))

    # Risks
    if signals.get("notice_period_days", 0) > 60:
        risks.append("Long notice period")

    if not signals.get("willing_to_relocate", True):
        risks.append("Not willing to relocate")

    if contradictions > 0:
        risks.append(f"{contradictions} contradiction signals detected")

    return {
        "candidate_id": candidate["candidate_id"],
        "fit_score": scores["fit_score"],
        "confidence": scores["confidence"],
        "strengths": strengths,
        "risks": risks
    }
def generate_reasoning(candidate):
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    skills = candidate["skills"]

    title = profile.get("current_title", "Unknown")
    years = profile.get("years_of_experience", 0)
    response = signals.get("recruiter_response_rate", 0)

    ai_skill_count = 0

    AI_KEYWORDS = [
        "llm", "rag", "nlp", "ml", "machine learning",
        "retrieval", "pinecone", "vector", "embedding"
    ]

    for skill in skills:
        skill_name = skill["name"].lower()

        for keyword in AI_KEYWORDS:
            if keyword in skill_name:
                ai_skill_count += 1
                break

    reasoning = (
        f"{title} with {years:.1f} yrs; "
        f"{ai_skill_count} AI core skills; "
        f"response rate {response:.2f}."
    )

    return reasoning