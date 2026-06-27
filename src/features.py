from datetime import datetime
JD_SKILL_MAP = {
    "python": ["python"],
    "retrieval": ["retrieval", "bm25", "rag"],
    "ranking": ["ranking", "reranking"],
    "vector_db": ["milvus", "pinecone", "faiss", "vector database"],
    "ml": ["machine learning", "ml", "nlp", "llm", "fine-tuning"]
}
def compute_skill_overlap(skill_names):
    matched = 0

    for category, aliases in JD_SKILL_MAP.items():
        found = False

        for skill in skill_names:
            skill = skill.lower()

            for alias in aliases:
                if alias in skill:
                    found = True
                    break

            if found:
                break

        if found:
            matched += 1

    return matched / len(JD_SKILL_MAP)

def normalize(value, min_val, max_val):
    if value is None:
        return 0
    return max(0, min(1, (value - min_val) / (max_val - min_val)))
def qualification_features(candidate):
    profile = candidate["profile"]
    skills = candidate["skills"]

    years_exp = profile.get("years_of_experience", 0)

    skill_names = [s["name"].lower() for s in skills]

    skill_overlap = compute_skill_overlap(skill_names)

    title = profile.get("current_title", "").lower()

    title_score = 0
    if "engineer" in title:
        title_score = 1
    elif "developer" in title:
        title_score = 0.8

    return {
        "years_experience": years_exp,
        "skill_overlap": skill_overlap,
        "title_score": title_score
    }
def behavioral_features(candidate):
    signals = candidate["redrob_signals"]

    github = signals.get("github_activity_score", 0)
    response = signals.get("recruiter_response_rate", 0)
    completion = signals.get("interview_completion_rate", 0)
    saves = signals.get("saved_by_recruiters_30d", 0)

    return {
        "github_score": normalize(github, 0, 100),
        "response_rate": response,
        "completion_rate": completion,
        "recruiter_saves": normalize(saves, 0, 50)
    }
def contradiction_features(candidate):
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    skills = candidate["skills"]

    contradictions = 0

    title = profile.get("current_title", "").lower()
    years_exp = profile.get("years_of_experience", 0)
    github = signals.get("github_activity_score", 0)
    response = signals.get("recruiter_response_rate", 0)
    open_to_work = signals.get("open_to_work_flag", False)
    assessment_scores = signals.get("skill_assessment_scores", {})

    # Contradiction 1
    if ("ai" in title or "ml" in title) and github < 20:
        contradictions += 1

    # Contradiction 2
    if open_to_work and response < 0.4:
        contradictions += 1

    # Contradiction 3
    if "staff" in title and years_exp < 5:
        contradictions += 1

    # Contradiction 4
    for skill in skills:
        endorsements = skill.get("endorsements", 0)
        skill_name = skill.get("name")

        if skill_name in assessment_scores:
            score = assessment_scores[skill_name]

            if endorsements > 20 and score < 50:
                contradictions += 1
                break

    saves = signals.get("saved_by_recruiters_30d", 0)

    # Contradiction 5
    if saves > 10 and response < 0.5:
        contradictions += 1

    return {
        "contradiction_score": contradictions
    }
def confidence_features(candidate):
    signals = candidate["redrob_signals"]

    github = normalize(signals.get("github_activity_score", 0), 0, 100)
    profile_complete = normalize(
        signals.get("profile_completeness_score", 0), 0, 100
    )
    response = signals.get("recruiter_response_rate", 0)
    completion = signals.get("interview_completion_rate", 0)

    assessment_scores = signals.get("skill_assessment_scores", {})

    assessment_quality = 0
    if len(assessment_scores) > 0:
        assessment_quality = sum(assessment_scores.values()) / (
            len(assessment_scores) * 100
        )

    evidence_density = (
        0.25 * github
        + 0.25 * profile_complete
        + 0.25 * response
        + 0.15 * completion
        + 0.10 * assessment_quality
    )

    return {
        "evidence_density": evidence_density
    }