def calculate_scores(q, b, c, conf):
    # Qualification score
    qualification_score = (
        0.4 * min(q["years_experience"] / 10, 1)
        + 0.4 * q["skill_overlap"]
        + 0.2 * q["title_score"]
    )

    # Behavior score
    behavior_score = (
        0.35 * b["github_score"]
        + 0.25 * b["response_rate"]
        + 0.20 * b["completion_rate"]
        + 0.20 * b["recruiter_saves"]
    )

    contradiction_penalty = c["contradiction_score"] * 0.2

    fit_score = (
        0.45 * qualification_score
        + 0.35 * behavior_score
        + 0.20 * conf["evidence_density"]
    ) - contradiction_penalty

    fit_score = max(0, min(1, fit_score))

    hiring_confidence = max(
        0,
        conf["evidence_density"] - contradiction_penalty
    )

    return {
        "fit_score": fit_score,
        "confidence": hiring_confidence
    }
def calculate_risk(candidate):
    signals = candidate["redrob_signals"]

    notice = signals.get("notice_period_days", 0)
    response = signals.get("recruiter_response_rate", 0)
    relocate = signals.get("willing_to_relocate", True)

    notice_risk = min(notice / 90, 1)
    response_risk = 1 - response
    relocation_risk = 0 if relocate else 0.5

    risk_score = (
        0.5 * notice_risk +
        0.3 * response_risk +
        0.2 * relocation_risk
    )

    return min(risk_score, 1)