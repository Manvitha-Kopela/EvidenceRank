from config import CONFIG


def calculate_scores(q, b, c, conf):
    qualification_score = (
        CONFIG["qualification_weights"]["experience"] * min(q["years_experience"] / 10, 1)
        + CONFIG["qualification_weights"]["skill_overlap"] * q["skill_overlap"]
        + CONFIG["qualification_weights"]["title_score"] * q["title_score"]
        + CONFIG["qualification_weights"]["semantic_score"] * q["semantic_score"]


    )

    behavior_score = (
        CONFIG["behavior_weights"]["github_score"] * b["github_score"]
        + CONFIG["behavior_weights"]["response_rate"] * b["response_rate"]
        + CONFIG["behavior_weights"]["completion_rate"] * b["completion_rate"]
        + CONFIG["behavior_weights"]["recruiter_saves"] * b["recruiter_saves"]
    )

    # Scaled penalty: 0 contradictions = 0, 5 contradictions = 0.25
    contradiction_penalty = (c["contradiction_score"] / 5) * 0.25

    fit_score = (
        CONFIG["fit_weights"]["qualification"] * qualification_score
        + CONFIG["fit_weights"]["behavior"] * behavior_score
        + CONFIG["fit_weights"]["confidence"] * conf["evidence_density"]
    ) - contradiction_penalty

    fit_score = max(0, min(1, fit_score))

    hiring_confidence = max(0, conf["evidence_density"] - contradiction_penalty)

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
        CONFIG["risk_weights"]["notice_period"] * notice_risk
        + CONFIG["risk_weights"]["response_rate"] * response_risk
        + CONFIG["risk_weights"]["relocation"] * relocation_risk
    )
    return min(risk_score, 1)