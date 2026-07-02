from weight_generator import generate_score_weights
from features import get_jd_info


def calculate_scores(q, b, c, conf):
    weights = generate_score_weights(get_jd_info())

    qualification_score = (
        0.25 * min(q["years_experience"] / 10, 1)
        + 0.20 * q["skill_overlap"]
        + 0.20 * q["title_score"]
        + 0.35 * q["semantic_score"]
    )

    behavior_score = b["behavior_score"]

    contradiction_penalty = (c["contradiction_score"] / 5) * 0.25

    fit_score = (
        weights["qualification"] * qualification_score
        + weights["behavior"] * behavior_score
        + weights["confidence"] * conf["evidence_density"]
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

    notice_risk = min(notice / 180, 1)
    response_risk = 1 - response
    relocation_risk = 0 if relocate else 0.4

    risk_score = (
        0.35 * notice_risk
        + 0.40 * response_risk
        + 0.25 * relocation_risk
    )

    return min(risk_score, 1)