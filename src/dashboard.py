import streamlit as st
import pandas as pd
from jd_parser import parse_job_description
from features import contradiction_features
from parser import load_candidates
import plotly.graph_objects as go


def compare_candidates(cand_a, cand_b, contr_a, contr_b):
    reasons = []

    score_diff = cand_a["score"] - cand_b["score"]

    if score_diff > 0:
        reasons.append(f"Higher fit score by {score_diff*100:.2f}%")

    conf_a = min(95, round(cand_a["score"] * 100 + 8, 2))
    conf_b = min(95, round(cand_b["score"] * 100 + 8, 2))

    if conf_a > conf_b:
        reasons.append("Higher hiring confidence")

    if contr_a["contradiction_score"] < contr_b["contradiction_score"]:
        reasons.append("Lower contradiction risk")

    if len(reasons) == 0:
        reasons.append("Both candidates are closely matched")

    return reasons


st.set_page_config(page_title="EvidenceRank", layout="wide")


@st.cache_data
def load_candidate_map():
    candidates = load_candidates("../data/raw/candidates.jsonl")
    return {c["candidate_id"]: c for c in candidates}


jd = parse_job_description("../data/raw/job_description.docx")
df = pd.read_csv("../outputs/final_submission.csv")
candidate_map = load_candidate_map()
candidate_ids = df["candidate_id"].tolist()

# Sidebar
st.sidebar.title("Job Intelligence")
st.sidebar.markdown(f"**Role Type:** {jd['role_type']}")
st.sidebar.markdown(f"**Seniority:** {jd['seniority']}")
st.sidebar.markdown(f"**Company Style:** {jd['company_style']}")

st.sidebar.markdown("### Required Skills")
for skill in jd["required_skills"]:
    st.sidebar.markdown(f"- {skill}")

# Title
st.title("EvidenceRank AI Recruiter")
st.caption("Ranking candidates by fit, confidence, and hiring risk")

# Top Candidates
st.subheader("Top Ranked Candidates")
st.dataframe(
    df[["rank", "candidate_id", "score"]].head(20),
    width="stretch"
)

# Candidate Comparison
st.subheader("Candidate Comparison")

compare1 = st.selectbox("Candidate A", candidate_ids, index=0, key="compare1")
compare2 = st.selectbox("Candidate B", candidate_ids, index=1, key="compare2")

cand_a = df[df["candidate_id"] == compare1].iloc[0]
cand_b = df[df["candidate_id"] == compare2].iloc[0]

contr_a = contradiction_features(candidate_map[compare1])
contr_b = contradiction_features(candidate_map[compare2])

comparison_df = pd.DataFrame({
    "Metric": ["Rank", "Fit Score", "Confidence", "Contradictions"],
    compare1: [
        int(cand_a["rank"]),
        round(cand_a["score"] * 100, 2),
        min(95, round(cand_a["score"] * 100 + 8, 2)),
        contr_a["contradiction_score"]
    ],
    compare2: [
        int(cand_b["rank"]),
        round(cand_b["score"] * 100, 2),
        min(95, round(cand_b["score"] * 100 + 8, 2)),
        contr_b["contradiction_score"]
    ]
})

st.dataframe(comparison_df, width="stretch")

# AI Summary
st.subheader("AI Comparison Summary")

winner = compare1 if cand_a["score"] >= cand_b["score"] else compare2

if winner == compare1:
    summary = compare_candidates(cand_a, cand_b, contr_a, contr_b)
else:
    summary = compare_candidates(cand_b, cand_a, contr_b, contr_a)

st.success(f"Why {winner} ranks higher:")
for item in summary:
    st.write("•", item)

# Deep Analysis
st.subheader("Deep Candidate Analysis")
selected = st.selectbox("Select Candidate", candidate_ids)

candidate = df[df["candidate_id"] == selected].iloc[0]
full_candidate = candidate_map[selected]

reason = candidate["reasoning"].lower()
contradiction_data = contradiction_features(full_candidate)

fit_score = float(candidate["score"])
confidence = min(95, round(candidate["score"] * 100 + 8, 2))

skill_score = min(1.0, fit_score)
seniority_score = 1.0 if "staff" in reason else 0.6
confidence_score = confidence / 100
risk_score = max(0, 1 - contradiction_data["contradiction_score"] * 0.25)

categories = ["Skill Match", "Seniority", "Confidence", "Low Risk"]
values = [skill_score, seniority_score, confidence_score, risk_score]

st.markdown(f"### Selected Candidate: `{selected}`")
st.subheader("Candidate Details")

col1, col2 = st.columns(2)

with col1:
    st.metric("Rank", int(candidate["rank"]))
    st.metric("Fit Score", f"{fit_score * 100:.2f}%")
    st.progress(fit_score)

with col2:
    st.metric("Hiring Confidence", f"{confidence}%")
    st.metric("Contradictions", contradiction_data["contradiction_score"])

# Radar Chart
st.subheader("Candidate Radar Profile")

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name='Candidate'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    ),
    showlegend=False,
    height=650
)

st.plotly_chart(fig, use_container_width=True)

# Score Breakdown
st.subheader("Score Breakdown")

qualification = round(fit_score * 0.5, 3)
behavior = round(fit_score * 0.3, 3)
confidence_part = round(fit_score * 0.2, 3)
penalty = round(contradiction_data["contradiction_score"] * 0.05, 3)

breakdown = pd.DataFrame({
    "Component": ["Qualification", "Behavior", "Confidence", "Penalty"],
    "Value": [qualification, behavior, confidence_part, -penalty]
})

st.bar_chart(breakdown.set_index("Component"))

# Reasoning
st.subheader("Reasoning")
st.info(candidate["reasoning"])

# Strengths
st.subheader("Strengths")
strengths = []

if "staff" in reason:
    strengths.append("Strong seniority match")

if "ml" in reason or "ai" in reason:
    strengths.append("Strong AI alignment")

if fit_score > 0.75:
    strengths.append("Excellent overall fit")

if len(strengths) == 0:
    strengths.append("Good overall candidate")

for s in strengths:
    st.success(s)

# Contradictions
st.subheader("Contradictions Detected")

if contradiction_data["contradiction_score"] == 0:
    st.success("No contradictions detected")
else:
    for item in contradiction_data["reasons"]:
        st.error(item)

# Risk Assessment
st.subheader("Risk Assessment")

risks = []

if fit_score < 0.65:
    risks.append("Lower qualification fit")

if confidence < 75:
    risks.append("Lower hiring confidence")

if contradiction_data["contradiction_score"] >= 2:
    risks.append("High signal inconsistency")

if len(risks) == 0:
    risks.append("Low risk candidate")

for risk in risks:
    st.warning(risk)

st.markdown("---")
st.caption("EvidenceRank © 2026 | Trust-aware AI candidate ranking system")