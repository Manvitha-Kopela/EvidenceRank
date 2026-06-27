# EvidenceRank — Trust-Aware Candidate Intelligence

## Problem Statement

Recruiters often review hundreds of profiles but still miss the right candidates because traditional Applicant Tracking Systems (ATS) rely heavily on keyword matching and static resume filters.

These systems fail to capture:

* Contextual role fit
* Behavioral signals
* Contradictions between claims and evidence
* Hiring confidence

Two candidates may appear equally qualified on paper but carry very different hiring risks.

---

## Our Solution

**EvidenceRank** is an AI-powered candidate ranking system that evaluates applicants using multiple evidence layers instead of simple keyword matching.

EvidenceRank ranks candidates using:

* Qualification Score
* Behavioral Score
* Contradiction Score
* Hiring Confidence

The system generates a ranked shortlist with explainable reasoning for every candidate.

---

## Core Insight

> Hiring is decision-making under uncertainty.

Traditional ATS asks:

“How similar is this resume to the job description?”

EvidenceRank asks:

“How confident should a recruiter be in hiring this candidate?”

---

## Architecture

Candidate Profiles
↓
Feature Engineering
↓
Qualification + Behavioral + Contradiction Analysis
↓
Fit Score + Confidence + Risk Assessment
↓
Ranked Candidate Shortlist

---

## Feature Engineering

### Qualification Features

* Years of experience
* Skill overlap with job description
* Current title relevance

### Behavioral Features

* GitHub activity
* Recruiter response rate
* Interview completion rate
* Recruiter saves

### Contradiction Features

* Seniority gap
* Skill inflation
* Demand gap
* Availability mismatch

### Confidence Features

* Evidence density
* Data completeness
* Signal consistency

---

## Explainability

For each candidate, EvidenceRank generates:

* Fit Score
* Confidence Score
* Supporting strengths
* Hiring risks
* Human-readable reasoning

Example:

Candidate: CAND_0088025

Fit Score: 80.14
Confidence: 77.76%

Strengths:

* Strong GitHub activity
* High recruiter engagement
* Strong AI skill alignment

Risks:

* Long notice period
* Not willing to relocate

---

## Results

* Processed 100,000 candidate profiles
* Generated ranked shortlist in ~11 seconds
* Produced top 100 ranked candidates
* Submission validated successfully

---

## Repository Structure

```bash
evidencerank/
│
├── data/
│   └── raw/
│
├── outputs/
│   ├── ranked_candidates.csv
│   └── final_submission.csv
│
├── src/
│   ├── parser.py
│   ├── features.py
│   ├── scorer.py
│   ├── explain.py
│   ├── rank_candidates.py
│   ├── demo_candidate.py
│   └── generate_submission.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Tech Stack

* Python
* Pandas
* NumPy
* tqdm

---

## Future Improvements

* Learning-to-rank models (XGBoost / LightGBM)
* Semantic job understanding using LLMs
* Interactive recruiter dashboard
* Risk-aware hiring recommendations
