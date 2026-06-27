from docx import Document
import re


def extract_text(docx_path):
    doc = Document(docx_path)

    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())

    return "\n".join(text)


def parse_job_description(docx_path):
    text = extract_text(docx_path)
    lower_text = text.lower()

    skill_keywords = [
        "python",
        "machine learning",
        "deep learning",
        "nlp",
        "rag",
        "llm",
        "embeddings",
        "vector database",
        "retrieval",
        "ranking",
        "sql",
        "pytorch",
        "tensorflow",
        "aws",
        "gcp",
        "docker",
        "kubernetes"
    ]

    found_skills = []

    for skill in skill_keywords:
        if skill in lower_text:
            found_skills.append(skill)

    seniority = "mid"

    if "staff" in lower_text or "principal" in lower_text:
        seniority = "staff"
    elif "senior" in lower_text:
        seniority = "senior"
    elif "intern" in lower_text:
        seniority = "intern"

    role_type = "general"

    if "retrieval" in lower_text or "search" in lower_text:
        role_type = "retrieval"
    elif "recommendation" in lower_text:
        role_type = "recommendation"
    elif "ml engineer" in lower_text:
        role_type = "ml"

    company_style = "enterprise"

    if "startup" in lower_text:
        company_style = "startup"

    return {
        "required_skills": found_skills,
        "seniority": seniority,
        "role_type": role_type,
        "company_style": company_style
    }


if __name__ == "__main__":
    jd = parse_job_description("../data/raw/job_description.docx")
    print(jd)