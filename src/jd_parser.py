from docx import Document
from parser.spacy_parser import extract_skills_spacy, extraction_confidence


def extract_text(docx_path):
    doc = Document(docx_path)

    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())

    return "\n".join(text)


def extract_job_title(text):
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if 3 < len(line) < 100:
            return line

    return "Unknown Role"


def extract_seniority(text):
    lower = text.lower()

    if "principal" in lower:
        return "principal"
    elif "staff" in lower:
        return "staff"
    elif "senior" in lower:
        return "senior"
    elif "junior" in lower:
        return "junior"
    elif "intern" in lower:
        return "intern"

    return "mid"


def parse_job_description(docx_path):
    text = extract_text(docx_path)
    lower_text = text.lower()

    job_title = extract_job_title(text)
    seniority = extract_seniority(text)

    # -------- spaCy skill extraction --------
    skills = extract_skills_spacy(text)
    confidence = extraction_confidence(skills)

    if confidence < 0.5:
        print("Low confidence parser -> LLM fallback needed")

    # -------- role detection --------
    role_type = "general"
    role_family = "general"

    if any(word in lower_text for word in [
        "machine learning engineer",
        "ml engineer",
        "ai engineer",
        "data scientist"
    ]):
        role_type = "ml"
        role_family = "data_ai"

    elif any(word in lower_text for word in [
        "backend engineer",
        "backend developer",
        "software engineer",
        "platform engineer",
        "devops"
    ]):
        role_type = "backend"
        role_family = "engineering"

    elif any(word in lower_text for word in [
        "frontend engineer",
        "frontend developer",
        "ui engineer"
    ]):
        role_type = "frontend"
        role_family = "engineering"

    elif any(word in lower_text for word in [
        "marketing manager",
        "seo",
        "digital marketing"
    ]):
        role_type = "marketing"
        role_family = "marketing"

    elif any(word in lower_text for word in [
        "sales manager",
        "account executive",
        "business development"
    ]):
        role_type = "sales"
        role_family = "sales"

    elif "retrieval" in lower_text or "search" in lower_text:
        role_type = "retrieval"
        role_family = "data_ai"

    elif "recommendation" in lower_text:
        role_type = "recommendation"
        role_family = "data_ai"

    # -------- company style --------
    company_style = "enterprise"

    if "startup" in lower_text:
        company_style = "startup"

    return {
        "job_title": job_title,
        "required_skills": skills,
        "seniority": seniority,
        "role_type": role_type,
        "role_family": role_family,
        "company_style": company_style,
        "parser_confidence": confidence
    }


if __name__ == "__main__":
    jd = parse_job_description("../data/raw/job_description.docx")
    print(jd)