"""
reasoning.py
Turns a candidate's numeric score breakdown into a one/two-sentence, readable
explanation of why they ranked where they did.

By design this works two ways:
  1. If ANTHROPIC_API_KEY is set, we ask Claude to write a natural, specific
     explanation grounded strictly in the extracted facts (matched/missing
     skills, years, education) — the LLM is explaining pre-computed numbers,
     not inventing its own score. This avoids the classic RAG failure mode
     of the model hallucinating a justification.
  2. If no API key is set, we fall back to a template-based explanation using
     the same facts. The agent still runs end-to-end with zero API cost or
     setup — useful for reviewers who just want to see it work immediately.
"""

import os


SYSTEM_PROMPT = """You are a hiring analyst assistant. You will be given a \
candidate's extracted resume facts and their computed score breakdown \
against a job description. Write exactly one or two sentences explaining \
why they scored the way they did. Only reference facts given to you — do \
not invent skills, experience, or qualifications not listed. Be direct and \
specific (name actual matched/missing skills), not generic."""


def _template_reasoning(candidate: dict) -> str:
    matched = candidate["matched_skills"]
    missing = candidate["missing_skills"]
    years = candidate["years_experience"]

    matched_str = ", ".join(matched) if matched else "no listed JD skills"
    parts = [f"Matches on {matched_str}"]
    if years:
        parts.append(f"with {years} years of stated experience")
    if missing:
        top_missing = ", ".join(missing[:3])
        parts.append(f"but is missing {top_missing}")
    return "; ".join(parts) + "."


def generate_reasoning(candidate: dict, jd_text: str, use_llm: bool = True) -> str:
    """Return a short explanation string for one scored candidate."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not use_llm or not api_key:
        return _template_reasoning(candidate)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        user_content = (
            f"Job description (excerpt): {jd_text[:600]}\n\n"
            f"Candidate: {candidate['filename']}\n"
            f"Total score: {candidate['total_score']}/100\n"
            f"TF-IDF similarity: {candidate['similarity_score']}\n"
            f"Matched skills: {candidate['matched_skills']}\n"
            f"Missing skills: {candidate['missing_skills']}\n"
            f"Years experience (stated): {candidate['years_experience']}\n"
            f"Education level: {candidate['education_level']}\n"
        )
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=150,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        # Never let an API hiccup break the ranking run — fall back quietly.
        return _template_reasoning(candidate) + f" [LLM reasoning unavailable: {e}]"
