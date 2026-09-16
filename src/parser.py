"""
parser.py
Extracts raw text from a resume file, regardless of format (PDF, DOCX, TXT).
This is deliberately dumb: it just gets clean text out. All the "understanding"
of that text happens later in extractor.py and scorer.py.
"""

import os


def extract_text(file_path: str) -> str:
    """Return the plain text content of a resume file.

    Supports .pdf, .docx, and .txt. Raises ValueError for anything else.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return _extract_txt(file_path)
    elif ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext == ".docx":
        return _extract_docx(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}' for {file_path}. "
            "Supported: .pdf, .docx, .txt"
        )


def _extract_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_pdf(file_path: str) -> str:
    import pdfplumber

    text_chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def _extract_docx(file_path: str) -> str:
    import docx

    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def load_resumes_from_dir(dir_path: str) -> dict:
    """Return {filename: raw_text} for every supported resume file in dir_path."""
    supported = {".pdf", ".docx", ".txt"}
    resumes = {}
    for fname in sorted(os.listdir(dir_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in supported:
            full_path = os.path.join(dir_path, fname)
            try:
                resumes[fname] = extract_text(full_path)
            except Exception as e:
                print(f"  [warn] could not parse {fname}: {e}")
    return resumes
