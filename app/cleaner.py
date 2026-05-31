# app/cleaner.py

import re
from langchain_core.documents import Document
from typing import List


def remove_headers_footers(text: str) -> str:
    text = re.sub(r'Page\s+\d+(\s+of\s+\d+)?', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'-\s*\d+\s*-', '', text)
    text = re.sub(r'©.*?\n', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    return text


def fix_spaced_characters(text: str) -> str:
    text = re.sub(r'(?<=[A-Z])\s(?=[A-Z])', '', text)
    return text


def fix_line_breaks(text: str) -> str:
    lines = text.split('\n')
    cleaned = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            cleaned.append('')
            continue
        if i < len(lines) - 1 and not line[-1] in '.!?:;,':
            cleaned.append(line + ' ')
        else:
            cleaned.append(line + '\n')
    return ''.join(cleaned)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()


def clean_text(text: str) -> str:
    """
    Master cleaning function. Order matters:
    1. Remove headers/footers first
    2. Fix spaced characters
    3. Fix line breaks
    4. Normalize whitespace last
    """
    text = remove_headers_footers(text)
    text = fix_spaced_characters(text)
    text = fix_line_breaks(text)
    text = normalize_whitespace(text)
    return text


def clean_documents(pages: List[Document]) -> List[Document]:
    cleaned = []
    for page in pages:
        cleaned_text = clean_text(page.page_content)
        cleaned.append(Document(
            page_content=cleaned_text,
            metadata=page.metadata
        ))
    print(f"Cleaned {len(cleaned)} pages")
    return cleaned


def filter_short_chunks(chunks: List[Document], min_length: int = 100) -> List[Document]:
    before = len(chunks)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= min_length]
    print(f"Filtered short chunks: {before} -> {len(chunks)}")
    return chunks


def remove_duplicate_chunks(chunks: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for chunk in chunks:
        fingerprint = chunk.page_content[:200].strip()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(chunk)
    print(f"Removed duplicates: {len(chunks)} -> {len(unique)}")
    return unique