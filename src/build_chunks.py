"""
Day 3-4 — Combine all 3 insurers' chunks into one unified, tagged dataset.
Each chunk becomes a dict: {"text": ..., "insurer": ..., "section_type": ...,
"source_section": ...} - ready for embedding and storage in ChromaDB.
"""

import re
from load_documents import load_and_clean


def split_by_pattern(text, pattern, flags=re.MULTILINE):
    """Generic splitter: find all matches of `pattern`, slice text between
    consecutive matches. Returns a list of (label, chunk_text) tuples."""
    matches = list(re.finditer(pattern, text, flags))
    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        label = match.group().strip()
        chunks.append((label, chunk_text))
    return chunks


def classify_section_type(label):
    """Map a chunk's header label to a broad section_type tag, based on
    keywords commonly found in insurance policy headers."""
    label_lower = label.lower()
    if "preamble" in label_lower:
        return "preamble"
    if "definition" in label_lower:
        return "definitions"
    if "exclu" in label_lower or "excl" in label_lower or "waiting period" in label_lower:
        return "exclusions"
    if "benefit" in label_lower or "section" in label_lower:
        return "benefits"
    if "claim" in label_lower:
        return "claims_process"
    if "term" in label_lower or "condition" in label_lower or "renewal" in label_lower or "cancel" in label_lower:
        return "conditions"
    if "wellness" in label_lower:
        return "wellness"
    return "other"


def chunk_star_health():
    """Chunk Star Health into tagged chunks."""
    path = "data/star_health/SHAHLIP22028V072122_HEALTH2050 (STAR HEALTH).pdf"
    clean_text = load_and_clean(path, "star_health")

    roman_zone_start = clean_text.find("II. DEFINITIONS")
    if roman_zone_start == -1:
        roman_zone_start = len(clean_text)

    section_zone_text = clean_text[:roman_zone_start]
    roman_zone_text = clean_text[roman_zone_start:]

    # Pass 1: Section 1-13
    pattern = r"^Section \d{1,2}[^\n]+"
    matches = list(re.finditer(pattern, section_zone_text, re.MULTILINE))
    real_headers = [
        m for m in matches
        if m.group().count("Section") == 1 and len(m.group()) < 100
    ]
    section_chunks = split_by_pattern(section_zone_text, pattern)
    section_chunks = [
        (label, text) for label, text in section_chunks
        if label.count("Section") == 1 and len(label) < 100 and len(text) > 50
    ]

    # Pass 2: Exclusions (Excl 01 - Excl 38, various formats)
    exclusion_pattern = r"^\d{1,2}\.\s[A-Z0-9][^\n]*Code\s*-?\s*Excl\s*\d{2}"
    roman_chunks = split_by_pattern(roman_zone_text, exclusion_pattern)

    tagged_chunks = []
    for label, text in section_chunks:
        tagged_chunks.append({
            "text": text,
            "insurer": "star_health",
            "section_type": classify_section_type(label),
            "source_section": label,
        })
    for label, text in roman_chunks:
        tagged_chunks.append({
            "text": text,
            "insurer": "star_health",
            "section_type": "exclusions",
            "source_section": label,
        })

    return tagged_chunks


def chunk_hdfc_ergo():
    """Chunk HDFC Ergo into tagged chunks."""
    path = "data/hdfc_ergo/myhealth-medisure-super-top-up-insurance-569138917326(HDFC ERGO).pdf"
    clean_text = load_and_clean(path, "hdfc_ergo")

    body_start = clean_text.find("SECTION A. PREAMBLE")
    body = clean_text[body_start:] if body_start != -1 else clean_text

    section_pattern = r"^SECTION [A-Z]\.\s[^\n]+"
    section_chunks = split_by_pattern(body, section_pattern)

    tagged_chunks = []
    for label, text in section_chunks:
        section_type = classify_section_type(label)

        # Drill into the Exclusions section for finer-grained sub-chunks
        if section_type == "exclusions":
            excl_pattern = r"^\d{1,2}\.\s[A-Za-z][^\n]*Excl\d{2}:?"
            excl_chunks = split_by_pattern(text, excl_pattern)
            if excl_chunks:
                for excl_label, excl_text in excl_chunks:
                    tagged_chunks.append({
                        "text": excl_text,
                        "insurer": "hdfc_ergo",
                        "section_type": "exclusions",
                        "source_section": excl_label,
                    })
                continue  # don't also add the giant parent chunk

        tagged_chunks.append({
            "text": text,
            "insurer": "hdfc_ergo",
            "section_type": section_type,
            "source_section": label,
        })

    return tagged_chunks


def chunk_niva_bupa():
    """Chunk Niva Bupa into tagged chunks."""
    path = "data/niva_bupa/Health-Premia-Policy-Wording(niva health insurance).pdf"
    clean_text = load_and_clean(path, "niva_bupa")

    top_pattern = r"^\d{1,2}\.\s[A-Z][^\n]+"
    top_matches = list(re.finditer(top_pattern, clean_text, re.MULTILINE))
    top_headers = [m for m in top_matches if len(m.group()) < 80]

    top_chunks = []
    for i, match in enumerate(top_headers):
        start = match.start()
        end = top_headers[i + 1].start() if i + 1 < len(top_headers) else len(clean_text)
        chunk_text = clean_text[start:end].strip()
        label = match.group().strip()
        top_chunks.append((label, chunk_text))

    # Keep only the first 6 real sections (Preamble through General Terms)
    top_chunks = top_chunks[:6]

    tagged_chunks = []
    for label, text in top_chunks:
        section_type = classify_section_type(label)

        # Drill into Exclusions (section 5) for sub-section chunks
        if section_type == "exclusions":
            sub_pattern = r"^5\.\d+(\.\d+)?\s[A-Za-z][^\n]+"
            sub_chunks = split_by_pattern(text, sub_pattern)
            if sub_chunks:
                for sub_label, sub_text in sub_chunks:
                    tagged_chunks.append({
                        "text": sub_text,
                        "insurer": "niva_bupa",
                        "section_type": "exclusions",
                        "source_section": sub_label,
                    })
                continue

        tagged_chunks.append({
            "text": text,
            "insurer": "niva_bupa",
            "section_type": section_type,
            "source_section": label,
        })

    return tagged_chunks


def chunk_all_documents():
    """Run all 3 chunkers and combine into one unified list."""
    all_chunks = []
    all_chunks.extend(chunk_star_health())
    all_chunks.extend(chunk_hdfc_ergo())
    all_chunks.extend(chunk_niva_bupa())
    return all_chunks


if __name__ == "__main__":
    all_chunks = chunk_all_documents()

    print(f"\n=== Total chunks across all 3 insurers: {len(all_chunks)} ===\n")

    # Summary by insurer
    for insurer in ["star_health", "hdfc_ergo", "niva_bupa"]:
        count = sum(1 for c in all_chunks if c["insurer"] == insurer)
        print(f"{insurer}: {count} chunks")

    # Summary by section_type
    print()
    section_types = set(c["section_type"] for c in all_chunks)
    for st in sorted(section_types):
        count = sum(1 for c in all_chunks if c["section_type"] == st)
        print(f"{st}: {count} chunks")

    # Show a few sample chunks with full metadata
    print("\n=== Sample tagged chunks ===\n")
    for chunk in all_chunks[:3]:
        print(f"insurer: {chunk['insurer']}")
        print(f"section_type: {chunk['section_type']}")
        print(f"source_section: {chunk['source_section']}")
        print(f"text preview: {chunk['text'][:100]}...")
        print()