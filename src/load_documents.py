"""
Day 1 — Load and clean all 3 policy PDFs.
Loads each PDF page by page, detects lines that repeat across most pages
(these are headers/footers like company name, address, IRDAI Reg No.),
strips them out, and outputs one clean text blob per insurer.
"""

import re
from pypdf import PdfReader
from collections import Counter


def load_pdf_pages(filepath):
    """Load a PDF and return a list of per-page text strings."""
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        pages.append(text)
    return pages


def find_repeated_lines(pages, min_repeat_ratio=0.5):
    """
    Find lines that appear on at least `min_repeat_ratio` fraction of pages.
    These are very likely headers/footers, not real content.
    """
    line_page_count = Counter()

    for page_text in pages:
        # Get unique lines on this page (avoid double-counting if a line
        # appears twice on the same page)
        lines_on_this_page = set(
            line.strip() for line in page_text.split("\n") if line.strip()
        )
        for line in lines_on_this_page:
            line_page_count[line] += 1

    num_pages = len(pages)
    threshold = num_pages * min_repeat_ratio

    repeated_lines = {
        line for line, count in line_page_count.items() if count >= threshold
    }
    return repeated_lines


def clean_pages(pages, repeated_lines):
    """Remove repeated header/footer lines from every page, then join."""
    cleaned_pages = []
    for page_text in pages:
        lines = page_text.split("\n")
        kept_lines = [
            line for line in lines if line.strip() not in repeated_lines
        ]
        cleaned_pages.append("\n".join(kept_lines))
    return "\n".join(cleaned_pages)


# Some insurers (e.g. Star Health) squash the page number and policy name
# into the same line as the header, so the line never repeats exactly even
# though the boilerplate phrase inside it does. We strip these phrases
# wherever they appear, mid-line or not.
BOILERPLATE_PHRASES = [
    "STAR HEALTH AND ALLIED INSURANCE COMPANY LIMITED",
    "Regd. & Corporate Office: 1, New Tank Street, Valluvar Kottam High Road, Nungambakkam,",
    "Chennai - 600 034.",
    "Phone : 044 - 28288800",
    "Email : support@starhealth.in",
    "Website : www.starhealth.in",
    "CIN : U66010TN2005PLC056649",
    "IRDAI Regn. No. : 129",
    "Star Comprehensive Insurance Policy",
    "POL / COMP / V.13 / 2021",
]


def strip_boilerplate_phrases(text, phrases):
    """Remove known boilerplate phrases wherever they appear in the text,
    even mid-line (for PDFs that squash header text into content lines)."""
    for phrase in phrases:
        text = text.replace(phrase, "")
    return text


# Patterns that change per page (page numbers, unique ID codes) so they
# can't be matched as fixed phrases - regex handles the varying part.
BOILERPLATE_PATTERNS = [
    r"\d+ of \d+",                          # "1 of 16", "2 of 16", etc.
    r"Unique Identi[ﬁf]cation No\.: \S+",   # the policy's unique ID code
    r"Regd\.\s*&\s*Corporate Of[ﬁf]ce:[\s\S]*?Nungambakkam,?",  # Star Health address block (handles ligature + line break)
]


def strip_boilerplate_patterns(text, patterns):
    """Remove regex-based boilerplate (things that vary slightly per page,
    like page numbers, but follow a consistent pattern)."""
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text


def fix_ligatures(text):
    """PDF extraction sometimes keeps the 'fi' and 'fl' ligature characters
    as single glyphs instead of two letters. Replace them with normal text."""
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return text


def cleanup_stray_punctuation(text):
    """After removing phrases, leftover separator symbols (like the « used
    between fields) and extra blank lines can remain. Tidy these up."""
    # Remove stray « bullet/separator characters
    text = text.replace("«", "")
    # Collapse 3+ blank lines into a single blank line
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    # Collapse repeated spaces left behind after phrase removal
    text = re.sub(r" {2,}", " ", text)
    return text


def load_and_clean(filepath, label):
    """Full pipeline for one PDF: load, detect repeats, strip, return clean text."""
    print(f"\n--- Processing {label} ---")
    pages = load_pdf_pages(filepath)
    print(f"Loaded {len(pages)} pages")

    repeated_lines = find_repeated_lines(pages)
    print(f"Detected {len(repeated_lines)} repeated header/footer lines:")
    for line in list(repeated_lines)[:5]:  # show a sample
        print(f"  - {line[:80]}")

    clean_text = clean_pages(pages, repeated_lines)

    # Second pass: strip known boilerplate phrases that may be squashed
    # into content lines rather than sitting on their own (Star Health style)
    if label == "star_health":
        clean_text = strip_boilerplate_phrases(clean_text, BOILERPLATE_PHRASES)

    # Third pass: strip patterns that vary per page (page numbers, IDs)
    clean_text = strip_boilerplate_patterns(clean_text, BOILERPLATE_PATTERNS)

    # Fourth pass: fix ligature characters and tidy up stray punctuation
    clean_text = fix_ligatures(clean_text)
    clean_text = cleanup_stray_punctuation(clean_text)

    print(f"Clean text length: {len(clean_text)} characters")
    return clean_text


if __name__ == "__main__":
    documents = {
        "star_health": "data/star_health/SHAHLIP22028V072122_HEALTH2050 (STAR HEALTH).pdf",
        "hdfc_ergo": "data/hdfc_ergo/myhealth-medisure-super-top-up-insurance-569138917326(HDFC ERGO).pdf",
        "niva_bupa": "data/niva_bupa/Health-Premia-Policy-Wording(niva health insurance).pdf",
    }

    # DEBUG: inspect the raw repr of the address line before any cleaning,
    # to see exactly what characters pypdf actually extracted
    reader = PdfReader(documents["star_health"])
    page1_text = reader.pages[0].extract_text()
    idx = page1_text.find("Regd")
    if idx != -1:
        snippet = page1_text[idx:idx + 120]
        print("=== RAW REPR of address line (first 120 chars) ===")
        print(repr(snippet))
    else:
        print("Could not find 'Regd' in page 1 text at all.")

    clean_texts = {}
    for label, filepath in documents.items():
        clean_texts[label] = load_and_clean(filepath, label)

    print("\n\n=== Sample of cleaned Star Health text (first 500 chars) ===")
    print(clean_texts["star_health"][:500])

    # DEBUG: find where "Section 1" and "Section 2" actually appear in the
    # cleaned text, so we can see the exact format to split on
    print("\n\n=== Locating 'Section 1' and 'Section 2' headers ===")
    text = clean_texts["star_health"]
    idx1 = text.find("Section 1 ")
    idx2 = text.find("Section 2 ")
    if idx1 != -1:
        print(f"Found 'Section 1' at position {idx1}:")
        print(repr(text[idx1:idx1 + 60]))
    if idx2 != -1:
        print(f"Found 'Section 2' at position {idx2}:")
        print(repr(text[idx2:idx2 + 60]))