"""
Day 2 — Chunk Star Health's cleaned text into section-based chunks.
Splits on "Section <number> <Title>" boundaries (Section 1 through Section 13),
plus separately handles the Customer Information Sheet and Definitions/
Exclusions zones, per the Week 1 chunking strategy.
"""

import re
from load_documents import load_and_clean


def split_by_section_markers(text, pattern, label_prefix="Section"):
    """
    Generic splitter: finds all positions where `pattern` matches, then
    slices the text between consecutive matches. Returns a list of
    (chunk_label, chunk_text) tuples.
    """
    matches = list(re.finditer(pattern, text))

    if not matches:
        return []

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        # Use the matched text itself as a label (e.g. "Section 1 Hospitalization")
        label = match.group().strip()
        chunks.append((label, chunk_text))

    return chunks


def chunk_star_health(clean_text):
    """
    Chunk Star Health's full policy body in two passes:
    1. Sections 1-13 (the coverage/benefits zone) - stops at "II. DEFINITIONS"
    2. The Roman-numeral zone (II. DEFINITIONS, III. EXCLUSIONS, IV., V.)
       which uses individual exclusion codes / numbered conditions as
       finer-grained chunk boundaries.
    """
    # Find where the Roman-numeral zone begins - everything before this
    # point is fair game for Section-based chunking, everything after
    # needs different handling.
    roman_zone_start = clean_text.find("II. DEFINITIONS")
    if roman_zone_start == -1:
        roman_zone_start = len(clean_text)  # fallback: no split found

    section_zone_text = clean_text[:roman_zone_start]
    roman_zone_text = clean_text[roman_zone_start:]

    # --- Pass 1: Section 1 - Section 13 ---
    pattern = r"^Section \d{1,2}[^\n]+"
    matches = list(re.finditer(pattern, section_zone_text, re.MULTILINE))

    # Filter out cross-reference noise: real headers mention "Section" once,
    # are reasonably short on their own header line, AND have real content
    # following them (cross-reference stubs are short overall).
    real_headers = [
        m for m in matches
        if m.group().count("Section") == 1 and len(m.group()) < 100
    ]

    section_chunks = []
    for i, match in enumerate(real_headers):
        start = match.start()
        end = real_headers[i + 1].start() if i + 1 < len(real_headers) else len(section_zone_text)
        chunk_text = section_zone_text[start:end].strip()
        label = match.group().strip()
        section_chunks.append((label, chunk_text))

    # Drop tiny noise chunks (real section content is never this short -
    # these are leftover cross-reference fragments that slipped through)
    section_chunks = [
        (label, text) for label, text in section_chunks if len(text) > 50
    ]

    # --- Pass 2: Definitions / Exclusions / Conditions (Roman-numeral zone) ---
    # Exclusion codes follow the pattern "<number>. <Title> - Code Excl NN"
    # but with real-world inconsistencies found in this document:
    #   - title can start with a digit (e.g. "30-day waiting period") not
    #     just a capital letter, so we allow [A-Z0-9] as the first character
    #   - the dash before "Code" is sometimes missing a space ("Code- Excl 04")
    #     and sometimes the dash itself is missing entirely ("Code Excl 08")
    #   - "Excl" and the number sometimes have a space, sometimes don't
    exclusion_pattern = r"^\d{1,2}\.\s[A-Z0-9][^\n]*Code\s*-?\s*Excl\s*\d{2}"
    exclusion_matches = list(re.finditer(exclusion_pattern, roman_zone_text, re.MULTILINE))

    roman_chunks = []
    if exclusion_matches:
        for i, match in enumerate(exclusion_matches):
            start = match.start()
            end = exclusion_matches[i + 1].start() if i + 1 < len(exclusion_matches) else len(roman_zone_text)
            chunk_text = roman_zone_text[start:end].strip()
            label = match.group().strip()
            roman_chunks.append((label, chunk_text))

    return section_chunks, roman_chunks


if __name__ == "__main__":
    # Load and clean Star Health using yesterday's function
    star_health_path = "data/star_health/SHAHLIP22028V072122_HEALTH2050 (STAR HEALTH).pdf"
    clean_text = load_and_clean(star_health_path, "star_health")

    print("\n\n=== Chunking Star Health by Section ===")
    section_chunks, roman_chunks = chunk_star_health(clean_text)
    print(f"Found {len(section_chunks)} section chunks (Sections 1-13)\n")

    for label, chunk_text in section_chunks:
        print(f"--- {label} ---")
        print(f"Length: {len(chunk_text)} characters")
        print(f"Preview: {chunk_text[:100]}...")
        print()

    print(f"\n=== Found {len(roman_chunks)} exclusion/definition chunks ===\n")
    for label, chunk_text in roman_chunks:
        print(f"--- {label} ---")
        print(f"Length: {len(chunk_text)} characters")
        print(f"Preview: {chunk_text[:100]}...")
        print()

    # --- Now load and inspect HDFC Ergo to find its real header format ---
    print("\n\n=== Chunking HDFC Ergo by SECTION letter ===")
    hdfc_path = "data/hdfc_ergo/myhealth-medisure-super-top-up-insurance-569138917326(HDFC ERGO).pdf"
    hdfc_clean_text = load_and_clean(hdfc_path, "hdfc_ergo")

    # Skip the Table of Contents (first ~2300 characters) by starting the
    # search from the real "SECTION A. PREAMBLE" onwards, not the ToC's
    # mention of section titles.
    body_start = hdfc_clean_text.find("SECTION A. PREAMBLE")
    hdfc_body = hdfc_clean_text[body_start:] if body_start != -1 else hdfc_clean_text

    # Real top-level headers look like "SECTION A. PREAMBLE", "SECTION B.
    # DEFINITIONS", "SECTION D. WAITING PERIOD & EXCLUSIONS" etc.
    hdfc_section_pattern = r"^SECTION [A-Z]\.\s[^\n]+"
    hdfc_matches = list(re.finditer(hdfc_section_pattern, hdfc_body, re.MULTILINE))

    hdfc_chunks = []
    for i, match in enumerate(hdfc_matches):
        start = match.start()
        end = hdfc_matches[i + 1].start() if i + 1 < len(hdfc_matches) else len(hdfc_body)
        chunk_text = hdfc_body[start:end].strip()
        label = match.group().strip()
        hdfc_chunks.append((label, chunk_text))

    print(f"Found {len(hdfc_chunks)} top-level SECTION chunks\n")
    for label, chunk_text in hdfc_chunks:
        print(f"--- {label} ---")
        print(f"Length: {len(chunk_text)} characters")
        print(f"Preview: {chunk_text[:100]}...")
        print()

    # The Exclusions section (SECTION D) is itself long and contains many
    # numbered exclusion codes - find that specific chunk and split it
    # further by its "Excl01:", "Excl02:" markers.
    exclusions_chunk_text = next(
        (text for label, text in hdfc_chunks if "WAITING PERIOD" in label), None
    )
    if exclusions_chunk_text:
        excl_pattern = r"^\d{1,2}\.\s[A-Za-z][^\n]*Excl\d{2}:?"
        excl_matches = list(re.finditer(excl_pattern, exclusions_chunk_text, re.MULTILINE))
        print(f"\n=== Found {len(excl_matches)} individual exclusion codes within SECTION D ===")
        for m in excl_matches:
            print(f"  {m.group().strip()[:80]}")

    # --- Now inspect Niva Bupa's structure ---
    print("\n\n=== Inspecting Niva Bupa structure ===")
    niva_path = "data/niva_bupa/Health-Premia-Policy-Wording(niva health insurance).pdf"
    niva_clean_text = load_and_clean(niva_path, "niva_bupa")

    # --- Now build real Niva Bupa chunking ---
    print("\n\n=== Chunking Niva Bupa by top-level numbered section ===")
    niva_path = "data/niva_bupa/Health-Premia-Policy-Wording(niva health insurance).pdf"
    niva_clean_text = load_and_clean(niva_path, "niva_bupa")

    # Top-level sections: "1. Preamble", "2. Definitions & Interpretation", etc.
    # Anchor to line-start so we don't catch sub-section numbers like "5.1"
    # (a real top-level header is "<1-2 digits>." with NO additional dot
    # right after the number, e.g. "5." not "5.1.")
    niva_top_pattern = r"^\d{1,2}\.\s[A-Z][^\n]+"
    niva_top_matches = list(re.finditer(niva_top_pattern, niva_clean_text, re.MULTILINE))

    # Filter: real top-level headers are short title lines, not long body
    # sentences that happen to start a line with a number due to PDF wrapping
    niva_top_headers = [
        m for m in niva_top_matches if len(m.group()) < 80
    ]

    niva_chunks = []
    for i, match in enumerate(niva_top_headers):
        start = match.start()
        end = niva_top_headers[i + 1].start() if i + 1 < len(niva_top_headers) else len(niva_clean_text)
        chunk_text = niva_clean_text[start:end].strip()
        label = match.group().strip()
        niva_chunks.append((label, chunk_text))

    # Keep only the first 6 real top-level sections (Preamble through
    # General Terms and Clauses). After section 6, the document contains
    # bundled digital-health-service terms (Doctor Policy, User Content,
    # Privacy, etc.) which restart numbering from 1 - these aren't part
    # of the core policy terms and aren't relevant to typical user questions.
    niva_chunks = niva_chunks[:6]

    print(f"Found {len(niva_chunks)} top-level chunks\n")
    for label, chunk_text in niva_chunks:
        print(f"--- {label} ---")
        print(f"Length: {len(chunk_text)} characters")
        print(f"Preview: {chunk_text[:100]}...")
        print()

    # Drill into the Exclusions chunk specifically, splitting by its
    # dotted sub-section numbers (5.1, 5.1.1, 5.2, etc.)
    niva_exclusions_text = next(
        (text for label, text in niva_chunks if "Exclusions" in label), None
    )
    if niva_exclusions_text:
        sub_pattern = r"^5\.\d+(\.\d+)?\s[A-Za-z][^\n]+"
        sub_matches = list(re.finditer(sub_pattern, niva_exclusions_text, re.MULTILINE))
        print(f"\n=== Found {len(sub_matches)} exclusion sub-sections within section 5 ===")
        for m in sub_matches[:20]:  # show first 20
            print(f"  {m.group().strip()[:80]}")