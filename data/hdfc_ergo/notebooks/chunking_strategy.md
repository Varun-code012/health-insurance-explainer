# Chunking Strategy — Health Insurance Policy Explainer

## 1. HDFC Ergo (my:health Medisure Super Top Up)
Confirmed structure (from Table of Contents):
- A. Preamble
- B. Definitions (I. Standard Definitions, II. Specific Definitions)
- C. Benefits Covered
- D. Waiting Period & Exclusions
- E. General Terms & Clauses
- F. Other terms and conditions


## 2. Niva Bupa (Health Premia)
Confirmed structure so far:
- 1. Preamble
- 2. Definitions & Interpretation (numbered sub-points like 2.1, 2.2...)
- 3. Benefits available under the policy(numbered sub-points like 3.1, 3.2, 3.3...)
- 4. Optional benefits(numbered sub-points like 4.1, 4.2, 4.3....)
- 5. Exclusions (mentioned in preamble, not confirmed yet)
- 6. General Terms and Clauses(numbered sub-points like 6.1, 6.2, 6.3....)


## 3. Star Health (Comprehensive Insurance Policy)
Confirmed structure:
- Customer Information Sheet (Q&A style table — "What am I covered for," "What are the Major Exclusions")
- Main policy body referenced as Section I, II, III (e.g. III(A), III(A)(17))

[Customer Information Sheet ends at 2nd page and the main policy body (Section I onwards) begins at 3rd page.]

## Chunking rule
Split each document at its section headers (not by fixed character count), so each chunk stays a complete, meaningful unit.

## Metadata tags per chunk
Every chunk will be tagged with:
- insurer: hdfc_ergo | niva_bupa | star_health
- section_type: preamble | definitions | benefits | exclusions | waiting_period | terms

## Pre-processing
Strip the repeated header/footer block before chunking (company name, address, phone, CIN, IRDAI Reg No. — repeats on every page, adds no value).