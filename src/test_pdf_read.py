from pypdf import PdfReader

reader = PdfReader("data/star_health/SHAHLIP22028V072122_HEALTH2050 (STAR HEALTH).pdf")
print(f"Number of pages: {len(reader.pages)}")

# Print first page's raw text
first_page_text = reader.pages[0].extract_text()
print(first_page_text)