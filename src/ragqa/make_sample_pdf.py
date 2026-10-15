"""Phase 1 helper: write the fictional sample PDF into data/raw/."""

from pathlib import Path

import pymupdf

# Resolve from this file so it works no matter which folder you run from
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "brightwater_travel_policy.pdf"

PAGE_TEXTS = [
    (
        "Brightwater Robotics - Travel and Expenses Policy\n"
        "Version 2.1, effective 1 March 2026\n\n"
        "1. Who this policy covers\n"
        "This policy applies to every Brightwater employee and contractor who travels "
        "for work. Travel must be approved in the Compass portal by your team lead "
        "at least 10 working days before departure."
    ),
    (
        "2. Booking travel\n"
        "Trains are the default for trips under 4 hours. Flights must be economy class, "
        "except for flights longer than 8 hours, which may be premium economy.\n\n"
        "3. Hotels\n"
        "The nightly hotel limit is 140 euros in most cities and 190 euros in London, "
        "Paris and Zurich."
    ),
    "",
    (
        "4. Daily allowance and claims\n"
        "The daily meal allowance is 45 euros. Alcohol is never reimbursed. "
        "Submit receipts in Compass within 30 days of returning; late claims are "
        "rejected automatically."
    ),
]


# Write a PDF with one page per text (an empty text gives a blank page).
def write_pdf(output_path: Path, page_texts: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_document = pymupdf.open()
    for page_text in page_texts:
        new_page = pdf_document.new_page()
        if page_text != "":
            text_area = pymupdf.Rect(72, 72, 523, 770)
            new_page.insert_textbox(text_area, page_text, fontsize=11, fontname="helv")
    pdf_document.save(output_path)
    pdf_document.close()


# Create the sample travel-policy PDF in data/raw.
def main() -> None:
    write_pdf(OUTPUT_PATH, PAGE_TEXTS)
    print(f"Wrote {OUTPUT_PATH} with {len(PAGE_TEXTS)} pages (page 3 is blank on purpose)")


if __name__ == "__main__":
    main()