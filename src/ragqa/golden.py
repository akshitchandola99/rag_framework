"""Phase 5: load the golden set and check that its evidence quotes really exist."""

import json
from dataclasses import dataclass
from pathlib import Path

from ragqa.loaders import Page

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_V0_PATH = PROJECT_ROOT / "eval" / "datasets" / "golden_v0.jsonl"

REQUIRED_FIELDS = [
    "id",
    "question",
    "category",
    "expected_behavior",
    "reference_answer",
    "expected_evidence",
    "notes",
    "verified_by",
    "verified_on",
]


# One piece of expected evidence: an exact quote and where it lives.
@dataclass
class Evidence:
    source: str
    page: int
    quote: str


# One hand-written test question with its expected answer and evidence.
@dataclass
class GoldenItem:
    id: str
    question: str
    category: str
    expected_behavior: str
    reference_answer: str
    expected_evidence: list[Evidence]
    notes: str
    verified_by: str
    verified_on: str


# Squash all runs of spaces and line breaks into single spaces.
def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


# Read a JSON Lines file into golden items, one item per line.
def load_golden_set(path: Path) -> list[GoldenItem]:
    items = []
    with open(path, encoding="utf-8") as golden_file:
        for line_number, line in enumerate(golden_file, start=1):
            if line.strip() == "":
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name} line {line_number}: not valid JSON ({error})") from error
            for field_name in REQUIRED_FIELDS:
                if field_name not in record:
                    raise ValueError(f"{path.name} line {line_number}: missing field '{field_name}'")
            evidence_list = []
            for evidence_record in record["expected_evidence"]:
                evidence_list.append(
                    Evidence(
                        source=evidence_record["source"],
                        page=evidence_record["page"],
                        quote=evidence_record["quote"],
                    )
                )
            record["expected_evidence"] = evidence_list
            items.append(GoldenItem(**record))
    return items


# Return a problem message for every quote that can't be found on its page.
def find_missing_quotes(items: list[GoldenItem], pages: list[Page]) -> list[str]:
    problems = []
    for item in items:
        for evidence in item.expected_evidence:
            found = False
            for page in pages:
                same_place = page.source == evidence.source and page.page == evidence.page
                if same_place and normalize_whitespace(evidence.quote) in normalize_whitespace(page.text):
                    found = True
                    break
            if not found:
                problems.append(f"{item.id}: quote not found on {evidence.source} page {evidence.page}: {evidence.quote!r}")
    return problems
