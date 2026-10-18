import pytest

from ragqa import golden
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder

GOOD_LINE = '{"id": "t1", "question": "Q?", "category": "single_fact", "expected_behavior": "answer", "reference_answer": "A.", "expected_evidence": [{"source": "a.md", "page": 1, "quote": "hello"}], "notes": "", "verified_by": "me", "verified_on": "2026-10-15"}'


# A valid JSON Lines file loads into GoldenItem and Evidence records.
def test_golden_file_loads_into_records(tmp_path):
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(GOOD_LINE + "\n", encoding="utf-8")

    items = golden.load_golden_set(golden_path)

    assert len(items) == 1
    assert items[0].id == "t1"
    assert items[0].expected_evidence[0].quote == "hello"


# A line missing a required field is rejected with its line number.
def test_missing_field_raises(tmp_path):
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text('{"id": "t1", "question": "Q?"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1: missing field 'category'"):
        golden.load_golden_set(golden_path)


# The real golden_v0 file has 8 items with unique IDs.
def test_golden_v0_has_eight_unique_items():
    items = golden.load_golden_set(golden.GOLDEN_V0_PATH)

    item_ids = set()
    for item in items:
        item_ids.add(item.id)
    assert len(items) == 8
    assert len(item_ids) == 8


# Every evidence quote in golden_v0 exists on its page in the real documents.
def test_every_golden_v0_quote_exists_in_the_documents():
    items = golden.load_golden_set(golden.GOLDEN_V0_PATH)
    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)

    problems = golden.find_missing_quotes(items, pages)

    assert problems == []


# A line that isn't valid JSON is rejected with its line number.
def test_broken_json_line_raises(tmp_path):
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(GOOD_LINE + "\n" + GOOD_LINE[:-1] + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2: not valid JSON"):
        golden.load_golden_set(golden_path)
