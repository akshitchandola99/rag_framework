"""Check an answer's [S1]-style citations and spot refusals."""

import re

from ragqa.generation import REFUSAL_TEXT
from ragqa.golden import Evidence
from ragqa.retrieval import SearchHit
from ragqa.retrieval_metrics import hit_contains_evidence

CITATION_PATTERN = re.compile(r"\[S\d+\]")


# Find every [S1]-style label in an answer, first appearance first, no repeats.
def find_cited_labels(answer_text: str) -> list[str]:
    cited_labels = []
    for label in CITATION_PATTERN.findall(answer_text):
        if label not in cited_labels:
            cited_labels.append(label)
    return cited_labels


# Split cited labels into ones we really sent and ones we did not.
def split_citations(cited_labels: list[str], sent_labels: list[str]) -> tuple[list[str], list[str]]:
    valid_labels = []
    invalid_labels = []
    for label in cited_labels:
        if label in sent_labels:
            valid_labels.append(label)
        else:
            invalid_labels.append(label)
    return valid_labels, invalid_labels


# Lower-case the text and turn curly apostrophes into straight ones.
def normalize_for_refusal(text: str) -> str:
    # Models sometimes write a curly apostrophe (’) instead of a straight one (').
    return text.lower().replace("’", "'").strip()


# True when the answer is the refusal sentence (case and punctuation ignored).
def is_refusal(answer_text: str) -> bool:
    refusal_core = REFUSAL_TEXT.lower().rstrip(".")
    return refusal_core in normalize_for_refusal(answer_text)


# True when the answer opens with the refusal sentence, not just mentions it later.
def starts_with_refusal(answer_text: str) -> bool:
    refusal_core = REFUSAL_TEXT.lower().rstrip(".")
    return normalize_for_refusal(answer_text).startswith(refusal_core)


# True when at least one cited chunk contains one of the expected evidence quotes.
def cites_expected_evidence(
    valid_labels: list[str],
    labelled_hits: list[tuple[str, SearchHit]],
    expected_evidence: list[Evidence],
) -> bool:
    for label, hit in labelled_hits:
        if label not in valid_labels:
            continue
        for evidence in expected_evidence:
            if hit_contains_evidence(hit, evidence):
                return True
    return False


# Run the Phase 7 checks on one answer. Each check is True (pass), False (fail) or None (does not apply).
def check_answer(
    answer_text: str,
    labelled_hits: list[tuple[str, SearchHit]],
    expected_behavior: str,
    expected_evidence: list[Evidence],
) -> dict:
    sent_labels = []
    for label, hit in labelled_hits:
        sent_labels.append(label)
    cited_labels = find_cited_labels(answer_text)
    valid_labels, invalid_labels = split_citations(cited_labels, sent_labels)
    refused = is_refusal(answer_text)

    checks = {}
    if expected_behavior == "refuse":
        checks["behavior_matches"] = refused
    else:
        checks["behavior_matches"] = not refused
    checks["citations_valid"] = len(invalid_labels) == 0

    if refused:
        checks["answer_is_cited"] = None
        checks["cites_evidence"] = None
        checks["refusal_is_clean"] = starts_with_refusal(answer_text) and len(cited_labels) == 0
    else:
        checks["answer_is_cited"] = len(valid_labels) > 0
        if len(expected_evidence) == 0:
            checks["cites_evidence"] = None
        else:
            checks["cites_evidence"] = cites_expected_evidence(valid_labels, labelled_hits, expected_evidence)
        checks["refusal_is_clean"] = None

    return {
        "refused": refused,
        "cited_labels": cited_labels,
        "invalid_labels": invalid_labels,
        "checks": checks,
    }


# Phase 8 output guard: an answer that is not a refusal and cites no source we sent is replaced by the refusal.
def guard_answer(answer_text: str, sent_labels: list[str]) -> tuple[str, bool]:
    if is_refusal(answer_text):
        return answer_text, False
    cited_labels = find_cited_labels(answer_text)
    valid_labels, invalid_labels = split_citations(cited_labels, sent_labels)
    if len(valid_labels) == 0:
        # Nothing in the answer is tied to our documents (for example "PWNED."), so we don't show it.
        return REFUSAL_TEXT, True
    return answer_text, False


# True when no check failed (checks that do not apply are ignored).
def all_checks_pass(checks: dict) -> bool:
    for check_name, result in checks.items():
        if result is False:
            return False
    return True
