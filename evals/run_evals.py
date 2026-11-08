"""Evaluate PDF question answering against evals/cases.json.

Each case checks the answer text (expected_all: every keyword must appear;
expected_any: at least one must appear; whole-word, case-insensitive) and, when
source_page is set, that the top-ranked source chunk comes from that page.

Usage: python evals/run_evals.py    (needs OPENAI_API_KEY; exits 1 on any failure)
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "DocGenius"))

from PyPDF2 import PdfReader  # noqa: E402


def contains(text, keyword):
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text, re.IGNORECASE) is not None


def check(case, answer, sources):
    answer = answer.replace("’", "'")
    problems = []
    missing = [k for k in case.get("expected_all", []) if not contains(answer, k)]
    if missing:
        problems.append(f"missing {missing}")
    any_of = case.get("expected_any", [])
    if any_of and not any(contains(answer, k) for k in any_of):
        problems.append(f"none of {any_of}")
    page = case.get("source_page")
    if page is not None:
        top = sources[0].metadata.get("page") if sources else None
        if top != page:
            problems.append(f"top source is page {top}, expected page {page}")
    return problems


def main():
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))

    # Offline sanity check: every fixture must yield extractable text.
    for pdf in sorted({c["pdf"] for c in cases}):
        text = "".join(p.extract_text() or "" for p in PdfReader(HERE / "fixtures" / pdf).pages)
        if not text.strip():
            print(f"FAIL fixture {pdf} has no extractable text")
            return 1

    if not os.getenv("OPENAI_API_KEY"):
        print("::warning::OPENAI_API_KEY not set; skipping model evals (fixture checks passed).")
        return 0

    from qa import build_knowledge_base, answer_question

    kbs, failures = {}, 0
    for case in cases:
        try:
            if case["pdf"] not in kbs:
                kbs[case["pdf"]] = build_knowledge_base(str(HERE / "fixtures" / case["pdf"]))
            answer, sources = answer_question(kbs[case["pdf"]], case["question"], temperature=0)
            problems = check(case, answer, sources)
        except Exception as e:  # a crashing call is a failed case, not a crashed run
            answer, problems = "", [f"error: {e}"]
        failures += bool(problems)
        status = "PASS" if not problems else "FAIL"
        print(f"{status} {case['id']}: {answer!r}" + (f"  -> {'; '.join(problems)}" if problems else ""))

    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
