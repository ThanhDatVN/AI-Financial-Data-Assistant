from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.submission.validate import validate_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate schema, evidence, and execution")
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/raw/ViFinQA/questions/questions.jsonl",
    )
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--no-execute", action="store_true")
    parser.add_argument("--execution-timeout", type=float, default=10.0)
    args = parser.parse_args()
    predictions = validate_submission(
        args.submission,
        questions_path=args.questions,
        evidence_root=args.evidence_root,
        execute=not args.no_execute,
        execution_timeout=args.execution_timeout,
    )
    print(f"valid submission: {len(predictions)} questions")


if __name__ == "__main__":
    main()
