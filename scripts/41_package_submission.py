from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.submission.package import package_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and package the competition ZIP")
    parser.add_argument("submission", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/raw/ViFinQA/questions/questions.jsonl",
    )
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reuse-execution-from",
        type=Path,
        help=(
            "Skip re-executing every query, reusing a submission already validated in "
            "full. Refused unless the answers, queries and evidence are identical."
        ),
    )
    args = parser.parse_args()
    output = package_submission(
        args.submission,
        args.output,
        questions_path=args.questions,
        evidence_root=args.evidence_root,
        force=args.force,
        reuse_execution_from=args.reuse_execution_from,
    )
    print(output)


if __name__ == "__main__":
    main()
