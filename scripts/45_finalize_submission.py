"""Apply the organizer answer convention and the measured z4 citation profile.

This is the single pre-packaging entry point for a newly generated submission.  It keeps the
answer/query parity fix and the citation-only transforms in a fixed, auditable order so a raw
generation ZIP cannot accidentally be uploaded as the final candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Raw generated submission JSON")
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.submission.resolve():
        parser.error("--output must differ from the raw submission")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vifinqa-finalize-") as temporary:
        temp_root = Path(temporary)
        semantic = temp_root / "submission_semantic.json"
        widened = temp_root / "submission_z4_tables.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/44_apply_semantic_conventions.py"),
                str(args.submission),
                "--output",
                str(semantic),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/43_widen_citations.py"),
                str(semantic),
                "--retrieval",
                str(args.retrieval),
                "--manifest",
                str(args.manifest),
                "--output",
                str(widened),
                "--tables",
                "5",
                "--table-router",
                "consolidated",
                "--doc-router",
                "consolidated",
                "--router-only",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/42_retarget_table_refs.py"),
                str(widened),
                "--output",
                str(args.output),
                "--grammar",
                "line",
                "--manifest",
                str(args.manifest),
            ],
            check=True,
        )
    print(f"finalized={args.output} sha256={_sha256(args.output)} profile=z4_abs_line")


if __name__ == "__main__":
    main()
