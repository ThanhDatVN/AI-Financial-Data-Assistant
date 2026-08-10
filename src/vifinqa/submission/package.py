from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from vifinqa.submission.validate import validate_submission


def _reuses_validated_execution(candidate: Path, validated: Path) -> bool:
    """Whether two submissions differ only in the fields execution never touches.

    Testing which table-reference grammar the dashboard accepts means packaging the same
    answers half a dozen times over. Re-executing 1,012 queries for each costs a quarter of
    an hour to re-confirm something already confirmed, but skipping execution is only safe
    when the queries and the evidence they read are identical, so check that rather than
    trust the caller.
    """
    reference = {
        int(row["id"]): (row["question"], row["answer"], row["pandas_query"], row["evidence"])
        for row in json.loads(validated.read_text(encoding="utf-8"))
    }
    rows = json.loads(candidate.read_text(encoding="utf-8"))
    if len(rows) != len(reference):
        return False
    return all(
        int(row["id"]) in reference
        and reference[int(row["id"])]
        == (row["question"], row["answer"], row["pandas_query"], row["evidence"])
        for row in rows
    )


def package_submission(
    submission_path: Path,
    output_zip: Path,
    *,
    questions_path: Path,
    evidence_root: Path | None = None,
    force: bool = False,
    reuse_execution_from: Path | None = None,
) -> Path:
    if output_zip.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing ZIP: {output_zip}")
    root = evidence_root or submission_path.parent
    execute = True
    if reuse_execution_from is not None:
        if not _reuses_validated_execution(submission_path, reuse_execution_from):
            raise ValueError(
                f"{submission_path.name} changes an answer, a query or its evidence relative "
                f"to {reuse_execution_from.name}, so its execution has to be checked."
            )
        execute = False
    predictions = validate_submission(
        submission_path,
        questions_path=questions_path,
        evidence_root=root,
        execute=execute,
    )
    canonical = [prediction.model_dump(mode="json") for prediction in predictions]
    csv_paths = sorted({item.csv_path for row in predictions for item in row.evidence})
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(
            "submission.json",
            json.dumps(canonical, ensure_ascii=False, indent=2) + "\n",
        )
        for relative in csv_paths:
            archive.write(root / Path(relative), arcname=relative)
    return output_zip
