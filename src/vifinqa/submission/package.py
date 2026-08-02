from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from vifinqa.submission.validate import validate_submission


def package_submission(
    submission_path: Path,
    output_zip: Path,
    *,
    questions_path: Path,
    evidence_root: Path | None = None,
    force: bool = False,
) -> Path:
    if output_zip.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing ZIP: {output_zip}")
    root = evidence_root or submission_path.parent
    predictions = validate_submission(
        submission_path,
        questions_path=questions_path,
        evidence_root=root,
        execute=True,
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
