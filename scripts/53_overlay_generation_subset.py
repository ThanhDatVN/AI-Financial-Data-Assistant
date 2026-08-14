"""Overlay a generated experiment subset onto a fixed full-run baseline.

The public-score loop reruns only the immutable 200-question subset.  This utility replaces
exactly those predictions, keeps the other baseline rows, and builds a fresh evidence directory
so the result can pass the normal submission validator and packager.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_submission(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"{path} must contain a JSON list of objects")
    return rows


def _by_id(rows: list[dict[str, Any]], *, label: str) -> dict[int, dict[str, Any]]:
    indexed = {int(row["id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate IDs in {label}")
    return indexed


def overlay_predictions(
    baseline: list[dict[str, Any]],
    variant: list[dict[str, Any]],
    expected_variant_ids: set[int],
) -> list[dict[str, Any]]:
    baseline_by_id = _by_id(baseline, label="baseline")
    variant_by_id = _by_id(variant, label="variant")
    actual_variant_ids = set(variant_by_id)
    if actual_variant_ids != expected_variant_ids:
        missing = sorted(expected_variant_ids - actual_variant_ids)
        extra = sorted(actual_variant_ids - expected_variant_ids)
        raise ValueError(f"Variant ID mismatch; missing={missing[:20]}, extra={extra[:20]}")
    if not expected_variant_ids <= set(baseline_by_id):
        raise ValueError("Variant IDs must all exist in the baseline")
    baseline_by_id.update(variant_by_id)
    return [baseline_by_id[question_id] for question_id in sorted(baseline_by_id)]


def _copy_evidence(rows: list[dict[str, Any]], source_root: Path, output_root: Path) -> None:
    for row in rows:
        evidence = row.get("evidence")
        if not isinstance(evidence, list):
            raise TypeError(f"Prediction id={row.get('id')} has invalid evidence")
        for item in evidence:
            if not isinstance(item, dict) or not isinstance(item.get("csv_path"), str):
                raise TypeError(f"Prediction id={row.get('id')} has invalid evidence item")
            relative = Path(item["csv_path"])
            source = source_root / relative
            destination = output_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"Missing evidence file: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline generation root")
    parser.add_argument("--variant", required=True, type=Path, help="Subset generation root")
    parser.add_argument(
        "--variant-submission",
        type=Path,
        help="Finalized subset JSON; defaults to <variant>/submission.json",
    )
    parser.add_argument("--subset", required=True, type=Path, help="Fixed subset JSON config")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    baseline_path = args.baseline / "submission.json"
    variant_path = args.variant_submission or args.variant / "submission.json"
    subset = json.loads(args.subset.read_text(encoding="utf-8"))
    subset_ids = subset.get("ids") if isinstance(subset, dict) else None
    if not isinstance(subset_ids, list) or not subset_ids:
        raise TypeError("Subset config must contain a non-empty ids list")
    expected_variant_ids = {int(question_id) for question_id in subset_ids}
    if len(expected_variant_ids) != len(subset_ids):
        raise ValueError("Subset config contains duplicate IDs")

    baseline = _load_submission(baseline_path)
    variant = _load_submission(variant_path)
    merged = overlay_predictions(baseline, variant, expected_variant_ids)
    args.output.mkdir(parents=True)
    baseline_ids = set(_by_id(baseline, label="baseline"))
    baseline_rows = [row for row in merged if int(row["id"]) in baseline_ids - expected_variant_ids]
    variant_rows = [row for row in merged if int(row["id"]) in expected_variant_ids]
    _copy_evidence(baseline_rows, args.baseline, args.output)
    _copy_evidence(variant_rows, args.variant, args.output)

    output_submission = args.output / "submission.json"
    output_submission.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "baseline_submission_sha256": _sha256(baseline_path),
        "variant_submission_sha256": _sha256(variant_path),
        "subset_config_sha256": _sha256(args.subset),
        "replaced_ids": sorted(expected_variant_ids),
        "rows": len(merged),
        "submission_sha256": _sha256(output_submission),
    }
    (args.output / "overlay_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"baseline={len(baseline)} replaced={len(expected_variant_ids)} "
        f"merged={len(merged)} output={args.output}"
    )


if __name__ == "__main__":
    main()
