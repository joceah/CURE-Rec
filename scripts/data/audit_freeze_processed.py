"""Audit and freeze processed Beauty/Sports datasets and held-out sample IDs.

This is intentionally read-only with respect to data/raw and data/processed.
It reconstructs the leave-two-out split from the raw review files to verify
that saved train sequences, validation targets, and test targets match the
timestamp ordering used by the preprocessing code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("title", "description", "features", "categories", "store", "details")
TEXT_FIELDS = ("title", "features", "categories")
DATASETS = {
    "beauty_2018_2023": {
        "domain": "beauty",
        "review_file": "Beauty.jsonl",
        "meta_file": "meta_Beauty.jsonl",
    },
    "sports_2018_2023": {
        "domain": "sports",
        "review_file": "Sports.jsonl",
        "meta_file": "meta_Sports.jsonl",
    },
}


def sha256_file(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while hashing: {path}")
    return {"path": path.relative_to(ROOT).as_posix(), "size_bytes": after.st_size,
            "sha256": digest.hexdigest()}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_dense_mapping(mapping: dict[str, Any]) -> tuple[bool, dict[int, str]]:
    values: list[int] = []
    inverse: dict[int, str] = {}
    for raw_id, mapped_id in mapping.items():
        if not isinstance(mapped_id, int) or isinstance(mapped_id, bool):
            return False, {}
        values.append(mapped_id)
        if mapped_id in inverse:
            return False, {}
        inverse[mapped_id] = raw_id
    return set(values) == set(range(len(mapping))), inverse


def read_split(path: Path) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["user_id", "item_id"]:
            raise ValueError(f"Unexpected CSV columns in {path}: {reader.fieldnames}")
        for row in reader:
            rows.append((int(row["user_id"]), int(row["item_id"])))
    return rows


def percentile(sorted_values: list[int], p: float) -> int | None:
    if not sorted_values:
        return None
    return sorted_values[max(0, math.ceil(p * len(sorted_values)) - 1)]


def distribution(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": None, "p50": None, "p90": None,
                "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 4),
    }


def check(checks: dict[str, dict[str, Any]], name: str, passed: bool,
          detail: Any = None) -> None:
    checks[name] = {"passed": bool(passed), "detail": detail}


def audit_reviews(
    path: Path,
    user_to_id: dict[str, int],
    item_to_id: dict[str, int],
    start_ms: int,
    end_ms: int,
) -> tuple[dict[int, list[tuple[int, int]]], dict[str, Any]]:
    before = path.stat()
    digest = hashlib.sha256()
    events: dict[int, list[tuple[int, int]]] = defaultdict(list)
    total_rows = in_window = mapped_rows = malformed = invalid_timestamps = 0

    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            total_rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            timestamp = record.get("timestamp", 0)
            if not isinstance(timestamp, int) or isinstance(timestamp, bool):
                invalid_timestamps += 1
                continue
            if timestamp < start_ms or timestamp > end_ms:
                continue
            in_window += 1
            user_id = user_to_id.get(record.get("user_id"))
            item_id = item_to_id.get(record.get("parent_asin"))
            if user_id is not None and item_id is not None:
                events[user_id].append((timestamp, item_id))
                mapped_rows += 1

    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Raw review file changed during audit: {path}")
    source = {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": after.st_size,
        "sha256": digest.hexdigest(),
        "jsonl_records": total_rows,
        "malformed_json_records_skipped_like_preprocessor": malformed,
        "invalid_timestamp_records": invalid_timestamps,
        "records_in_date_window": in_window,
        "records_with_both_final_mapped_ids": mapped_rows,
    }
    return events, source


def audit_dataset(name: str, raw_dir: Path, processed_dir: Path) -> dict[str, Any]:
    spec = DATASETS[name]
    folder = processed_dir / name
    expected_files = (
        "item2id.json", "user2id.json", "item_meta.json", "user_sequences.json",
        "valid.csv", "test.csv",
    )
    missing_files = [file for file in expected_files if not (folder / file).is_file()]
    if missing_files:
        raise FileNotFoundError(f"{name} is missing processed files: {missing_files}")

    checks: dict[str, dict[str, Any]] = {}
    item_to_id = load_json(folder / "item2id.json")
    user_to_id = load_json(folder / "user2id.json")
    item_dense, item_inverse = validate_dense_mapping(item_to_id)
    user_dense, user_inverse = validate_dense_mapping(user_to_id)
    check(checks, "item_id_mapping_bijective_and_dense", item_dense,
          {"mapped_items": len(item_to_id)})
    check(checks, "user_id_mapping_bijective_and_dense", user_dense,
          {"mapped_users": len(user_to_id)})

    raw_sequences = load_json(folder / "user_sequences.json")
    sequences = {int(user): [int(item) for item in item_list]
                 for user, item_list in raw_sequences.items()}
    valid_rows = read_split(folder / "valid.csv")
    test_rows = read_split(folder / "test.csv")
    valid_by_user = {user: item for user, item in valid_rows}
    test_by_user = {user: item for user, item in test_rows}
    valid_users = [user for user, _ in valid_rows]
    test_users = [user for user, _ in test_rows]

    check(checks, "one_valid_and_test_target_per_user",
          len(valid_by_user) == len(valid_rows) and len(test_by_user) == len(test_rows),
          {"valid_rows": len(valid_rows), "test_rows": len(test_rows)})
    expected_users = set(user_inverse)
    sequence_users = set(sequences)
    check(checks, "sequence_users_match_user_mapping",
          sequence_users == expected_users,
          {"mapped_users": len(expected_users), "sequence_users": len(sequence_users),
           "missing_sequence_users": len(expected_users - sequence_users),
           "extra_sequence_users": len(sequence_users - expected_users)})
    check(checks, "valid_and_test_users_match_user_mapping",
          set(valid_users) == expected_users and set(test_users) == expected_users,
          {"valid_users": len(set(valid_users)), "test_users": len(set(test_users)),
           "mapped_users": len(expected_users)})

    metadata = load_json(folder / "item_meta.json")
    metadata = {int(item): record for item, record in metadata.items()}
    item_ids = set(item_inverse)
    meta_ids = set(metadata)
    check(checks, "metadata_ids_match_item_mapping", meta_ids == item_ids,
          {"mapped_items": len(item_ids), "metadata_items": len(meta_ids),
           "missing_metadata": len(item_ids - meta_ids),
           "extra_metadata": len(meta_ids - item_ids)})

    metadata_missing: dict[str, Any] = {}
    for field in FIELDS:
        missing_key = sum(field not in row for row in metadata.values())
        empty_value = sum(not str(row.get(field) or "").strip()
                          for row in metadata.values())
        metadata_missing[field] = {
            "missing_key_count": missing_key,
            "empty_or_whitespace_count": empty_value,
            "empty_or_whitespace_rate": round(empty_value / max(1, len(metadata)), 6),
        }
    no_selected_text = sum(
        not any(str(metadata.get(item, {}).get(field) or "").strip()
                for field in TEXT_FIELDS)
        for item in item_ids
    )

    all_rows = valid_rows + test_rows
    bad_target_ids = [(split, row_index, user, item)
                      for split, rows in (("valid", valid_rows), ("test", test_rows))
                      for row_index, (user, item) in enumerate(rows)
                      if item not in item_ids or item not in meta_ids]
    bad_users = [(split, index, user)
                 for split, rows in (("valid", valid_rows), ("test", test_rows))
                 for index, (user, _) in enumerate(rows) if user not in expected_users]
    check(checks, "heldout_targets_covered_by_item_map_and_metadata", not bad_target_ids,
          {"valid_targets": len(valid_rows), "test_targets": len(test_rows),
           "bad_target_rows": len(bad_target_ids), "examples": bad_target_ids[:5]})
    check(checks, "heldout_users_covered_by_user_map", not bad_users,
          {"bad_user_rows": len(bad_users), "examples": bad_users[:5]})

    lengths = [len(items) for items in sequences.values()]
    train_interactions = sum(lengths)
    item_frequency: Counter[int] = Counter()
    for items in sequences.values():
        item_frequency.update(items)
    item_frequency.update(item for _, item in all_rows)
    total_user_degrees = [length + 2 for length in lengths]
    check(checks, "all_sequences_and_targets_use_mapped_items",
          set(item_frequency).issubset(item_ids),
          {"unmapped_item_ids": len(set(item_frequency) - item_ids)})
    check(checks, "k_core_minimum_degree_at_least_five",
          min(total_user_degrees, default=0) >= 5 and
          min((item_frequency.get(item, 0) for item in item_ids), default=0) >= 5,
          {"min_user_interactions": min(total_user_degrees, default=0),
           "min_item_interactions": min((item_frequency.get(item, 0)
                                          for item in item_ids), default=0)})

    start_ms = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(2023, 12, 31, 23, 59, 59, 999000,
                          tzinfo=timezone.utc).timestamp() * 1000)
    events, raw_review = audit_reviews(
        raw_dir / spec["review_file"], user_to_id, item_to_id, start_ms, end_ms
    )

    exact_train = exact_valid = exact_test = 0
    timestamp_tie_users = timestamp_tie_adjacencies = 0
    equal_valid_test_timestamp_users = 0
    mismatch_examples: list[dict[str, Any]] = []
    raw_mapped_interactions = sum(map(len, events.values()))
    for user_id in sorted(expected_users | set(events)):
        user_events = events.get(user_id, [])
        user_events.sort(key=lambda pair: pair[0])  # stable: source order breaks timestamp ties
        if any(user_events[i][0] == user_events[i - 1][0]
               for i in range(1, len(user_events))):
            timestamp_tie_users += 1
            timestamp_tie_adjacencies += sum(
                user_events[i][0] == user_events[i - 1][0]
                for i in range(1, len(user_events))
            )
        ordered_items = [item for _, item in user_events]
        expected_train = ordered_items[:-2] if len(ordered_items) >= 2 else []
        expected_valid = ordered_items[-2] if len(ordered_items) >= 2 else None
        expected_test = ordered_items[-1] if ordered_items else None
        actual_train = sequences.get(user_id)
        actual_valid = valid_by_user.get(user_id)
        actual_test = test_by_user.get(user_id)
        train_ok = actual_train == expected_train
        valid_ok = actual_valid == expected_valid
        test_ok = actual_test == expected_test
        exact_train += train_ok
        exact_valid += valid_ok
        exact_test += test_ok
        if len(user_events) >= 2 and user_events[-2][0] == user_events[-1][0]:
            equal_valid_test_timestamp_users += 1
        if not (train_ok and valid_ok and test_ok) and len(mismatch_examples) < 10:
            mismatch_examples.append({
                "user_id": user_id,
                "raw_mapped_events": len(user_events),
                "stored_train_length": len(actual_train) if actual_train is not None else None,
                "train_exact": train_ok,
                "valid_exact": valid_ok,
                "test_exact": test_ok,
            })

    temporal_ok = (
        len(events) == len(expected_users)
        and raw_mapped_interactions == train_interactions + len(valid_rows) + len(test_rows)
        and exact_train == len(expected_users)
        and exact_valid == len(expected_users)
        and exact_test == len(expected_users)
    )
    check(checks, "raw_timestamp_order_reproduces_saved_leave_two_out_split",
          temporal_ok,
          {"mapped_raw_users": len(events), "mapped_users": len(expected_users),
           "mapped_raw_interactions": raw_mapped_interactions,
           "stored_interactions": train_interactions + len(valid_rows) + len(test_rows),
           "exact_train_histories": exact_train, "exact_valid_targets": exact_valid,
           "exact_test_targets": exact_test,
           "same_timestamp_users": timestamp_tie_users,
           "same_timestamp_adjacent_pairs": timestamp_tie_adjacencies,
           "valid_test_same_timestamp_users": equal_valid_test_timestamp_users,
           "mismatch_examples": mismatch_examples})

    raw_meta = sha256_file(raw_dir / spec["meta_file"])
    processed_hashes = {
        filename: sha256_file(folder / filename)
        for filename in expected_files
    }
    processing_code = {
        relative: sha256_file(ROOT / relative)
        for relative in ("src/data/amazon_process.py", "scripts/data/run_process.py")
    }
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            check=True, capture_output=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_head = None

    metrics = {
        "counts": {
            "users": len(expected_users), "items": len(item_ids),
            "train_interactions": train_interactions,
            "valid_samples": len(valid_rows), "test_samples": len(test_rows),
            "total_interactions": train_interactions + len(valid_rows) + len(test_rows),
            "unique_valid_target_items": len({item for _, item in valid_rows}),
            "unique_test_target_items": len({item for _, item in test_rows}),
            "unique_heldout_target_items": len({item for _, item in all_rows}),
        },
        "train_history_length": distribution(lengths),
        "heldout_target_coverage": {
            "valid_item_map_rate": round(sum(item in item_ids for _, item in valid_rows) /
                                          max(1, len(valid_rows)), 8),
            "test_item_map_rate": round(sum(item in item_ids for _, item in test_rows) /
                                         max(1, len(test_rows)), 8),
            "valid_metadata_rate": round(sum(item in meta_ids for _, item in valid_rows) /
                                          max(1, len(valid_rows)), 8),
            "test_metadata_rate": round(sum(item in meta_ids for _, item in test_rows) /
                                         max(1, len(test_rows)), 8),
        },
        "metadata_missingness": {
            "item_count": len(metadata), "fields": metadata_missing,
            "items_empty_across_stage2_text_fields": no_selected_text,
            "stage2_text_fields": list(TEXT_FIELDS),
        },
        "degree_audit": {
            "user_interactions_including_two_targets": distribution(total_user_degrees),
            "item_interaction_frequency": distribution(list(item_frequency.values())),
        },
        "temporal_order": checks["raw_timestamp_order_reproduces_saved_leave_two_out_split"]["detail"],
    }

    failed = [name for name, result in checks.items() if not result["passed"]]
    return {
        "dataset": name,
        "domain": spec["domain"],
        "processed_directory": folder.relative_to(ROOT).as_posix(),
        "checks": checks,
        "failed_checks": failed,
        "metrics": metrics,
        "preprocessing_recipe": {
            "review_date_window_utc_inclusive": ["2018-01-01", "2023-12-31"],
            "k_core_threshold": 5,
            "split": "per-user timestamp ascending; final interaction=test, previous=valid",
            "timestamp_tie_break": "stable source JSONL order, matching Python sorted()",
            "source_timezone": "UTC",
        },
        "source_artifacts": {"reviews": raw_review, "metadata": raw_meta},
        "processed_artifacts": processed_hashes,
        "preprocessing_code": processing_code,
        "git_head": git_head,
        "sample_rows": {"valid": valid_rows, "test": test_rows},
    }


def write_sample_ids(path: Path, dataset: str, split: str,
                     rows: list[tuple[int, int]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["row_index", "sample_id", "user_id", "target_item_id"])
        for row_index, (user_id, item_id) in enumerate(rows):
            writer.writerow([row_index, f"{dataset}:{split}:{user_id}:{item_id}",
                             user_id, item_id])
    return {**sha256_file(path), "rows": len(rows),
            "sample_id_format": f"{dataset}:{split}:<user_id>:<target_item_id>"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--force", action="store_true",
                        help="replace existing freeze outputs after a complete passing audit")
    args = parser.parse_args()
    manifest_dir = args.manifest_dir.resolve()
    if (manifest_dir / "freeze.sha256").exists() and not args.force:
        parser.error(f"Freeze already exists at {manifest_dir}; pass --force to regenerate")

    start = datetime.now(timezone.utc)
    results = [audit_dataset(name, args.raw_dir.resolve(), args.processed_dir.resolve())
               for name in DATASETS]
    failed = [(result["dataset"], check_name)
              for result in results for check_name in result["failed_checks"]]
    if failed:
        for result in results:
            print(f"{result['dataset']}: {'PASS' if not result['failed_checks'] else 'FAIL'}")
            for check_name in result["failed_checks"]:
                print(f"  failed: {check_name}: {result['checks'][check_name]['detail']}")
        print("Freeze outputs were not written because blocking audit checks failed.",
              file=sys.stderr)
        return 1

    manifest_dir.mkdir(parents=True, exist_ok=True)
    frozen_files: list[Path] = []
    for result in results:
        name = result["dataset"]
        rows = result.pop("sample_rows")
        domain_dir = manifest_dir / name
        samples = {}
        for split in ("valid", "test"):
            samples[split] = write_sample_ids(
                domain_dir / f"{split}_sample_ids.csv", name, split, rows[split]
            )
            samples[split]["path"] = (domain_dir / f"{split}_sample_ids.csv").relative_to(ROOT).as_posix()
            frozen_files.append(domain_dir / f"{split}_sample_ids.csv")
        result["fixed_sample_ids"] = samples
        result["freeze_status"] = "frozen"
        result["audit_timestamp_utc"] = start.isoformat()
        result["python_version"] = platform.python_version()
        result["audit_script"] = sha256_file(Path(__file__).resolve())
        manifest_path = domain_dir / "manifest.json"
        domain_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        frozen_files.append(manifest_path)
        print(f"{name}: FROZEN; users={result['metrics']['counts']['users']:,}; "
              f"items={result['metrics']['counts']['items']:,}; "
              f"train={result['metrics']['counts']['train_interactions']:,}; "
              f"valid/test={result['metrics']['counts']['valid_samples']:,}/"
              f"{result['metrics']['counts']['test_samples']:,}")

    sidecar = manifest_dir / "freeze.sha256"
    sidecar_lines = []
    for path in sorted(frozen_files):
        entry = sha256_file(path)
        sidecar_lines.append(f"{entry['sha256']}  {entry['path']}")
    sidecar.write_text("\n".join(sidecar_lines) + "\n", encoding="utf-8")
    print(f"Freeze manifests and sample IDs: {manifest_dir.relative_to(ROOT)}")
    print(f"Checksum index: {sidecar.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
