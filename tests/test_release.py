"""Release bundle round-trip, tamper rejection, and budget audit contracts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from kojev.release import (
    MANIFEST_NAME,
    PAYLOAD_DIR,
    BundleError,
    audit_budget,
    build_bundle,
    load_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path


def _source_tree(root: Path) -> Path:
    src = root / "src"
    (src / "nested").mkdir(parents=True)
    _ = (src / "config.json").write_text(
        json.dumps({"backbone": "skt/A.X-Encoder-base", "markers": ["[STATE]"]}),
        encoding="utf-8",
    )
    _ = (src / "weights.bin").write_bytes(b"\x00\x01\x02\x03" * 64)
    _ = (src / "nested" / "tokenizer.json").write_text("{}", encoding="utf-8")
    return src


def test_round_trip_restores_every_file_into_a_clean_directory(
    tmp_path: Path,
) -> None:
    src = _source_tree(tmp_path)
    bundle = tmp_path / "bundle"
    restored = tmp_path / "restored"

    manifest = build_bundle(src, bundle)
    returned = load_bundle(bundle, restored)

    assert manifest["file_count"] == 3
    assert returned["file_count"] == 3
    for relative in ("config.json", "weights.bin", "nested/tokenizer.json"):
        assert (restored / relative).read_bytes() == (src / relative).read_bytes()


def test_rejects_tampered_config_with_typed_error(tmp_path: Path) -> None:
    src = _source_tree(tmp_path)
    bundle = tmp_path / "bundle"
    _ = build_bundle(src, bundle)

    config = bundle / PAYLOAD_DIR / "config.json"
    _ = config.write_text(
        json.dumps({"backbone": "attacker/swapped", "markers": []}),
        encoding="utf-8",
    )

    with pytest.raises(BundleError, match="content hash mismatch") as caught:
        _ = load_bundle(bundle, tmp_path / "restored")

    assert type(caught.value) is BundleError
    assert "config.json" in str(caught.value)


def test_rejects_tampered_payload_bytes_naming_the_offending_path(
    tmp_path: Path,
) -> None:
    src = _source_tree(tmp_path)
    bundle = tmp_path / "bundle"
    _ = build_bundle(src, bundle)

    _ = (bundle / PAYLOAD_DIR / "nested" / "tokenizer.json").write_text(
        '{"tampered": true}', encoding="utf-8"
    )

    with pytest.raises(BundleError, match=r"nested/tokenizer\.json"):
        _ = load_bundle(bundle, tmp_path / "restored")


def test_rejects_manifest_listed_file_that_is_missing(tmp_path: Path) -> None:
    src = _source_tree(tmp_path)
    bundle = tmp_path / "bundle"
    _ = build_bundle(src, bundle)

    (bundle / PAYLOAD_DIR / "weights.bin").unlink()

    with pytest.raises(BundleError, match="missing file") as caught:
        _ = load_bundle(bundle, tmp_path / "restored")

    assert "weights.bin" in str(caught.value)


def test_rejects_unsupported_manifest_version(tmp_path: Path) -> None:
    src = _source_tree(tmp_path)
    bundle = tmp_path / "bundle"
    _ = build_bundle(src, bundle)

    manifest_path = bundle / MANIFEST_NAME
    _ = manifest_path.write_text(
        json.dumps({"bundle_version": 99, "files": []}), encoding="utf-8"
    )

    with pytest.raises(BundleError, match="unsupported bundle_version"):
        _ = load_bundle(bundle, tmp_path / "restored")


def test_refuses_to_restore_into_a_non_empty_destination(tmp_path: Path) -> None:
    src = _source_tree(tmp_path)
    bundle = tmp_path / "bundle"
    _ = build_bundle(src, bundle)
    dirty = tmp_path / "restored"
    dirty.mkdir()
    _ = (dirty / "leftover.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(BundleError, match="destination is not empty"):
        _ = load_bundle(bundle, dirty)


def test_budget_audit_totals_ledger_costs_exactly(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _ = ledger.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"kind": "usage", "cost_usd": 0.25},
                {"kind": "usage", "cost_usd": 1.5},
                {"kind": "marker", "marker": "BUDGET_REACHED"},
                {"kind": "usage", "cost_usd": 0.25},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    audit = audit_budget(ledger)

    assert audit.records == 3
    assert audit.total_usd == pytest.approx(2.0)
    assert audit.under_cap is True


def test_budget_audit_flags_a_ledger_that_reaches_the_cap(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _ = ledger.write_text(
        json.dumps({"kind": "usage", "cost_usd": 100.0}) + "\n", encoding="utf-8"
    )

    audit = audit_budget(ledger)

    assert audit.total_usd == pytest.approx(100.0)
    assert audit.under_cap is False


def test_budget_audit_treats_a_missing_ledger_as_zero_spend(tmp_path: Path) -> None:
    audit = audit_budget(tmp_path / "absent.jsonl")

    assert audit.records == 0
    assert audit.total_usd == pytest.approx(0.0)
    assert audit.under_cap is True


def test_build_rejects_an_empty_source_tree(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(BundleError, match="no files"):
        _ = build_bundle(empty, tmp_path / "bundle")
