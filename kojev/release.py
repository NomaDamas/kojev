"""Release bundle packaging with manifest verification and budget audit.

Bundles are local artefacts only: this module never uploads, publishes, or
otherwise contacts the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override

if TYPE_CHECKING:
    from kojev.schema import JsonValue

BUNDLE_VERSION: Final = 1
MANIFEST_NAME: Final = "manifest.json"
PAYLOAD_DIR: Final = "payload"
BUDGET_CAP_USD: Final = 100.0
_HASH_CHUNK: Final = 65536


class BundleError(ValueError):
    """A bundle failed structural, manifest, or configuration validation.

    Deliberately a plain exception subclass rather than a frozen dataclass:
    frozen dataclass exceptions reject ``__traceback__`` assignment, which has
    already bitten this codebase once in the teacher client.
    """

    def __init__(self, reason: str) -> None:
        """Record the structured reason."""
        super().__init__(reason)
        self.reason: str = reason

    @override
    def __str__(self) -> str:
        """Return the structured reason."""
        return self.reason

    @classmethod
    def missing_source(cls, src: Path) -> BundleError:
        """Reject a source directory that does not exist."""
        return cls(f"source directory does not exist: {src}")

    @classmethod
    def empty_source(cls, src: Path) -> BundleError:
        """Reject a source directory holding no files."""
        return cls(f"source directory has no files: {src}")

    @classmethod
    def missing_manifest(cls, bundle: Path) -> BundleError:
        """Reject a bundle without a manifest."""
        return cls(f"bundle is missing {MANIFEST_NAME}: {bundle}")

    @classmethod
    def missing_payload(cls, bundle: Path) -> BundleError:
        """Reject a bundle without a payload directory."""
        return cls(f"bundle is missing {PAYLOAD_DIR}/: {bundle}")

    @classmethod
    def unreadable_manifest(cls, path: Path) -> BundleError:
        """Reject a manifest that is not valid JSON."""
        return cls(f"{path}: manifest is not valid JSON")

    @classmethod
    def malformed_manifest(cls, path: Path, detail: str) -> BundleError:
        """Reject a manifest whose structure is wrong."""
        return cls(f"{path}: {detail}")

    @classmethod
    def unsupported_version(cls, path: Path, version: object) -> BundleError:
        """Reject a manifest written by an incompatible bundler."""
        detail = f"unsupported bundle_version {version!r}, expected {BUNDLE_VERSION}"
        return cls(f"{path}: {detail}")

    @classmethod
    def missing_file(cls, relative: str) -> BundleError:
        """Reject a manifest entry whose payload file is absent."""
        return cls(f"manifest lists a missing file: {relative}")

    @classmethod
    def hash_mismatch(cls, relative: str, expected: str, actual: str) -> BundleError:
        """Reject a payload file whose content no longer matches the manifest."""
        detail = f"expected {expected}, found {actual}"
        return cls(f"content hash mismatch for {relative}: {detail}")

    @classmethod
    def dirty_destination(cls, dest: Path) -> BundleError:
        """Reject restoring over an existing tree."""
        return cls(f"destination is not empty: {dest}")

    @classmethod
    def bad_ledger_line(cls, ledger: Path, number: int) -> BundleError:
        """Reject a ledger line that is not valid JSON."""
        return cls(f"{ledger}:{number}: invalid JSON")

    @classmethod
    def missing_argument(cls, name: str) -> BundleError:
        """Reject a CLI namespace lacking an expected path argument."""
        return cls(f"missing path argument: {name}")


@dataclass(frozen=True, slots=True)
class BudgetAudit:
    """Cumulative OpenRouter spend read from an append-only ledger."""

    total_usd: float
    records: int
    cap_usd: float

    @property
    def under_cap(self) -> bool:
        """Report whether cumulative spend is strictly below the cap."""
        return self.total_usd < self.cap_usd


def _hash_file(path: Path) -> str:
    """Return the SHA-256 of a file, streamed so large weights stay cheap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_files(root: Path) -> list[Path]:
    """List files under a root in deterministic order."""
    return sorted(p for p in root.rglob("*") if p.is_file())


def build_bundle(src: Path, dest: Path) -> dict[str, JsonValue]:
    """Copy a source tree into a bundle and record a hashed manifest.

    Args:
        src: Directory holding model config, tokenizer metadata, and weights.
        dest: Bundle directory to create.

    Returns:
        The manifest written alongside the payload.

    Raises:
        BundleError: If the source directory is missing or empty.
    """
    if not src.is_dir():
        raise BundleError.missing_source(src)
    payload_src = _payload_files(src)
    if not payload_src:
        raise BundleError.empty_source(src)

    payload_dest = dest / PAYLOAD_DIR
    payload_dest.mkdir(parents=True, exist_ok=True)
    entries: list[JsonValue] = []
    for path in payload_src:
        relative = path.relative_to(src)
        target = payload_dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copy2(path, target)
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": _hash_file(target),
                "bytes": target.stat().st_size,
            }
        )

    manifest: dict[str, JsonValue] = {
        "bundle_version": BUNDLE_VERSION,
        "files": entries,
        "file_count": len(entries),
    }
    _ = (dest / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_manifest(bundle: Path) -> dict[str, JsonValue]:
    """Load and structurally validate a bundle manifest."""
    manifest_path = bundle / MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleError.missing_manifest(bundle)
    try:
        raw = cast("JsonValue", json.loads(manifest_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise BundleError.unreadable_manifest(manifest_path) from error
    if not isinstance(raw, dict):
        raise BundleError.malformed_manifest(
            manifest_path, "manifest must be a JSON object"
        )
    version = raw.get("bundle_version")
    if version != BUNDLE_VERSION:
        raise BundleError.unsupported_version(manifest_path, version)
    if not isinstance(raw.get("files"), list):
        raise BundleError.malformed_manifest(
            manifest_path, "manifest 'files' must be a list"
        )
    return raw


def _entry_fields(entry: JsonValue, manifest_path: Path) -> tuple[str, str]:
    """Extract and validate one manifest entry."""
    if not isinstance(entry, dict):
        raise BundleError.malformed_manifest(
            manifest_path, "manifest entry must be an object"
        )
    path = entry.get("path")
    digest = entry.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise BundleError.malformed_manifest(
            manifest_path, "manifest entry missing path or sha256"
        )
    return path, digest


def load_bundle(bundle: Path, dest: Path) -> dict[str, JsonValue]:
    """Verify a bundle against its manifest and restore it into a clean tree.

    Args:
        bundle: Directory produced by :func:`build_bundle`.
        dest: Destination directory; it must not already contain files.

    Returns:
        The verified manifest.

    Raises:
        BundleError: If the manifest is absent or malformed, a listed file is
            missing, a file's content no longer matches its recorded hash, or
            the destination already holds files.
    """
    manifest = _read_manifest(bundle)
    manifest_path = bundle / MANIFEST_NAME
    payload = bundle / PAYLOAD_DIR
    if not payload.is_dir():
        raise BundleError.missing_payload(bundle)
    if dest.exists() and _payload_files(dest):
        raise BundleError.dirty_destination(dest)

    files = manifest["files"]
    if not isinstance(files, list):  # pragma: no cover - guarded in _read_manifest
        raise BundleError.malformed_manifest(
            manifest_path, "manifest 'files' must be a list"
        )

    verified: list[tuple[Path, Path]] = []
    for entry in files:
        relative, expected = _entry_fields(entry, manifest_path)
        source = payload / relative
        if not source.is_file():
            raise BundleError.missing_file(relative)
        actual = _hash_file(source)
        if actual != expected:
            raise BundleError.hash_mismatch(relative, expected, actual)
        verified.append((source, dest / relative))

    for source, target in verified:
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copy2(source, target)
    return manifest


def audit_budget(ledger: Path, cap_usd: float = BUDGET_CAP_USD) -> BudgetAudit:
    """Sum cumulative USD spend from an append-only OpenRouter ledger.

    A missing ledger reports zero spend rather than raising: budget auditing
    must work before any paid call has been made.

    Args:
        ledger: Path to the append-only JSONL ledger.
        cap_usd: Hard cap the total is compared against.

    Returns:
        The cumulative audit.

    Raises:
        BundleError: If a ledger line is not valid JSON.
    """
    if not ledger.is_file():
        return BudgetAudit(total_usd=0.0, records=0, cap_usd=cap_usd)
    total = 0.0
    records = 0
    lines = ledger.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = cast("JsonValue", json.loads(line))
        except json.JSONDecodeError as error:
            raise BundleError.bad_ledger_line(ledger, number) from error
        if not isinstance(row, dict):
            continue
        cost = row.get("cost_usd")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            continue
        total += float(cost)
        records += 1
    return BudgetAudit(total_usd=total, records=records, cap_usd=cap_usd)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the release CLI parser."""
    parser = argparse.ArgumentParser(prog="kojev.release")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="package a source tree into a bundle")
    _ = build.add_argument("src", type=Path)
    _ = build.add_argument("dest", type=Path)
    restore = sub.add_parser("restore", help="verify and restore a bundle")
    _ = restore.add_argument("bundle", type=Path)
    _ = restore.add_argument("dest", type=Path)
    audit = sub.add_parser("audit", help="sum ledger spend")
    _ = audit.add_argument("ledger", type=Path)
    return parser


def _arg_path(values: dict[str, object], name: str) -> Path:
    """Read one Path argument without leaking an untyped namespace."""
    value = values.get(name)
    if not isinstance(value, Path):
        raise BundleError.missing_argument(name)
    return value


def main(argv: list[str] | None = None) -> int:
    """Run the release CLI."""
    values = cast("dict[str, object]", vars(_build_parser().parse_args(argv)))
    command = values.get("command")
    payload: dict[str, JsonValue]
    if command == "build":
        dest = _arg_path(values, "dest")
        manifest = build_bundle(_arg_path(values, "src"), dest)
        payload = {"built": str(dest), "files": manifest["file_count"]}
    elif command == "restore":
        dest = _arg_path(values, "dest")
        manifest = load_bundle(_arg_path(values, "bundle"), dest)
        payload = {"restored": str(dest), "files": manifest["file_count"]}
    else:
        audit = audit_budget(_arg_path(values, "ledger"))
        payload = {
            "total_usd": audit.total_usd,
            "records": audit.records,
            "under_cap": audit.under_cap,
        }
    _ = sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
