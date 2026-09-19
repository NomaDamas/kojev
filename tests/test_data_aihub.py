"""Absent-directory behavior of the optional AI Hub ingestion stub."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from kojev.data_aihub import AIHUB_DATA_DIR, load_aihub_examples

if TYPE_CHECKING:
    from pathlib import Path


def test_load_raises_not_implemented_when_directory_missing(tmp_path: Path) -> None:
    """Given a missing directory, When loading, Then the error points to the doc."""
    missing = tmp_path / "no-such-dir"

    with pytest.raises(NotImplementedError, match=re.escape("docs/aihub.md")):
        _ = load_aihub_examples(missing)


def test_load_raises_not_implemented_when_path_is_a_file(tmp_path: Path) -> None:
    """Given a file in place of the directory, When loading, Then the same error."""
    not_a_dir = tmp_path / "aihub"
    _ = not_a_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(NotImplementedError, match=re.escape("docs/aihub.md")):
        _ = load_aihub_examples(not_a_dir)


def test_default_dir_is_outside_repo_and_error_mentions_doc() -> None:
    """Given the default path absent, When loading, Then the doc pointer survives."""
    if AIHUB_DATA_DIR.is_dir():
        pytest.skip("default AI Hub directory exists on this machine")

    with pytest.raises(NotImplementedError) as excinfo:
        _ = load_aihub_examples()

    message = str(excinfo.value)
    assert "docs/aihub.md" in message
    assert str(AIHUB_DATA_DIR) in message
