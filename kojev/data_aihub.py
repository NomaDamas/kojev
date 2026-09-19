"""Optional AI Hub ingestion lane: absent-directory stub.

This module is never imported by the default pipeline. It exists so the
optional AI Hub acquisition lane (see ``docs/aihub.md``) has a stable,
typed entry point before any download has happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from kojev.schema import Example

AIHUB_DATA_DIR: Final = Path("/data2/jeffrey/kojev/data/aihub")

_DOC_POINTER: Final = "docs/aihub.md"


def load_aihub_examples(root: Path = AIHUB_DATA_DIR) -> list[Example]:
    """Load typed-decision examples from a downloaded AI Hub directory.

    Args:
        root: Local directory holding the AI Hub datasets the user
            downloaded by hand, following ``docs/aihub.md``.

    Returns:
        Parsed typed-decision examples.

    Raises:
        NotImplementedError: Always, until the datasets are downloaded and
            a real parser lands. The message points to ``docs/aihub.md``.
    """
    if not root.is_dir():
        msg = (
            f"AI Hub directory not found: {root}. "
            f"Follow the five acquisition steps in {_DOC_POINTER} "
            "to download the datasets manually; this lane is optional and "
            "never runs automatically."
        )
        raise NotImplementedError(msg)
    msg = (
        f"AI Hub parsing is not implemented yet: {root}. "
        f"See {_DOC_POINTER} for the typed-decision mappings."
    )
    raise NotImplementedError(msg)
