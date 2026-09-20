from collections.abc import Mapping
from pathlib import Path

from torch import Tensor

def save_file(
    tensors: Mapping[str, Tensor],
    filename: str | Path,
    metadata: Mapping[str, str] | None = ...,
) -> None: ...
def load_file(
    filename: str | Path,
    device: str | int = ...,
) -> dict[str, Tensor]: ...
