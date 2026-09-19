from collections.abc import Iterator, Mapping
from typing import Protocol

type RowValue = (
    str
    | int
    | float
    | bool
    | list[str]
    | list[int]
    | Mapping[str, str | int | float]
)

class Dataset(Protocol):
    def __iter__(self) -> Iterator[Mapping[str, RowValue]]: ...

def load_dataset(path: str, *, name: str | None = ..., split: str) -> Dataset: ...
