from collections.abc import Mapping

class WordLevel:
    def __init__(self, vocab: Mapping[str, int], unk_token: str = ...) -> None: ...
