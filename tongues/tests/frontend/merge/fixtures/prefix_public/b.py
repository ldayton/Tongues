class Token:
    def __init__(self, kind: str) -> None:
        self.kind: str = kind


def make_b() -> Token:
    return Token("b")
