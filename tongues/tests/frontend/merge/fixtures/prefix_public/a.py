class Token:
    def __init__(self, value: str) -> None:
        self.value: str = value


def make_a() -> Token:
    return Token("a")
