class Foo:
    def __init__(self, value: str) -> None:
        self.value: str = value


def lib_make() -> Foo:
    return Foo("lib")
