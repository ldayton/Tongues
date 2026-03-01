from .lib import Foo as LibFoo


class Foo:
    def __init__(self, kind: str) -> None:
        self.kind: str = kind


def app_make() -> LibFoo:
    return LibFoo("app")
