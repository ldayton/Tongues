def _helper() -> int:
    return 2


def wrapper() -> int:
    return _helper()
