def _helper() -> int:
    return 1


def wrapper() -> int:
    return _helper()
