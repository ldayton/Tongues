from . import parse


def make_token(v: str) -> parse.Token:
    return parse.Token(v)
