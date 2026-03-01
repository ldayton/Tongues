from .a import make_a

ASTNode = dict[str, object]


def make_b() -> ASTNode:
    x: ASTNode = make_a()
    return x
