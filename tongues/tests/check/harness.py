"""Harness — orchestration, shrinking, and coverage tracking."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from src.taytsh.ast import TModule
from src.taytsh.check import check
from src.taytsh.emit import to_source

from . import Generator
from .features import FeatureVector
from .mutator import ALL_MUTATIONS, MutationResult


@dataclass
class MutationFailure:
    seed: int
    mutation_name: str
    expected_error: str
    actual_errors: list[str]
    source: str


def run_well_typed(features: FeatureVector, seed: int) -> list[str]:
    gen = Generator(features, seed)
    module = gen.generate()
    errors = check(module)
    return [e.msg for e in errors]


def run_mutations(features: FeatureVector, seed: int) -> list[MutationFailure]:
    gen = Generator(features, seed)
    module = gen.generate()
    rng = Random(seed)

    failures: list[MutationFailure] = []
    for mutate_fn in ALL_MUTATIONS:
        result = mutate_fn(module, rng)
        if result is None:
            continue

        errors = check(result.module)
        error_msgs = [e.msg for e in errors]
        # If the mutation produced no errors at all, it was a no-op — skip
        if not error_msgs:
            continue
        found = any(result.expected_error in msg for msg in error_msgs)
        if not found:
            try:
                src = to_source(result.module)
            except Exception:
                src = "<emit failed>"
            failures.append(
                MutationFailure(
                    seed=seed,
                    mutation_name=result.name,
                    expected_error=result.expected_error,
                    actual_errors=error_msgs,
                    source=src,
                )
            )
    return failures
