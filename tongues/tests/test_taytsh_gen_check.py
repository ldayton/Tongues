"""Generative tests for the Taytsh type checker."""

from __future__ import annotations

import pytest

from src.taytsh.emit import to_source

from tests.taytsh.check_gen import Generator
from tests.taytsh.check_gen.exhaust import ALL_CONFIGS, run_exhaustiveness
from tests.taytsh.check_gen.features import FeatureVector
from tests.taytsh.check_gen.harness import run_mutations, run_well_typed
from tests.taytsh.check_gen.narrow import ALL_SPECS, run_narrowing_spec


class TestWellTyped:
    @pytest.mark.parametrize("seed", range(500))
    def test_accepted(self, seed: int) -> None:
        features = FeatureVector.random(__import__("random").Random(seed))
        errors = run_well_typed(features, seed)
        if errors:
            # Emit source for debugging
            gen = Generator(features, seed)
            module = gen.generate()
            try:
                src = to_source(module)
            except Exception:
                src = "<emit failed>"
            msg = f"Seed {seed} produced {len(errors)} error(s):\n"
            for e in errors:
                msg += f"  - {e}\n"
            msg += f"\nSource:\n{src}"
            pytest.fail(msg)


class TestMutations:
    @pytest.mark.parametrize("seed", range(500))
    def test_detected(self, seed: int) -> None:
        features = FeatureVector.random(__import__("random").Random(seed))
        failures = run_mutations(features, seed)
        if failures:
            msgs: list[str] = []
            for f in failures:
                msgs.append(
                    f"  Mutation '{f.mutation_name}': expected '{f.expected_error}', "
                    f"got {f.actual_errors}"
                )
            pytest.fail(
                f"Seed {seed}: {len(failures)} undetected mutation(s):\n"
                + "\n".join(msgs)
            )


class TestNarrowing:
    @pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
    def test_narrowing(self, spec) -> None:
        failures = run_narrowing_spec(spec)
        if not failures:
            return
        msgs: list[str] = []
        for f in failures:
            label = "should accept" if f.expected_clean else "should reject"
            msgs.append(f"  {f.case} ({label}): {f.actual_errors}")
        pytest.fail(
            f"Spec '{spec.name}': {len(failures)} failure(s):\n" + "\n".join(msgs)
        )


class TestExhaustiveness:
    @pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
    def test_exhaustiveness(self, config) -> None:
        failures = run_exhaustiveness(config)
        if failures:
            msgs: list[str] = []
            for f in failures:
                msgs.append(
                    f"  Cases {f.cases}, default={f.with_default}: "
                    f"expected_accept={f.expected_accept}, errors={f.actual_errors}"
                )
            pytest.fail(
                f"Config '{config.name}': {len(failures)} failure(s):\n"
                + "\n".join(msgs)
            )
