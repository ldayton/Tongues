"""FeatureVector — boolean flags for type checker features under test."""

from __future__ import annotations

from dataclasses import dataclass, fields
from random import Random


@dataclass
class FeatureVector:
    union_type: bool = False
    optional_type: bool = False
    nil_narrowing: bool = False
    match_interface: bool = False
    match_enum: bool = False
    match_union: bool = False
    match_optional: bool = False
    match_default: bool = False
    match_default_bind: bool = False
    try_catch_typed: bool = False
    try_catch_all: bool = False
    try_catch_union: bool = False
    try_finally: bool = False
    fn_literal: bool = False
    fn_value: bool = False
    higher_order: bool = False
    for_collection: bool = False
    for_range: bool = False
    for_two_vars: bool = False
    tuple_destructure: bool = False
    struct_method: bool = False
    compound_assign: bool = False
    nested_collection: bool = False
    union_field_access: bool = False
    bytes_ops: bool = False
    rune_ops: bool = False
    overloaded_builtin: bool = False

    def active(self) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name)]

    @staticmethod
    def random(rng: Random) -> FeatureVector:
        fv = FeatureVector()
        for f in fields(fv):
            setattr(fv, f.name, rng.random() < 0.4)
        return fv
