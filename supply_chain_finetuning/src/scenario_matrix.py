"""Scenario matrix: the full cross-product plus a deterministic stratified sample.

The full matrix is 12 families x 5 product categories x 3 regions x 3 severity
levels = 540 tuples. Sampling draws ~14 per family (round-robin over families)
so diversity is guaranteed by construction, and the seed makes the exact
selection reproducible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

from . import config

@dataclass(frozen=True)
class ScenarioTuple:
    """One cell of the scenario matrix."""

    family: str
    anomaly_class: str          # the class the teacher must produce ("none" for controls)
    product_category: str
    region: str
    severity_level: str         # mild | severe | critical
    severity_band: Tuple[int, int]
    is_control: bool

    @property
    def tuple_id(self) -> str:
        """Stable identifier used for resumability."""
        return f"{self.family}|{self.product_category}|{self.region}|{self.severity_level}"

def build_matrix() -> List[ScenarioTuple]:
    """Full cross-product: 12 families x 5 categories x 3 regions x 3 levels = 540."""
    tuples: List[ScenarioTuple] = []
    for family, category, region, level in product(
        config.SCENARIO_FAMILIES, config.PRODUCT_CATEGORIES, config.REGIONS,
        config.SEVERITY_LEVELS,
    ):
        is_control = family in config.CONTROL_FAMILIES
        tuples.append(
            ScenarioTuple(
                family=family,
                anomaly_class=config.FAMILY_TO_CLASS[family],
                product_category=category,
                region=region,
                severity_level=level,
                severity_band=(config.CONTROL_SEVERITY, config.CONTROL_SEVERITY)
                if is_control
                else config.SEVERITY_BANDS[level],
                is_control=is_control,
            )
        )
    return tuples

def sample_tuples(
    count: int = config.SAMPLE_OVERGENERATE, seed: int = config.SEED
) -> List[ScenarioTuple]:
    """Deterministic stratified sample: round-robin across families (~14 each).

    Each family's tuples are shuffled with the shared seed, then families are
    consumed round-robin until ``count`` tuples are drawn. With 12 families and
    170 draws every family contributes 14 or 15 tuples.
    """
    matrix = build_matrix()
    if count > len(matrix):
        raise ValueError(f"requested {count} tuples but the matrix has {len(matrix)}")
    by_family: Dict[str, List[ScenarioTuple]] = {}
    for tup in matrix:
        by_family.setdefault(tup.family, []).append(tup)
    rng = random.Random(seed)
    queues = {
        family: rng.sample(members, len(members))  # shuffle in place deterministically
        for family, members in by_family.items()
    }
    family_names = sorted(queues)
    picked: List[ScenarioTuple] = []
    while len(picked) < count:
        for family in family_names:
            if len(picked) >= count:
                break
            if queues[family]:
                picked.append(queues[family].pop())
    return picked
