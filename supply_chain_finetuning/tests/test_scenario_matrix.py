# AI-ASSISTED: Cline (Claude Sonnet 5.5), Prompt: 'scenario matrix verification: 540 tuples, stratified deterministic sample, severity bands', Date: 2026-10-06
"""Offline verification for the scenario matrix (pure functions, no network)."""
from __future__ import annotations

from collections import Counter

from src import config
from src.scenario_matrix import build_matrix, sample_tuples


def test_matrix_is_full_cross_product():
    matrix = build_matrix()
    assert len(matrix) == config.MATRIX_EXPECTED_SIZE == 540
    combos = {(t.family, t.product_category, t.region, t.severity_level) for t in matrix}
    assert len(combos) == 540  # every tuple is unique


def test_family_classes_and_control_flags():
    matrix = build_matrix()
    for tup in matrix:
        assert tup.anomaly_class == config.FAMILY_TO_CLASS[tup.family]
        assert tup.is_control == (tup.family in config.CONTROL_FAMILIES)
        if tup.is_control:
            assert tup.severity_band == (config.CONTROL_SEVERITY, config.CONTROL_SEVERITY)


def test_sample_is_stratified_and_deterministic():
    first = sample_tuples(config.SAMPLE_OVERGENERATE, seed=config.SEED)
    second = sample_tuples(config.SAMPLE_OVERGENERATE, seed=config.SEED)
    assert [t.tuple_id for t in first] == [t.tuple_id for t in second]  # deterministic

    counts = Counter(t.family for t in first)
    assert len(counts) == len(config.SCENARIO_FAMILIES)          # every family present
    assert max(counts.values()) - min(counts.values()) <= 1      # round-robin balance
    assert all(14 <= n <= 15 for n in counts.values())           # ~14 per family
    assert len(first) == config.SAMPLE_OVERGENERATE
    assert len({t.tuple_id for t in first}) == len(first)        # no duplicate tuples


def test_severity_bands_respected_by_construction():
    for tup in build_matrix():
        low, high = tup.severity_band
        assert config.SEVERITY_MIN <= low <= high <= config.SEVERITY_MAX
