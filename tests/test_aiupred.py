"""AIUPred-specific tests.

Tests for features specific to AIUPred that don't apply to IUPred2.
These tests require model download (~82MB) and are skipped by default.
Run with: pytest --run-aiupred
"""

import pytest

from iupred import aiupred_binding, aiupred_disorder
from .conftest import P53_SEQUENCE


pytestmark = pytest.mark.aiupred


class TestAiupredRegressionValues:
    """Regression tests with expected values for p53.

    These test specific positions in both disordered and ordered regions.
    Values calibrated with smoothing=True (default).
    """

    # Expected disorder values for p53 at specific positions
    # Format: (position, amino_acid, expected_value, tolerance)
    EXPECTED_DISORDER_VALUES = [
        # N-terminal (disordered region) - high values with smoothing
        (0, 'M', 0.95, 0.10),
        (4, 'Q', 0.96, 0.10),
        (9, 'V', 0.97, 0.10),
        # DBD region (ordered) - should be noticeably lower
        (150, 'P', 0.45, 0.30),
        (175, 'C', 0.35, 0.30),
        (200, 'R', 0.40, 0.30),
    ]

    # Expected binding values for p53 at specific positions
    EXPECTED_BINDING_VALUES = [
        # N-terminal positions
        (0, 'M', 0.87, 0.15),
        (4, 'Q', 0.30, 0.15),
        (9, 'V', 0.40, 0.20),
        # DBD region
        (150, 'P', 0.15, 0.20),
        (175, 'C', 0.10, 0.20),
        (200, 'R', 0.10, 0.20),
    ]

    def test_p53_disorder_specific_positions(self, aiupred_models):
        """Test disorder values at specific positions in p53."""
        scores = aiupred_disorder(P53_SEQUENCE, force_cpu=True)

        for pos, expected_aa, expected_value, tolerance in self.EXPECTED_DISORDER_VALUES:
            actual_aa = P53_SEQUENCE[pos]
            assert actual_aa == expected_aa, (
                f'Position {pos}: expected {expected_aa}, found {actual_aa}'
            )
            actual_value = float(scores[pos])
            assert abs(actual_value - expected_value) < tolerance, (
                f'Disorder at position {pos} ({actual_aa}): '
                f'expected ~{expected_value}, got {actual_value:.3f}'
            )

    def test_p53_binding_specific_positions(self, aiupred_models):
        """Test binding values at specific positions in p53."""
        scores = aiupred_binding(P53_SEQUENCE, force_cpu=True)

        for pos, expected_aa, expected_value, tolerance in self.EXPECTED_BINDING_VALUES:
            actual_aa = P53_SEQUENCE[pos]
            assert actual_aa == expected_aa
            actual_value = float(scores[pos])
            assert abs(actual_value - expected_value) < tolerance, (
                f'Binding at position {pos} ({actual_aa}): '
                f'expected ~{expected_value}, got {actual_value:.3f}'
            )


class TestAiupredDisorderVsBinding:
    """Tests comparing disorder and binding predictions."""

    def test_disorder_and_binding_differ(self, aiupred_models, p53_sequence):
        """Disorder and binding predictions should be different."""
        disorder_scores = aiupred_disorder(p53_sequence, force_cpu=True)
        binding_scores = aiupred_binding(p53_sequence, force_cpu=True)

        # Calculate mean absolute difference
        differences = [
            abs(float(d) - float(b))
            for d, b in zip(disorder_scores, binding_scores, strict=True)
        ]
        mean_diff = sum(differences) / len(differences)

        assert mean_diff > 0.1, (
            f'Disorder and binding should differ substantially '
            f'(mean diff: {mean_diff:.3f})'
        )
