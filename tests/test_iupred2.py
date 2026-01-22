"""IUPred2-specific tests.

Tests for features specific to IUPred2 that don't apply to AIUPred,
such as prediction modes (short, long, glob) and ANCHOR2 integration.
"""

from iupred import iupred, anchor2
from .conftest import P53_SEQUENCE


class TestIupredModes:
    """Tests for different IUPred2 prediction modes."""

    def test_iupred_returns_tuple(self, short_sequence):
        """iupred should return a tuple of (scores, glob_text)."""
        result = iupred(short_sequence)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_long_mode(self, short_sequence):
        """Long mode should work without errors."""
        scores, _ = iupred(short_sequence, mode='long')
        assert len(scores) == len(short_sequence)

    def test_short_mode(self, short_sequence):
        """Short mode should work without errors."""
        scores, _ = iupred(short_sequence, mode='short')
        assert len(scores) == len(short_sequence)

    def test_glob_mode_returns_glob_text(self, p53_sequence):
        """Glob mode should return non-empty glob_text for p53."""
        scores, glob_text = iupred(p53_sequence, mode='glob')
        assert len(scores) == len(p53_sequence)
        assert isinstance(glob_text, str)
        assert 'globular domain' in glob_text.lower()

    def test_different_modes_give_different_results(self, p53_sequence):
        """Different modes should produce different results."""
        scores_long, _ = iupred(p53_sequence, mode='long')
        scores_short, _ = iupred(p53_sequence, mode='short')

        # At least some scores should differ between modes
        differences = sum(
            1 for a, b in zip(scores_long, scores_short) if abs(a - b) > 0.01
        )
        assert differences > 0, (
            'Long and short modes should give different results'
        )


class TestAnchor2:
    """Tests specific to ANCHOR2 binding prediction."""

    def test_anchor2_requires_iupred_scores(self, short_sequence):
        """ANCHOR2 should work with IUPred scores as input."""
        iupred_scores, _ = iupred(short_sequence)
        anchor_scores = anchor2(short_sequence, iupred_scores)
        assert len(anchor_scores) == len(short_sequence)

    def test_anchor2_returns_list(self, short_sequence):
        """ANCHOR2 should return a list."""
        iupred_scores, _ = iupred(short_sequence)
        anchor_scores = anchor2(short_sequence, iupred_scores)
        assert isinstance(anchor_scores, list)


class TestIupredRegressionValues:
    """Regression tests with expected values for p53.

    These test specific positions in both disordered and ordered regions.
    """

    # Expected disorder values for p53 at specific positions (long mode)
    # Format: (position, amino_acid, expected_value, tolerance)
    EXPECTED_VALUES = [
        # N-terminal (disordered region) - should be high (~0.97-0.99)
        (0, 'M', 0.98, 0.05),
        (4, 'Q', 0.97, 0.05),
        (9, 'V', 0.97, 0.05),
        # DBD region (more ordered) - should be lower than N-terminal
        (150, 'P', 0.43, 0.15),
        (175, 'C', 0.48, 0.15),
        (200, 'R', 0.32, 0.15),
    ]

    def test_p53_specific_positions(self):
        """Test disorder values at specific positions in p53."""
        scores, _ = iupred(P53_SEQUENCE, mode='long')

        for pos, expected_aa, expected_value, tolerance in self.EXPECTED_VALUES:
            actual_aa = P53_SEQUENCE[pos]
            assert actual_aa == expected_aa, (
                f'Position {pos}: expected {expected_aa}, found {actual_aa}'
            )
            actual_value = scores[pos]
            assert abs(actual_value - expected_value) < tolerance, (
                f'Position {pos} ({actual_aa}): expected ~{expected_value}, '
                f'got {actual_value:.3f}'
            )


class TestProlineRichSequences:
    """Tests for proline-rich sequences (typically disordered)."""

    def test_proline_rich_is_disordered(self):
        """Proline-rich sequences should show high disorder."""
        proline_rich = 'PPPPEPPPPEPP'
        scores, _ = iupred(proline_rich)
        mean_score = sum(scores) / len(scores)
        assert mean_score > 0.5, (
            f'Proline-rich mean disorder {mean_score:.3f} should be > 0.5'
        )
