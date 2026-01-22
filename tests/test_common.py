"""Common tests for both IUPred2 and AIUPred prediction methods.

These tests are parametrized to run with both predictors, testing
common functionality like output format, value ranges, and biological
patterns in p53.
"""

import math

import pytest

from .conftest import P53_TEST_POSITIONS


class TestDisorderPredictionBasics:
    """Basic functionality tests for disorder prediction."""

    def test_output_length_matches_input(
        self, short_sequence, iupred_predictor
    ):
        """Output length should match input sequence length."""
        scores = iupred_predictor.predict_disorder(short_sequence)
        assert len(scores) == len(short_sequence)

    def test_scores_in_valid_range(self, p53_sequence, iupred_predictor):
        """All scores should be between 0 and 1."""
        scores = iupred_predictor.predict_disorder(p53_sequence)
        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_no_nan_values(self, p53_sequence, iupred_predictor):
        """Scores should not contain NaN values."""
        scores = iupred_predictor.predict_disorder(p53_sequence)
        assert not any(math.isnan(float(s)) for s in scores)

    def test_no_inf_values(self, p53_sequence, iupred_predictor):
        """Scores should not contain infinite values."""
        scores = iupred_predictor.predict_disorder(p53_sequence)
        assert not any(math.isinf(float(s)) for s in scores)


class TestBindingPredictionBasics:
    """Basic functionality tests for binding prediction."""

    def test_output_length_matches_input(
        self, short_sequence, iupred_predictor
    ):
        """Output length should match input sequence length."""
        scores = iupred_predictor.predict_binding(short_sequence)
        assert len(scores) == len(short_sequence)

    def test_scores_in_valid_range(self, p53_sequence, iupred_predictor):
        """All binding scores should be between 0 and 1."""
        scores = iupred_predictor.predict_binding(p53_sequence)
        for score in scores:
            assert 0.0 <= score <= 1.0


class TestP53BiologicalPatterns:
    """Tests for expected biological patterns in p53.

    p53 has well-characterized regions:
    - N-terminal (1-50): disordered transactivation domain
    - DBD (100-290): ordered DNA-binding domain
    """

    def test_nterminal_is_disordered(self, p53_sequence, iupred_predictor):
        """p53 N-terminal region should show high disorder."""
        scores = iupred_predictor.predict_disorder(p53_sequence)
        nterminal_scores = scores[:50]
        mean_disorder = sum(float(s) for s in nterminal_scores) / len(
            nterminal_scores
        )
        assert mean_disorder > 0.5, (
            f'N-terminal mean disorder {mean_disorder:.3f} should be > 0.5'
        )

    def test_dbd_is_more_ordered(self, p53_sequence, iupred_predictor):
        """p53 DBD should be more ordered than N-terminus."""
        scores = iupred_predictor.predict_disorder(p53_sequence)
        nterminal_mean = sum(float(s) for s in scores[:50]) / 50
        dbd_mean = sum(float(s) for s in scores[100:290]) / 190
        assert nterminal_mean > dbd_mean, (
            f'N-terminal ({nterminal_mean:.3f}) should be more disordered '
            f'than DBD ({dbd_mean:.3f})'
        )

    def test_disordered_vs_ordered_residues(
        self, p53_sequence, iupred_predictor
    ):
        """Specific disordered residues should score higher than ordered ones."""
        scores = iupred_predictor.predict_disorder(p53_sequence)

        disordered_scores = []
        ordered_scores = []

        for pos, aa, region in P53_TEST_POSITIONS:
            assert p53_sequence[pos] == aa, f'Position {pos} should be {aa}'
            if region == 'disordered':
                disordered_scores.append(float(scores[pos]))
            else:
                ordered_scores.append(float(scores[pos]))

        mean_disordered = sum(disordered_scores) / len(disordered_scores)
        mean_ordered = sum(ordered_scores) / len(ordered_scores)

        assert mean_disordered > mean_ordered, (
            f'Mean disordered score ({mean_disordered:.3f}) should be > '
            f'mean ordered score ({mean_ordered:.3f})'
        )


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_sequence_with_unknown_amino_acid(
        self, sequence_with_x, iupred_predictor
    ):
        """Sequences with X (unknown) should be handled."""
        scores = iupred_predictor.predict_disorder(sequence_with_x)
        assert len(scores) == len(sequence_with_x)
        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_very_short_sequence(self, iupred_predictor):
        """Very short sequences should be handled."""
        scores = iupred_predictor.predict_disorder('MEEP')
        assert len(scores) == 4

    def test_single_amino_acid(self, iupred_predictor):
        """Single amino acid should be handled."""
        scores = iupred_predictor.predict_disorder('M')
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_all_same_amino_acid(self, iupred_predictor):
        """Sequence of identical amino acids should be handled."""
        scores = iupred_predictor.predict_disorder('AAAAAAAAAA')
        assert len(scores) == 10


# AIUPred-specific versions of the same tests
# These use the aiupred_predictor fixture and are marked to require models


@pytest.mark.aiupred
class TestAiupredDisorderPredictionBasics:
    """Basic functionality tests for AIUPred disorder prediction."""

    def test_output_length_matches_input(
        self, short_sequence, aiupred_predictor
    ):
        """Output length should match input sequence length."""
        scores = aiupred_predictor.predict_disorder(short_sequence)
        assert len(scores) == len(short_sequence)

    def test_scores_in_valid_range(self, p53_sequence, aiupred_predictor):
        """All scores should be between 0 and 1."""
        scores = aiupred_predictor.predict_disorder(p53_sequence)
        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_no_nan_values(self, p53_sequence, aiupred_predictor):
        """Scores should not contain NaN values."""
        scores = aiupred_predictor.predict_disorder(p53_sequence)
        assert not any(math.isnan(float(s)) for s in scores)

    def test_returns_numpy_array(self, short_sequence, aiupred_predictor):
        """AIUPred should return numpy array."""
        import numpy as np

        scores = aiupred_predictor.predict_disorder(short_sequence)
        assert isinstance(scores, np.ndarray)


@pytest.mark.aiupred
class TestAiupredBindingPredictionBasics:
    """Basic functionality tests for AIUPred binding prediction."""

    def test_output_length_matches_input(
        self, short_sequence, aiupred_predictor
    ):
        """Output length should match input sequence length."""
        scores = aiupred_predictor.predict_binding(short_sequence)
        assert len(scores) == len(short_sequence)

    def test_scores_in_valid_range(self, p53_sequence, aiupred_predictor):
        """All binding scores should be between 0 and 1."""
        scores = aiupred_predictor.predict_binding(p53_sequence)
        for score in scores:
            assert 0.0 <= score <= 1.0


@pytest.mark.aiupred
class TestAiupredP53BiologicalPatterns:
    """Tests for expected biological patterns in p53 using AIUPred."""

    def test_nterminal_is_disordered(self, p53_sequence, aiupred_predictor):
        """p53 N-terminal region should show high disorder."""
        scores = aiupred_predictor.predict_disorder(p53_sequence)
        nterminal_scores = scores[:50]
        mean_disorder = sum(float(s) for s in nterminal_scores) / len(
            nterminal_scores
        )
        assert mean_disorder > 0.5

    def test_dbd_is_more_ordered(self, p53_sequence, aiupred_predictor):
        """p53 DBD should be more ordered than N-terminus."""
        scores = aiupred_predictor.predict_disorder(p53_sequence)
        nterminal_mean = sum(float(s) for s in scores[:50]) / 50
        dbd_mean = sum(float(s) for s in scores[100:290]) / 190
        assert nterminal_mean > dbd_mean

    def test_disordered_vs_ordered_residues(
        self, p53_sequence, aiupred_predictor
    ):
        """Specific disordered residues should score higher than ordered ones."""
        scores = aiupred_predictor.predict_disorder(p53_sequence)

        disordered_scores = []
        ordered_scores = []

        for pos, _aa, region in P53_TEST_POSITIONS:
            if region == 'disordered':
                disordered_scores.append(float(scores[pos]))
            else:
                ordered_scores.append(float(scores[pos]))

        mean_disordered = sum(disordered_scores) / len(disordered_scores)
        mean_ordered = sum(ordered_scores) / len(ordered_scores)

        assert mean_disordered > mean_ordered


@pytest.mark.aiupred
class TestAiupredEdgeCases:
    """Tests for edge cases with AIUPred."""

    def test_sequence_with_unknown_amino_acid(
        self, sequence_with_x, aiupred_predictor
    ):
        """Sequences with X should be handled."""
        scores = aiupred_predictor.predict_disorder(sequence_with_x)
        assert len(scores) == len(sequence_with_x)

    def test_very_short_sequence(self, aiupred_predictor):
        """Very short sequences should be handled."""
        scores = aiupred_predictor.predict_disorder('MEEP')
        assert len(scores) == 4

    def test_consistency(self, short_sequence, aiupred_predictor):
        """Same input should produce same output."""
        scores1 = aiupred_predictor.predict_disorder(short_sequence)
        scores2 = aiupred_predictor.predict_disorder(short_sequence)
        for s1, s2 in zip(scores1, scores2):
            assert abs(s1 - s2) < 1e-6
