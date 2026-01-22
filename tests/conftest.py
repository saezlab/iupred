"""Pytest configuration and fixtures for iupred tests."""

import pytest


# p53 (P04637) sequence - well-characterized protein with disordered regions
P53_SEQUENCE = (
    'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP'
    'DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK'
    'SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHER'
    'CSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSS'
    'CMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPP'
    'GSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGG'
    'SRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD'
)

# Short test sequence
SHORT_SEQUENCE = 'MEEPQSDPSVEPPLSQETFS'

# Sequence with unknown amino acid
SEQUENCE_WITH_X = 'MEEPXSDPSVEPPLSQETFS'

# p53 regions for testing
# N-terminal transactivation domain (disordered)
P53_NTERMINAL_REGION = (1, 50)
# DNA-binding domain (ordered/structured)
P53_DBD_REGION = (100, 290)

# Selected residue positions for regression tests
# Format: (position_0_indexed, amino_acid, region_type)
P53_TEST_POSITIONS = [
    # N-terminal (disordered) - positions 0-9
    (0, 'M', 'disordered'),
    (4, 'Q', 'disordered'),
    (9, 'V', 'disordered'),
    # DBD region (ordered) - positions around 150-200
    (150, 'P', 'ordered'),
    (175, 'C', 'ordered'),
    (200, 'R', 'ordered'),
]


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        '--run-aiupred',
        action='store_true',
        default=False,
        help='Run AIUPred tests (requires ~82MB model download)',
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        'markers',
        'aiupred: mark test as requiring AIUPred models (skipped unless --run-aiupred)',
    )


def pytest_collection_modifyitems(config, items):
    """Skip AIUPred tests unless --run-aiupred is specified."""
    if config.getoption('--run-aiupred'):
        return

    skip_aiupred = pytest.mark.skip(
        reason='AIUPred tests require --run-aiupred option'
    )
    for item in items:
        if 'aiupred' in item.keywords:
            item.add_marker(skip_aiupred)


@pytest.fixture(scope='session')
def p53_sequence():
    """Return the p53 protein sequence."""
    return P53_SEQUENCE


@pytest.fixture(scope='session')
def short_sequence():
    """Return a short test sequence."""
    return SHORT_SEQUENCE


@pytest.fixture(scope='session')
def sequence_with_x():
    """Return a sequence containing unknown amino acid X."""
    return SEQUENCE_WITH_X


@pytest.fixture(scope='session')
def aiupred_models():
    """Ensure AIUPred models are downloaded and return cache directory.

    This fixture downloads the models if not already present.
    Only used by tests marked with @pytest.mark.aiupred.
    """
    from iupred import ensure_aiupred_data

    cache_dir = ensure_aiupred_data()
    return cache_dir


# Prediction function wrappers for unified testing


class IupredPredictor:
    """Wrapper for IUPred2 predictions."""

    name = 'iupred2'
    requires_models = False

    def predict_disorder(self, sequence):
        from iupred import iupred

        scores, _ = iupred(sequence, mode='long')
        return scores

    def predict_binding(self, sequence):
        from iupred import iupred, anchor2

        iupred_scores, _ = iupred(sequence, mode='long')
        return anchor2(sequence, iupred_scores)


class AiupredPredictor:
    """Wrapper for AIUPred predictions."""

    name = 'aiupred'
    requires_models = True

    def predict_disorder(self, sequence):
        from iupred import aiupred_disorder

        return aiupred_disorder(sequence, force_cpu=True)

    def predict_binding(self, sequence):
        from iupred import aiupred_binding

        return aiupred_binding(sequence, force_cpu=True)


@pytest.fixture
def iupred_predictor():
    """Return IUPred2 predictor wrapper."""
    return IupredPredictor()


@pytest.fixture
def aiupred_predictor(aiupred_models):
    """Return AIUPred predictor wrapper (requires models)."""
    return AiupredPredictor()
