# IUPred

[![GitHub release](https://img.shields.io/github/v/release/saezlab/iupred)](https://github.com/saezlab/iupred/releases/latest)
[![Tests](https://github.com/saezlab/iupred/actions/workflows/ci-testing-unit.yml/badge.svg)](https://github.com/saezlab/iupred/actions/workflows/ci-testing-unit.yml)
[![Documentation](https://github.com/saezlab/iupred/actions/workflows/ci-docs.yml/badge.svg)](https://github.com/saezlab/iupred/actions/workflows/ci-docs.yml)

Prediction of intrinsically disordered protein regions using IUPred2/ANCHOR2
and AIUPred methods.

**[Documentation](https://saezlab.github.io/iupred/)** | **[GitHub](https://github.com/saezlab/iupred)** | **[IUPred2A Web Server](https://iupred2a.elte.hu/)** | **[AIUPred Web Server](https://aiupred.elte.hu/)**

## Description

Intrinsically disordered proteins (IDPs) lack a single well-defined tertiary
structure under native conditions. This package provides methods for predicting
disordered regions in protein sequences:

- **IUPred2** - Energy-based prediction of intrinsically disordered regions
- **ANCHOR2** - Prediction of disordered binding regions where unstructured
  proteins interact with partners
- **AIUPred** - Deep learning approach combining energy estimation with
  transformer networks for enhanced disorder prediction

## Authors

- Zsuzsanna Dosztanyi (ELTE, Budapest)
- Gabor Erdos (ELTE, Budapest)

### Additional Contributors

- Balint Meszaros (IUPred2A)
- Akos Deutsch (AIUPred-binding)
- Denes Turei (modern tooling and packaging)

## Requirements

- Python >= 3.9
- numpy >= 1.20
- scipy >= 1.7
- torch >= 1.9 (for AIUPred)

## Installation

### Using pip (from GitHub)

```bash
pip install git+https://github.com/saezlab/iupred.git
```

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project with iupred
uv init my-project
cd my-project
uv add git+https://github.com/saezlab/iupred.git

# Or add to an existing project
uv add git+https://github.com/saezlab/iupred.git
```

### Development installation

```bash
git clone https://github.com/saezlab/iupred.git
cd iupred

# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[tests]"

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[tests]"
```

## Usage

### IUPred2 (bundled data, works out of the box)

IUPred2 predicts intrinsically disordered regions based on estimated
pairwise interaction energies.

```python
from iupred import iupred, anchor2

sequence = "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG..."

# Predict disorder using IUPred2
# Returns tuple of (scores, glob_text)
disorder_scores, glob_text = iupred(sequence, mode='long')

# Predict binding regions using ANCHOR2
# Requires IUPred scores as input
anchor_scores = anchor2(sequence, disorder_scores)
```

**Available modes for `iupred()`:**

- `'short'` - Short disorder prediction (default), optimized for shorter
  disordered segments
- `'long'` - Long disorder prediction, identifies extended disordered regions
- `'glob'` - Globular domain prediction, locates stable folded regions

### AIUPred (automatic model download)

AIUPred uses transformer-based deep learning for enhanced disorder prediction.
Model weights (~82 MB) are automatically downloaded on first use.

```python
from iupred import aiupred_disorder, aiupred_binding

sequence = "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG..."

# Predict disorder using AIUPred
# Models are downloaded automatically on first call
disorder_scores = aiupred_disorder(sequence)

# Predict binding regions
binding_scores = aiupred_binding(sequence)
```

Models are cached in `~/.cache/iupred/` and only downloaded once.

**Data management:**

```python
from iupred import ensure_aiupred_data, clear_aiupred_cache

# Pre-download models (optional, useful for offline use)
ensure_aiupred_data()

# Clear cached models to free disk space or force re-download
clear_aiupred_cache()
```

**Options:**

- `force_cpu=True` - Force CPU-only mode (default: use GPU if available)
- `gpu_num=0` - GPU index to use (default: 0)

### Performance (AIUPred)

| Hardware | Task | Speed |
|----------|------|-------|
| GPU (1080 Ti 12GB) | Single protein | ~3 seconds |
| GPU (1080 Ti 12GB) | Human proteome | ~100 proteins/second |
| CPU (Xeon E3-1270) | Single protein | ~1.7 seconds |
| CPU (Xeon E3-1270) | Human proteome | ~3.5 proteins/second |

**Memory requirements:** 2GB VRAM handles ~3,000 residues; 12GB handles ~16,000.

## References

If you use **IUPred2/ANCHOR2**, please cite:

> Meszaros B, Erdos G, Dosztanyi Z. IUPred2A: context-dependent prediction of
> protein disorder as a function of redox state and protein binding.
> *Nucleic Acids Res.* 2018;46(W1):W329-W337.
> https://doi.org/10.1093/nar/gky384

If you use **AIUPred**, please cite:

> Erdos G, Dosztanyi Z. AIUPred: combining energy estimation with deep learning
> for the enhanced prediction of protein disorder.
> *Nucleic Acids Res.* 2024;52(W1):W176-W181.
> https://doi.org/10.1093/nar/gkae385

If you use **AIUPred-binding**, please cite:

> Erdos G, Deutsch A, Dosztanyi Z. Identification of disordered binding regions
> using energy embedding.
> *J Mol Biol.* 2025;437(15):169071.
> https://doi.org/10.1016/j.jmb.2025.169071

## License

This software is distributed under the **CC-BY-NC-ND-4.0** license.
Commercial use and derivative works are not permitted without explicit
permission from the authors.

For commercial licensing inquiries, please contact:
Zsuzsanna Dosztanyi <zsuzsanna.dosztanyi@ttk.elte.hu>

## Development

### Running tests

```bash
# Install test dependencies
uv pip install -e ".[tests]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=iupred
```

### Linting and formatting

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Pre-commit hooks

[Pre-commit](https://pre-commit.com/) hooks are configured for automated checks:

```bash
# Install pre-commit hooks
uv pip install -e ".[dev]"
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files
```

### Building documentation

Documentation is built with [MkDocs](https://www.mkdocs.org/):

```bash
# Install docs dependencies
uv pip install -e ".[docs]"

# Serve docs locally
mkdocs serve

# Build static docs
mkdocs build
```

## Contact

For questions and support, please contact:
Zsuzsanna Dosztanyi <zsuzsanna.dosztanyi@ttk.elte.hu>

Feature requests and bug reports can be submitted via
[GitHub Issues](https://github.com/saezlab/iupred/issues).
