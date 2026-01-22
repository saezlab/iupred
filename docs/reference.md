# API Reference

This page documents the public API of the `iupred` package.

## IUPred2 Functions

Energy-based prediction of intrinsically disordered regions.

::: iupred.iupred

::: iupred.anchor2

## AIUPred Functions

Transformer-based deep learning prediction of disordered regions.

### Simple API

For single sequence predictions, use these convenience functions:

::: iupred.aiupred_disorder

::: iupred.aiupred_binding

### Batch Processing API

For efficient processing of multiple sequences, initialize models once
and reuse them:

::: iupred.init_aiupred_models

::: iupred.predict_aiupred_disorder

::: iupred.predict_aiupred_binding

### Low-Memory API

For very long sequences or memory-constrained systems, use chunked processing:

::: iupred.low_memory_aiupred_disorder

::: iupred.low_memory_aiupred_binding

## Data Management

Functions for managing AIUPred model weights.

::: iupred.ensure_aiupred_data

::: iupred.clear_aiupred_cache

## Command-Line Interface

The package provides a command-line interface for running predictions
from the terminal:

```bash
# IUPred2 prediction
iupred iupred2 -i sequences.fasta -o results.txt -m long -a

# AIUPred prediction
iupred aiupred -i sequences.fasta -o results.txt -b --force-cpu
```

### IUPred2 CLI Options

| Option | Description |
|--------|-------------|
| `-i, --input-file` | Input FASTA file (required) |
| `-o, --output-file` | Output file (default: stdout) |
| `-m, --mode` | Prediction mode: short, long, or glob (default: long) |
| `-a, --anchor` | Also predict binding regions with ANCHOR2 |
| `-v, --verbose` | Enable verbose output |

### AIUPred CLI Options

| Option | Description |
|--------|-------------|
| `-i, --input-file` | Input FASTA file (required) |
| `-o, --output-file` | Output file (default: stdout) |
| `-b, --binding` | Also predict binding regions |
| `-g, --gpu` | GPU index to use (default: 0) |
| `--force-cpu` | Force CPU-only mode |
| `--low-memory` | Use low-memory mode for long sequences |
| `--chunk-size` | Chunk size for low-memory mode (default: 1000) |
| `-v, --verbose` | Enable verbose output |
