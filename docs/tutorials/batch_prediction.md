# Batch Prediction for Multiple Sequences

This tutorial demonstrates how to efficiently predict disorder for multiple
protein sequences by initializing models once and reusing them.

## Why Batch Processing?

When processing multiple sequences, the simple approach of calling
`aiupred_disorder()` for each sequence works but is inefficient because:

1. Models are loaded from disk for each sequence
2. Models are transferred to GPU for each sequence

For batch processing, we initialize models once and reuse them,
which is much faster for large datasets.

## Setup

```python
from iupred import (
    iupred,
    anchor2,
    init_aiupred_models,
    predict_aiupred_disorder,
    predict_aiupred_binding,
)
```

## Example Sequences

Let's create a small dataset of protein sequences to analyze:

```python
# Example sequences (various human proteins with known disordered regions)
sequences = {
    'p53_TAD': 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP',
    'p53_DBD': 'SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE',
    'alpha_synuclein': 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTK',
    'ubiquitin': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYN',
}

print(f"Number of sequences: {len(sequences)}")
for name, seq in sequences.items():
    print(f"  {name}: {len(seq)} residues")
```

## IUPred2 Batch Processing

IUPred2 is already fast since it doesn't require loading neural network models.
Simply iterate over sequences:

```python
iupred2_results = {}

for name, sequence in sequences.items():
    disorder_scores, _ = iupred(sequence, mode='long')
    binding_scores = anchor2(sequence, disorder_scores)

    iupred2_results[name] = {
        'disorder': disorder_scores,
        'binding': binding_scores,
    }

    mean_disorder = sum(disorder_scores) / len(disorder_scores)
    print(f"{name}: mean disorder = {mean_disorder:.3f}")
```

## AIUPred Batch Processing

For AIUPred, we initialize the models once and reuse them for all sequences.
This is the recommended approach for processing multiple sequences.

### Initialize Models

```python
# Initialize disorder prediction models once
disorder_model, disorder_reg, device = init_aiupred_models('disorder', force_cpu=True)
print(f"Models loaded on device: {device}")

# Initialize binding prediction models once
binding_model, binding_reg, _ = init_aiupred_models('binding', force_cpu=True)
print("Binding models loaded")
```

### Process Sequences

```python
# Process all sequences using the pre-initialized models
aiupred_results = {}

for name, sequence in sequences.items():
    # Predict disorder
    disorder_scores = predict_aiupred_disorder(
        sequence, disorder_model, disorder_reg, device, smoothing=True
    )

    # Predict binding
    binding_scores = predict_aiupred_binding(
        sequence, binding_model, binding_reg, device, binding=True, smoothing=True
    )

    aiupred_results[name] = {
        'disorder': disorder_scores,
        'binding': binding_scores,
    }

    mean_disorder = float(disorder_scores.mean())
    print(f"{name}: mean disorder = {mean_disorder:.3f}")
```

## Comparing Results

```python
import numpy as np

print("Summary of mean disorder scores:")
print(f"{'Protein':<20} {'IUPred2':>10} {'AIUPred':>10}")
print("-" * 42)

for name in sequences:
    iupred2_mean = np.mean(iupred2_results[name]['disorder'])
    aiupred_mean = np.mean(aiupred_results[name]['disorder'])
    print(f"{name:<20} {iupred2_mean:>10.3f} {aiupred_mean:>10.3f}")
```

## Reading from FASTA Files

For larger datasets, you'll typically read sequences from a FASTA file:

```python
def read_fasta(filepath):
    """Read sequences from a FASTA file."""
    sequences = {}
    header = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                header = line[1:].split()[0]  # Take first word as ID
                sequences[header] = ''
            elif header and line:
                sequences[header] += line.upper()

    return sequences

# Example usage:
# sequences = read_fasta('proteins.fasta')
# for name, seq in sequences.items():
#     scores = predict_aiupred_disorder(seq, disorder_model, disorder_reg, device, smoothing=True)
#     print(f"{name}: mean = {scores.mean():.3f}")
```

## Performance Tips

1. **Initialize models once** - Don't call `aiupred_disorder()` in a loop; use `init_aiupred_models()` + `predict_aiupred_disorder()`
2. **Use GPU when available** - Set `force_cpu=False` for significant speedup
3. **Process in batches** - Don't load all sequences into memory at once for very large datasets
