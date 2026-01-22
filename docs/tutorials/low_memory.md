# Low-Memory Prediction for Long Sequences

This tutorial demonstrates how to predict disorder for very long protein
sequences using memory-efficient chunked processing.

## When to Use Low-Memory Mode

The standard AIUPred prediction requires memory proportional to sequence length.
Approximate memory requirements:

| GPU VRAM | Max Sequence Length |
|----------|---------------------|
| 2 GB     | ~3,000 residues     |
| 6 GB     | ~8,000 residues     |
| 12 GB    | ~16,000 residues    |

For sequences longer than your memory allows, or when running on CPU with
limited RAM, use the low-memory functions which process sequences in chunks.

## Setup

```python
from iupred import (
    init_aiupred_models,
    predict_aiupred_disorder,
    low_memory_aiupred_disorder,
    low_memory_aiupred_binding,
)
```

## Example: Long Protein Sequence

For demonstration, we'll create a long sequence by repeating p53:

```python
# Human p53 sequence
p53_sequence = (
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"
    "DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK"
    "SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE"
    "RCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNS"
    "SCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP"
    "PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG"
    "GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
)

# Create a ~4000 residue sequence
long_sequence = p53_sequence * 10
print(f"Long sequence length: {len(long_sequence)} residues")
```

## Initialize Models

```python
# Initialize disorder model
disorder_model, disorder_reg, device = init_aiupred_models('disorder', force_cpu=True)
print(f"Models loaded on device: {device}")
```

## Standard vs Low-Memory Prediction

### Standard Prediction

```python
import time

# Standard prediction (may fail for very long sequences)
print("Running standard prediction...")
start = time.time()
try:
    standard_scores = predict_aiupred_disorder(
        long_sequence, disorder_model, disorder_reg, device, smoothing=True
    )
    standard_time = time.time() - start
    print(f"Standard prediction completed in {standard_time:.2f}s")
except RuntimeError as e:
    print(f"Standard prediction failed (likely out of memory): {e}")
    standard_scores = None
```

### Low-Memory Prediction

```python
# Low-memory prediction with chunking
print("Running low-memory prediction...")
start = time.time()
low_mem_scores = low_memory_aiupred_disorder(
    long_sequence,
    disorder_model,
    disorder_reg,
    device,
    smoothing=True,
    chunk_len=1000  # Process 1000 residues at a time
)
low_mem_time = time.time() - start
print(f"Low-memory prediction completed in {low_mem_time:.2f}s")
```

## Low-Memory Binding Prediction

```python
# Initialize binding model
binding_model, binding_reg, _ = init_aiupred_models('binding', force_cpu=True)

# Predict binding with low-memory mode
print("Running low-memory binding prediction...")
start = time.time()
binding_scores = low_memory_aiupred_binding(
    long_sequence,
    binding_model,
    binding_reg,
    device,
    smoothing=True,
    chunk_len=1000
)
binding_time = time.time() - start
print(f"Binding prediction completed in {binding_time:.2f}s")
```

## Adjusting Chunk Size

The `chunk_len` parameter controls the trade-off between memory usage and accuracy:

- **Larger chunks** (e.g., 2000): Faster, more accurate, but more memory
- **Smaller chunks** (e.g., 500): Slower, slightly less accurate, but less memory

The minimum chunk size is 201 (due to 100-residue overlap).

```python
# Compare different chunk sizes
chunk_sizes = [500, 1000, 2000]

for chunk_size in chunk_sizes:
    start = time.time()
    scores = low_memory_aiupred_disorder(
        long_sequence,
        disorder_model,
        disorder_reg,
        device,
        smoothing=True,
        chunk_len=chunk_size
    )
    elapsed = time.time() - start
    print(f"Chunk size {chunk_size}: {elapsed:.2f}s")
```

## Important Notes

!!! warning "Results May Differ Slightly"
    The low-memory approach may produce slightly different results at chunk
    boundaries due to edge effects, but overall predictions should be very similar.

## Tips for Very Long Sequences

1. **Start with a large chunk size** and decrease if you run out of memory
2. **Use GPU if available** - set `force_cpu=False`
3. **Process sequences in batches** - don't load all sequences into memory at once
4. **Consider using IUPred2 first** - it's much faster and doesn't require GPU
