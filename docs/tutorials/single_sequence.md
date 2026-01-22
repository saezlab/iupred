# Single Sequence Prediction

This tutorial demonstrates how to predict intrinsically disordered regions
for a single protein sequence using both IUPred2 and AIUPred methods.

## Setup

First, import the necessary functions from the `iupred` package:

```python
from iupred import iupred, anchor2, aiupred_disorder, aiupred_binding
```

## Example Sequence

We'll use the human p53 tumor suppressor protein as our example.
p53 has well-characterized disordered regions at the N-terminus and C-terminus,
with a structured DNA-binding domain in the middle.

```python
# Human p53 protein sequence (UniProt P04637)
p53_sequence = (
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"
    "DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK"
    "SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE"
    "RCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNS"
    "SCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP"
    "PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG"
    "GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
)

print(f"Sequence length: {len(p53_sequence)} residues")
```

## IUPred2 Prediction

IUPred2 uses energy-based estimation to predict disordered regions.
It's fast and works without any additional downloads.

```python
# Predict disorder using IUPred2 (long mode)
disorder_scores, glob_text = iupred(p53_sequence, mode='long')

print("IUPred2 scores for first 10 residues:")
for i, (aa, score) in enumerate(zip(p53_sequence[:10], disorder_scores[:10])):
    print(f"  {i+1:3d} {aa}: {score:.3f}")
```

### ANCHOR2 Binding Prediction

ANCHOR2 predicts disordered binding regions - regions that are disordered
but can fold upon binding to partner proteins.

```python
# Predict binding regions using ANCHOR2
# Note: ANCHOR2 requires IUPred scores as input
binding_scores = anchor2(p53_sequence, disorder_scores)

print("ANCHOR2 binding scores for first 10 residues:")
for i, (aa, score) in enumerate(zip(p53_sequence[:10], binding_scores[:10])):
    print(f"  {i+1:3d} {aa}: {score:.3f}")
```

## AIUPred Prediction

AIUPred uses transformer-based deep learning for enhanced disorder prediction.

!!! note
    Model weights (~82 MB) are automatically downloaded on first use.

```python
# Predict disorder using AIUPred
ai_disorder_scores = aiupred_disorder(p53_sequence, force_cpu=True)

print("AIUPred disorder scores for first 10 residues:")
for i, (aa, score) in enumerate(zip(p53_sequence[:10], ai_disorder_scores[:10])):
    print(f"  {i+1:3d} {aa}: {score:.3f}")
```

```python
# Predict binding using AIUPred-binding
ai_binding_scores = aiupred_binding(p53_sequence, force_cpu=True)

print("AIUPred binding scores for first 10 residues:")
for i, (aa, score) in enumerate(zip(p53_sequence[:10], ai_binding_scores[:10])):
    print(f"  {i+1:3d} {aa}: {score:.3f}")
```

## Visualizing Results

Here's how to plot the disorder predictions:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

positions = range(1, len(p53_sequence) + 1)

# Disorder predictions
axes[0].plot(positions, disorder_scores, label='IUPred2', alpha=0.8)
axes[0].plot(positions, ai_disorder_scores, label='AIUPred', alpha=0.8)
axes[0].axhline(y=0.5, color='gray', linestyle='--', label='Threshold')
axes[0].set_ylabel('Disorder Score')
axes[0].set_title('Disorder Prediction for Human p53')
axes[0].legend()
axes[0].set_ylim(0, 1)

# Binding predictions
axes[1].plot(positions, binding_scores, label='ANCHOR2', alpha=0.8)
axes[1].plot(positions, ai_binding_scores, label='AIUPred-binding', alpha=0.8)
axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Threshold')
axes[1].set_xlabel('Residue Position')
axes[1].set_ylabel('Binding Score')
axes[1].set_title('Binding Region Prediction for Human p53')
axes[1].legend()
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()
```

## Interpreting Results

- **Scores > 0.5** indicate predicted disordered or binding regions
- The N-terminus (residues 1-100) shows high disorder - this is the transactivation domain
- The DNA-binding domain (residues ~100-290) shows lower disorder scores
- The C-terminus shows high disorder - this is the regulatory domain

Both IUPred2 and AIUPred capture these known structural features of p53.
