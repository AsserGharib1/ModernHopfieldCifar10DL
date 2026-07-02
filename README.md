# Modern Hopfield Network Image Classifier — CIFAR-10

PyTorch implementation of a **Modern (continuous) Hopfield Network** classifier with energy-based update rules, evaluated on the 60,000-image CIFAR-10 dataset.

## Highlights

- Custom `SparseHopfieldLayer` / `SparseHopfieldNet` modules implementing modern Hopfield associative retrieval over learned continuous embeddings.
- Multiple sparsity/architecture variants trained and compared (accuracy, precision, recall, F1).
- Preprocessing pipeline for continuous image embeddings; metrics and training curves logged (TensorBoard-compatible).
- Benchmarked against baseline deep classifiers on the same split.

## Repository contents

- `modern_hopfield_cifar10.ipynb` — dataset loading (via `kagglehub`), model definitions, training, and full evaluation with preserved outputs.

## Running

```bash
pip install -r requirements.txt
jupyter notebook modern_hopfield_cifar10.ipynb
```

Reference: Ramsauer et al., *Hopfield Networks is All You Need* (2020).
