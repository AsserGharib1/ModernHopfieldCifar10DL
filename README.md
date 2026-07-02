# Modern Hopfield Network Image Classifier — CIFAR-10

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsserGharib1/ModernHopfieldCifar10DL/blob/main/modern_hopfield_cifar10.ipynb)
[![View on nbviewer](https://img.shields.io/badge/view%20full%20notebook-nbviewer-F37626?logo=jupyter&logoColor=white)](https://nbviewer.org/github/AsserGharib1/ModernHopfieldCifar10DL/blob/main/modern_hopfield_cifar10.ipynb)

> **Viewing tip:** GitHub truncates the inline preview of large notebooks (this one preserves all training outputs). Use the **nbviewer** badge above to read it fully rendered in the browser, or **Colab** to open it interactively.


PyTorch implementation of a **Modern (continuous) Hopfield Network** classifier with energy-based update rules, evaluated on the 60,000-image CIFAR-10 dataset.

## Highlights

- Custom `SparseHopfieldLayer` / `SparseHopfieldNet` modules implementing modern Hopfield associative retrieval over learned continuous embeddings.
- Multiple sparsity/architecture variants trained and compared (accuracy, precision, recall, F1).
- Preprocessing pipeline for continuous image embeddings; metrics and training curves logged (TensorBoard-compatible).
- Benchmarked against baseline deep classifiers on the same split.

## Training curves

![Learning curves A](figures/learning_curves_a.png)

![Learning curves B](figures/learning_curves_b.png)

## Repository contents

- `modern_hopfield_cifar10.ipynb` — dataset loading (via `kagglehub`), model definitions, training, and full evaluation with preserved outputs.

## Running

```bash
pip install -r requirements.txt
jupyter notebook modern_hopfield_cifar10.ipynb
```

Reference: Ramsauer et al., *Hopfield Networks is All You Need* (2020).
