# vlm_basic_classifier

Add classifier end to CLIP pre-trained model to classify FashionDataset.

<figure>
    <img src="bar_clipfinetune_val.png" alt="Classification Performance">
    <figcaption>Figure 1: Classification Performance.</figcaption>
</figure>

Using CrossEntropy to map to known text.

Notes:
- Missing classes may be due to class imbalance or low confidence
- Remedies: weighted loss function, oversample minority class, data augmentation on minority class, reduce learning rate, label smoothing

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

### Prerequisites

Install `uv` if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install dependencies

```bash
uv venv            # create .venv in the project directory
uv sync            # install all dependencies from pyproject.toml
```

#### GPU / CUDA support (optional)

By default, `uv sync` installs the CPU-only PyTorch wheels. To use CUDA, uncomment and configure the `[[tool.uv.index]]` block in [pyproject.toml](pyproject.toml), then re-run `uv sync`.

---

## Usage

```bash
# Custom VLM trained with contrastive loss on CIFAR-10
uv run python basic_classifier.py

# Fine-tune CLIP on the Fashion Products dataset
uv run python clip_finetune.py
```

---

## Files

| File | Description |
|------|-------------|
| [basic_classifier.py](basic_classifier.py) | Custom CNN + text embedding VLM trained with InfoNCE loss on CIFAR-10 |
| [clip_finetune.py](clip_finetune.py) | Fine-tunes a frozen CLIP vision encoder with a linear classifier on FashionDataset |
| [plotter.py](plotter.py) | Shared utilities: confusion matrix, t-SNE, classification report bar chart |
| [pyproject.toml](pyproject.toml) | Project dependencies and ruff configuration |
