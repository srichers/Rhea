# Rhea

Rhea is a PyTorch + **e3nn** codebase for training **symmetry-aware neural networks** on neutrino moment data. The core model takes neutrino 4-flux/4-current style inputs (`F4`) and learns to predict downstream targets like:

- an “asymptotic/final” `F4` state, and
- an instability **growth rate** (returned in **physical units of 1/s** at inference time).

The architecture is built to respect relevant structure in the data (e.g., rotational equivariance via `e3nn`, plus permutation-equivariant mixing across species axes like flavor and ν/ν̄ where appropriate).

---

## Key features

- **Equivariance with e3nn**
  Uses `e3nn.o3` irreps, tensor products, gated blocks, etc., so vector/tensor components transform correctly under rotations.

- **Permutation-equivariant species mixing**
  Custom permutation-equivariant layers (e.g., “self/flavor/ν–ν̄/all” mixing patterns) for species-indexed data.

- **Multi-task training support**
  The training loop supports multiple loss terms (asymptotic four flux, growth rate). There is support for
  task-specific weighting via learned log-weights.

- **Physically consistent normalization & rescaling**
  - Training data are normalized by total number density (`ntot`).
  - During inference, outputs are rescaled back to number density units.

- **Two entry points**
  - `model_training/`: training + evaluation utilities
  - `cpp_interface/`: C++ integration scaffolding for inference (for embedding the trained model in a C++ code path)

- **CI/automation**
  - GitHub Actions workflows live in `.github/workflows/`
  - A `Jenkinsfile` is included for Jenkins-based pipelines
  - A `Dockerfile` is included for reproducible environments

---

## Repository layout

```text
Rhea/
  .github/workflows/        CI workflows
  cpp_interface/            C++ inference interface (embedding trained models)
  model_training/           Python training + data loading + model definition
    ml_neuralnet.py         e3nn model architecture + prediction helpers
    ml_read_data.py         dataset readers + normalization checks
    ml_trainmodel.py        training loop + loss composition
    ml_loss.py              loss functions (incl. constraints/penalties)
    ...                     (additional utilities: constants, tools, etc.)
  Dockerfile
  Jenkinsfile
  LICENSE                   GPL-3.0
```

---

### Recommended install pattern

Although it has been tested to work in a number of environments, the only supported environment is that described by the `Dockerfile`. Follow the `Dockerfile` to install required dependencies, but note that there are some features in the `Dockerfile` that are not actually needed by `Rhea`.

---

## Data expectations

### Core tensor: `F4`

Most of the pipeline is built around a tensor commonly named `F4`, representing a 4-component quantity (i.e., 3 spatial components + time component) across neutrino species axes.

In the training code, `F4` is typically treated as having the logical shape:

```text
[nsamples, n_nunubar, n_flavor, n_xyzt]
```

where commonly:

- `n_nunubar = 2` (ν and ν̄)
- `n_flavor = 3` (flavor triplet)
- `n_xyzt = 4` (number flux spacetime indices)

### Normalization rules (important)

`model_training/ml_read_data.py` expects data to be sane:

- After normalization, both `ntot_initial` and `ntot_final` are expected to be the same (within tolerance).
- `ntot` must be positive.
- growth rates must be finite and positive.

If your dataset violates these assumptions, training will (intentionally) fail loudly.

---

## Training

Training is orchestrated by scripts/utilities in `model_training/`. The code is parameterized via a Python dictionary typically named `parms`.

### Typical knobs in `parms`

You’ll see parameters along these lines (non-exhaustive):

- **Model/architecture**
  - `irreps_hidden` (e3nn irreps string)
  - `nhidden_F4`
  - `scalar_activation`, `nonscalar_activation`
  - `dropout_probability`
  - `do_batchnorm` (preferred)

- **Physics/data handling**
  - internal normalization via `ntot`
  - growth rate scaling handled inside prediction (see “Inference”)

- **Training loop**
  - `epoch_num_samples` (used with `WeightedRandomSampler`)
  - loss multipliers (kept in the parameter dictionary)

### Running training

Because setups differ (data paths, config patterns, cluster quirks), the repo does not force a single rigid CLI. Start from the `model_training/ml_pytorch` and adjust the input parameters listed in the script.

---

## Inference

Inference is handled in the model code (see `model_training/ml_neuralnet.py`).

### What “predict_all” returns

Prediction utilities rescale outputs back to number density units:

- `F4_out` is rescaled to match the original total density of the input (`ntot`)
- `growthrate` is also returned in units of `ntot`. It must be rescaled by `sqrt(2.)*G_F` to get units of 1/s.

This is intentional: callers should not have to remember hidden unit conversions.

### Minimal inference sketch

Exact loading/saving utilities depend on how you train/save the model in your workflow, but the high-level pattern is:

```python
import torch

# F4_in: torch.Tensor shaped like [N, 2, 3, 4] in physical units
# model: your trained NeuralNetwork (nn.Module)

model.eval()
with torch.no_grad():
    F4_out, growthrate = model.predict_all(F4_in)

# F4_out: number density units
# growthrate: number density units
```

---

## C++ integration (`cpp_interface/`)

The `cpp_interface/` directory exists for integrating a trained model into C++.

Typical flow (project-dependent):

1. Export a trained model in a C++-loadable format (often TorchScript).
2. Load and run the model from C++ using LibTorch.
3. Wrap input/output reshaping to match `F4` conventions.

See the contents of `cpp_interface/` for the expected integration approach in this repo.

---

## License

This project is under the **GNU GPL v3.0**. See `LICENSE` for details.

---
