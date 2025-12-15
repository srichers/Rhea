# Rhea (not-main) – working notes

This branch contains a multi-task PyTorch surrogate for fast flavor instabilities in NSM neutrino transport. It predicts
the final four-flux tensor, growth rate, and stability flag from an initial four-flux input. Notes below capture how the
code is wired on `not-main`, what data it expects, and improvements to consider next.

## Layout
- `model_training/`: training and inference code (PyTorch model, losses, data readers, tooling).
- `jmcguig_tests/run_many_models.py`: quick hyperparameter sweep script (learning-rate grid) that trains and gnuplots.
- `cpp_interface/`: minimal C++/libtorch interface for exported TorchScript models.
- `Dockerfile`, `Jenkinsfile`: GPU-ready CI pipeline that generates synthetic data, trains, and exercises Python/C++ use.

## Data shape and sources
- Training expects HDF5 files with datasets `F4_initial(1|ccm) [sim,4,2,NF]`, `F4_final(1|ccm) [sim,4,2,NF]`,
  `growthRate(1|s) [sim]`, and scalar `nf`. Stable sets use `F4_initial(1|ccm)` plus `stable` bit.
- Example inputs in this repo live in `../../datasets/`. From the autopsy report (`datasets/datasets_autopsy_directory.md`):
  - Asymptotic sets (e.g. `asymptotic_M1-NuLib-old.h5`, `asymptotic_M1-NuLib.h5`) each hold ~1.5M rows of 4×2×3 flux data + growth rates.
  - Stable sets (e.g. `stable_M1-Nulib-7ms_rl[0-3].h5`) carry stability labels for the same tensor layout.
- `ml_read_data.py` splits each file with `train_test_split` (`test_size` in parms), optional permutation augmentation, and
  optional downsampling via `samples_per_database` for stable sets.

## Model and training loop (current branch)
- Architecture (`ml_neuralnet.py`): shared trunk + task-specific heads for stability (logit), growth rate (log of rate),
  density transfer matrix, and flux transfer matrix. Enforces number conservation across flavors/matter, optionally
  averages heavy-lepton outputs, and can enforce lepton number directly or via the y-matrix. Outputs are converted back
  to physical four-fluxes (`F4_from_y`) and growth rates in physical units. Added `predict_WhiskyTHC` export helper
  (y-dens/flux + stability-masked growthrate).
- Losses (`ml_loss.py`): per-task MSEs for densities/flux magnitude/direction/growth rate, BCE-with-logits for stability,
  optional unphysical penalty (negative density, |flux|>n). `contribute_loss` logs both loss value and max error.
- Training (`ml_trainmodel.py`):
  - Dataloaders: `configure_loader` builds `WeightedRandomSampler` over concatenated datasets with `epoch_num_samples`
    (defaults to dataset length if unspecified) and `batch_size`.
  - Loop: true minibatch loop (zip of asymptotic + stable loaders) with per-batch backward/step; test loss evaluated
    once per epoch. Tracks ELN violation max per epoch. Scheduler still steps per epoch (warmup then ReduceLROnPlateau).
  - Loss multipliers: defaults set in `train_asymptotic_model`
    (`loss_multiplier_{stable,ndens,fluxmag,direction,growthrate}=1`, `unphysical=10`) and can be overridden in `parms`.
  - Optimization: AdamW (or configured op) with LinearLR warmup then ReduceLROnPlateau. `get_current_lr.py` keeps the
    logged LR in sync with the active scheduler.
  - Logging/outputs: writes `parameters.txt`, tab-separated `loss.dat` (epoch, per-task losses/max, ELN violation, LR),
    and saves `model{epoch}_{device}.pt` every `output_every`.
- `jmcguig_tests/run_many_models.py` (hyperparameter sweep) currently:
  - Uses asymptotic `asymptotic_M1-NuLib-old.h5` and stable `stable_M1-Nulib-7ms_rl2/rl3.h5`.
  - Downsamples each DB to 200k (also `random_samples_per_database` cap for asymptotic), `test_size=0.2`,
    `epoch_num_samples=200k`, `batch_size=4096`. Loss multipliers set explicitly (matches defaults above).
  - AdamW with `learning_rate` grid `[1e-5, 5e-6]` (base `5e-2` is overridden), warmup 10, patience/cooldown 100,
    `epochs=10_000`. Writes one subdir per LR (`model_learning_rate_*`) with `loss.dat`, model checkpoints, and
    `quickplot.gplt` output.

## Branch context and deltas
- `not-main` adds: explicit task loss multipliers in the parameter dict, LR logging helper, and the sweep script above.
- Upstream `origin/main` adds (not yet merged here): `predict_WhiskyTHC` export helper, NaN filtering when loading data,
  and an option to randomly subsample each dataset (`random_samples_per_database`). These are now mirrored locally.
- `origin/evolve_task_weights` prototypes per-batch optimization with learned log-task-weights (Kendall et al. style),
  proper loader iteration (`for ... in zip(loader_asymptotic, loader_stable)`) and warmup that tolerates zero steps.
  Adopting the iterator pattern from that branch would also fix the current loop repeatedly re-instantiating
  `iter(loader)` and backpropagating one giant graph at epoch end.

## Known gaps / quick wins
- Defaults for `loss_multiplier_*` are in place; override as needed per dataset scaling.
- Batch loop now uses proper zipped loaders with per-batch optimizer steps; monitor for stability improvements.
- Data hygiene: NaNs are dropped on load; `random_samples_per_database` is available to cap dataset size; WhiskyTHC
  export helper is present.
- Capture and plot loss curves by task (and ELN violation) for LR sweeps; they currently exist only in `loss.dat`.
- Revisit sampler settings if changing dataset sizes; `epoch_num_samples` now defaults to dataset length when unset.

## How to run (not-main)
1) Install deps: CUDA-capable PyTorch + numpy, h5py, scikit-learn (see `Dockerfile` for a working set).
2) Point `database_list` and `stable_database_list` to available H5 files; set `loss_multiplier_*` values.
3) Run training (examples):
   - Sweep: `python jmcguig_tests/run_many_models.py` (creates `model_*` dirs per LR).
   - Single run: adapt `model_training/ml_pytorch.py` or write a small driver that builds `parms`, calls
     `read_asymptotic_data`, `read_stable_data`, then `train_asymptotic_model`.
4) Inference: `python model_training/example_use_model.py path/to/model.pt` to load a scripted model and inspect predicted
   stability, growth rate, and final densities/fluxes for the canned NSM1 example.

## Next investigation ideas
- Compare LR sweep results against upstream `origin/main` baselines once the iterator/loss-multiplier issues are fixed.
- Try the learned task-weight schedule from `origin/evolve_task_weights` to balance stability vs. growthrate errors.
- Export CPU models (`convert_model_to_cpu.py` in `origin/convert_to_cpu`) for downstream C++ coupling tests.
