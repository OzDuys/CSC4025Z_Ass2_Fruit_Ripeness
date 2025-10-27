# Hyperparameter Tuning Setup (Weights & Biases)

This document describes the Colab-friendly hyperparameter tuning workflow implemented in `training notebooks/cnn/hyperparameter_tuning_wandb.ipynb`. The goal is to iterate on CNN architectures for fruit-ripeness classification while logging rich diagnostics to Weights & Biases (W&B).

## Environment & Dependencies

The notebook targets Google Colab GPU runtimes but works locally with the same steps:

1. Install dependencies with the bundled cell:
   ```python
   %%capture
   !pip install -q kagglehub wandb torch torchvision torchaudio scikit-learn tqdm
   ```
2. Provide Kaggle API credentials. The script copies `kaggle.json` into `~/.kaggle/`, fixes permissions, and downloads the dataset via `kagglehub`.
3. The dataset is unpacked to `data/fruit_ripeness_dataset/` with helper utilities that copy and extract archive contents on first run.

## Data Pipeline

- The raw directory structure must expose `train/` and `test/` folders with class subdirectories.
- A seeded split reserves `val_split` (default 15%) of the training set for validation using `torch.utils.data.random_split`.
- Custom `SubsetWithTransform` wrappers apply separate transform pipelines:
  - **Training**: resize to `image_size`, optional flips, rotations, color jitter, normalization, and optional random erasing.
  - **Validation/Test**: resize and normalization only.
- Data loaders use batch size from config, shuffling for train only, and enable pinned memory when CUDA is available.

## Configuration Defaults

`DEFAULT_CONFIG` stores tunable knobs and sensible starting points:

| Key | Purpose |
| --- | --- |
| `seed` | Controls data splits and Torch RNG state. |
| `image_size` | Resize target for all transforms. |
| `val_split` | Validation proportion of the training folder. |
| `batch_size` | Mini-batch size shared across loaders. |
| `epochs` | Maximum epochs before early stopping triggers. |
| `learning_rate`, `weight_decay` | Optimizer hyperparameters. |
| `optimizer` | One of `adamw`, `adam`, or `sgd`. |
| `architecture` | Model family to instantiate. |
| `pretrained`, `freeze_backbone` | Transfer-learning switches for torchvision models. |
| `dropout` | Classifier dropout rate (also applied inside MobileNet/EfficientNet heads). |
| `label_smoothing` | Applied in `nn.CrossEntropyLoss` to stabilise training. |
| `scheduler`, `min_lr` | Scheduler type (`cosine` or `plateau`) and floor learning rate. |
| `aug_*` | Augmentation toggles/intensities for flips, rotation, color jitter, and random erasing. |
| `use_amp` | Enables automated mixed precision when CUDA is present. |
| `max_grad_norm` | Clips gradients when > 0. |
| `patience` | Early-stopping patience in epochs based on validation macro-F1. |
| `log_model` | Uploads the best checkpoint as a W&B artifact when true. |
| `wandb_mode` | Allows `offline`/`disabled` runs if needed. |

Override any subset of keys when calling `run_experiment(config=...)`.

## Model Choices

`build_model` supports four architectures out of the box:

- **SimpleCNN** – Lightweight baseline with three convolution + pooling blocks and fully connected head sized by `image_size`.
- **ResNet-18** – Optionally pretrained on ImageNet. When `freeze_backbone=True`, only the new dropout + linear head trains.
- **MobileNetV3-Small** – Uses the torchvision implementation with dropout replaced and the classifier head resized to three classes.
- **EfficientNet-B0** – Similar treatment, with adjustable dropout and optional backbone freezing.

Adding new models involves extending the architecture switch and ensuring the classifier head outputs the correct number of classes.

## Optimisation & Training Loop

- Optimizers: AdamW (default), Adam, or SGD with optional momentum/Nesterov.
- Schedulers: cosine annealing or `ReduceLROnPlateau`; the latter monitors validation macro-F1.
- Mixed precision: `autocast_context` wraps forward passes while `create_grad_scaler` returns a device-aware AMP scaler. Both fall back cleanly on older Torch versions.
- Each epoch logs training loss/accuracy and validation metrics. Validation results also drive early stopping based on macro-F1.
- The best-performing weights (highest validation macro-F1) are cached and restored before running the test evaluation.
- Gradient clipping, label smoothing, and patience are governed by config values.

## Metrics & Reporting

During validation and testing the pipeline records multiple signals:

- Accuracy (sanity check baseline)
- Macro F1 (primary selection metric for sweeps)
- Weighted F1 (class-frequency-aware)
- Balanced accuracy (mean recall across classes)
- ROC-AUC (OvR) derived from probability scores; reported as NaN when splits lack all classes

W&B logging includes confusion matrices, ROC curves, and a classification report table so you can drill into per-class behaviour.

## Weights & Biases Integration

- `WANDB_PROJECT` defaults to `fruit-ripeness-cnn`; set `WANDB_ENTITY` to your workspace as needed.
- Runs inherit base tags (`fruit-ripeness`, `cnn`, `hyperparam-tuning`) and accept custom tags/notes via the config dict.
- Set the `WANDB_API_KEY` environment variable or run `wandb.login()` interactively in the notebook before launching sweeps.
- Optional artifact logging persists the best checkpoint when `log_model=True`.

## Running Single Experiments

```python
metrics = run_experiment(
    config={
        'architecture': 'resnet18',
        'pretrained': True,
        'epochs': 15,
        'learning_rate': 3e-4,
    },
    enable_wandb=True,
    run_name='resnet18-pretrained'
)
```

- Pass `enable_wandb=False` to run locally without logging (useful for smoke tests).
- A helper cell in the notebook switches `if False` to `if True` for a quick 2-epoch debug pass.

## Sweep Configuration

The sweep definition uses Bayesian optimisation over a mixed search space:

- Objective: maximise `val_macro_f1`.
- Architectures: `simple_cnn`, `resnet18`, `mobilenet_v3_small`, `efficientnet_b0`.
- Transfer-learning toggles: `pretrained`, `freeze_backbone`.
- Continuous ranges: learning rate (`1e-5` to `3e-3`), weight decay (`1e-6` to `1e-2`), dropout (`0.1` to `0.6`).
- Discrete choices: batch size (`32`, `48`, `64`), color jitter (0.0/0.15/0.3), random erasing (0.0/0.1/0.25), optimizer (`adamw`, `adam`, `sgd`).

To launch:

```python
sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT, entity=WANDB_ENTITY)
wandb.agent(sweep_id, sweep_entry, count=NUM_RUNS)
```

`count` limits how many configurations each agent evaluates; start with a small batch (e.g., 6–8 runs) to gauge trends before scaling up.

## Runtime Considerations

- The default `epochs=15` combined with early stopping keeps runtime manageable (~30 minutes on a Colab L4 GPU). Under-performing configs exit early when macro-F1 plateaus.
- Batch size affects memory; if Colab raises OOM errors, reduce `batch_size` or increase `image_size` incrementally.
- Sweep agents respect the notebook’s global config, so adjust `WANDB_PROJECT`, dataset paths, or environment variables before calling `wandb.agent`.

## Extending the Setup

- **New Architecture**: Add a branch in `build_model`, ensure pretrained weights are optional, and adapt the classifier head.
- **Custom Metrics**: Extend `compute_metrics` to calculate additional statistics, then log them from `run_experiment`.
- **Augmentation Experiments**: Introduce new config keys and update `build_transforms` to toggle them.
- **Longer Training**: Increase `epochs` within the sweep parameter space or per-run config; patience prevents wasted epochs if validation stalls.

This setup intentionally separates data preparation, model construction, training utilities, and W&B orchestration so you can iterate on each component without rewriting the entire workflow.
