# Hyperparameter Tuning Summary — Fruit Ripeness CNN (Oct 28, 2025)

_All results train from scratch (no pretrained fine-tuning). Data sourced from `Fruit Ripeness CNN Export Oct 28 2025.csv` plus W&B artefacts in `training notebooks/cnn/hyperparam_tuning/artefacts/`._

## Experiment Snapshot
- Total runs analysed: 31 (finished from-scratch runs: 24, crashed: 4).
- Architecture mix among finished runs: simple_cnn 45.8%, efficientnet_b0 20.8%, mobilenet_v3_small 16.7%, resnet18 16.7%.
- Primary selection metric: macro-averaged F1 on the validation split; test metrics captured for all finished runs.

## Top Configurations
| Run | Arch | LR | Weight Decay | Batch | Dropout | Color Jitter | Random Erasing | Test macro F1 | Test Acc | Test Bal Acc | Val macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| still-sweep-1 | resnet18 | 3.65e-04 | 1.15e-06 | 16 | 0.444 | 0.15 | 0.00 | 0.985 | 0.988 | 0.985 | 0.979 |
| youthful-sweep-3 | efficientnet_b0 | 4.34e-04 | 5.32e-03 | 32 | 0.164 | 0.30 | 0.25 | 0.982 | 0.985 | 0.982 | 0.974 |
| apricot-sweep-4 | resnet18 | 4.92e-05 | 8.42e-06 | 16 | 0.517 | 0.15 | 0.25 | 0.960 | 0.967 | 0.962 | 0.955 |
| glorious-sweep-1 | efficientnet_b0 | 1.71e-04 | 1.36e-04 | 32 | 0.539 | 0.15 | 0.10 | 0.960 | 0.966 | 0.961 | 0.954 |
| winter-sweep-2 | efficientnet_b0 | 1.21e-04 | 3.85e-06 | 16 | 0.116 | 0.30 | 0.00 | 0.952 | 0.959 | 0.953 | 0.947 |
| graceful-sweep-5 | mobilenet_v3_small | 3.84e-04 | 1.31e-03 | 48 | 0.178 | 0.00 | 0.00 | 0.944 | 0.953 | 0.945 | 0.934 |

## Architecture-Level Outcomes
- **resnet18** (4 runs) — mean test macro F1 0.959, best 0.985; mean test accuracy 0.966.
- **efficientnet_b0** (5 runs) — mean test macro F1 0.938, best 0.982; mean test accuracy 0.946.
- **mobilenet_v3_small** (4 runs) — mean test macro F1 0.885, best 0.944; mean test accuracy 0.899.
- **simple_cnn** (11 runs) — mean test macro F1 0.854, best 0.901; mean test accuracy 0.872.

ResNet-18 remains the strongest from-scratch backbone (macro F1 up to ≈0.985), closely followed by EfficientNet-B0 (≈0.982). Expanded SimpleCNN sweeps now peak at ≈0.901 macro F1, a modest gain over earlier attempts, narrowing the gap to MobileNetV3-Small (best ≈0.944).

## SimpleCNN Focus
- Completed SimpleCNN runs: 11; test macro F1 mean 0.854 (σ=0.044), test accuracy mean 0.872.
- Batch sizes explored: 48 (×5), 16 (×3), 32 (×2), 64 (×1). Higher macro F1 clustered around batch 48 with dropout 0.30–0.40 and color jitter ≈0.15.
- Parameter importance (`param_importance_basic_cnn.png`) ranks color jitter as the most influential knob (positive correlation with macro F1), followed by runtime/batch effects and dropout (negative correlation beyond ≈0.4). Learning rate shows moderate positive correlation, while weight decay and large batch sizes trend downward.
- Correlations (SimpleCNN subset): learning rate ρ≈+0.25, random erasing ρ≈+0.31, dropout ρ≈−0.36, batch size ρ≈−0.22, aligning with the importance chart and suggesting gains from moderate augmentation plus restrained dropout.

## Cross-Architecture Insights
- ResNet-18 rewards conservative learning rates (≤5e-4) and minimal weight decay (≤1e-5), with higher dropout (0.45–0.55) and limited random erasing providing regularisation.
- EfficientNet-B0 performs best with LR 1e-4–4e-4, color jitter up to 0.3, and random erasing ≤0.25; larger batch sizes (32) outperformed smaller ones.
- MobileNetV3-Small prefers larger batches (≥48) and little to no color jitter; its top run used LR ~3.8e-4 with AdamW and zero random erasing.

## Observations
- Additional SimpleCNN sweeps reduced variance (σ≈0.044 macro F1) but the architecture remains ~8 F1 points behind ResNet/EfficientNet.
- Several runs (e.g., `happy-sweep-4`, `fluent-sweep-1`, `vibrant-sweep-1`) lack test metrics due to early termination; review logs before reusing those settings.
- ROC-AUC remains `NaN` when validation splits miss classes; stratified sampling or larger validation batches may resolve this.

## Next Steps
1. **SimpleCNN refinement**: sweep LR 3e-4–6e-4, dropout 0.2–0.4, batch 32–48, color jitter 0.1–0.2, random erasing 0.1–0.25 to target macro F1 ≥0.905.
2. **ResNet/EfficientNet focus**: tighten LR 3e-4–5e-4, weight decay ≤5e-6, batch ≥32, dropout 0.4–0.6; capture confusion matrices and ROC curves for reproducibility.
3. **MobileNet improvements**: explore stronger weight decay (1e-4–1e-3) and alternative augmentations (CutMix/MixUp) to offset its accuracy gap while keeping compute light.
4. **Validation hygiene**: adopt stratified splits or fixed validation folds to stabilise ROC metrics and reduce variance across runs.
