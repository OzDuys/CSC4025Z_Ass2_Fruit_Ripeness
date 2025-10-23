# Initial basic CNN Implementation
Notebook: notebooks/baseline_cnn_pytorch.ipynb

## Hyperparams:
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 5
VAL_SPLIT = 0.15
IMAGE_SIZE = 224

# Training Curves
notebooks/artefacts/baseline_cnn_pytorch1.png

## Test Performance
Test loss: 0.6778
Test accuracy: 0.723

---

# Second basic CNN run with 20 epochs instead of 5
Notebook: notebooks/baseline_cnn_pytorch.ipynb

## Hyperparams:
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 20
VAL_SPLIT = 0.15
IMAGE_SIZE = 224

# Training Curves
notebooks/artefacts/baseline_cnn_pytorch2.png

# Models parameters
training notebooks/artefacts/Baseline_CNN2.pt

## Test Performance
Test loss: 0.3799
Test accuracy: 0.845


---
