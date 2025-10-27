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

# Per epoch losses
Epoch 01/20 | train_loss: 2.1261, train_acc: 0.352 | val_loss: 0.9207, val_acc: 0.605
Epoch 02/20 | train_loss: 1.3186, train_acc: 0.447 | val_loss: 0.8909, val_acc: 0.658
Epoch 03/20 | train_loss: 1.1796, train_acc: 0.508 | val_loss: 0.7665, val_acc: 0.688
Epoch 04/20 | train_loss: 1.1235, train_acc: 0.529 | val_loss: 0.7233, val_acc: 0.703
Epoch 05/20 | train_loss: 1.0864, train_acc: 0.550 | val_loss: 0.6947, val_acc: 0.699
Epoch 06/20 | train_loss: 1.0331, train_acc: 0.564 | val_loss: 0.5976, val_acc: 0.740
Epoch 07/20 | train_loss: 0.9934, train_acc: 0.577 | val_loss: 0.6209, val_acc: 0.733
Epoch 08/20 | train_loss: 0.9744, train_acc: 0.584 | val_loss: 0.5782, val_acc: 0.757
Epoch 09/20 | train_loss: 0.9169, train_acc: 0.607 | val_loss: 0.5825, val_acc: 0.724
Epoch 10/20 | train_loss: 0.9019, train_acc: 0.616 | val_loss: 0.5856, val_acc: 0.734
Epoch 11/20 | train_loss: 0.8725, train_acc: 0.627 | val_loss: 0.5649, val_acc: 0.745
Epoch 12/20 | train_loss: 0.8627, train_acc: 0.634 | val_loss: 0.5276, val_acc: 0.771
Epoch 13/20 | train_loss: 0.8269, train_acc: 0.644 | val_loss: 0.5155, val_acc: 0.779
Epoch 14/20 | train_loss: 0.8008, train_acc: 0.652 | val_loss: 0.5215, val_acc: 0.764
Epoch 15/20 | train_loss: 0.7607, train_acc: 0.671 | val_loss: 0.4949, val_acc: 0.760
Epoch 16/20 | train_loss: 0.7405, train_acc: 0.675 | val_loss: 0.4892, val_acc: 0.757
Epoch 17/20 | train_loss: 0.6992, train_acc: 0.694 | val_loss: 0.4652, val_acc: 0.792
Epoch 18/20 | train_loss: 0.6734, train_acc: 0.705 | val_loss: 0.4638, val_acc: 0.789
Epoch 19/20 | train_loss: 0.6587, train_acc: 0.707 | val_loss: 0.4371, val_acc: 0.807
Epoch 20/20 | train_loss: 0.6266, train_acc: 0.727 | val_loss: 0.4174, val_acc: 0.812

## Test Performance
Test loss: 0.3799
Test accuracy: 0.845


---
