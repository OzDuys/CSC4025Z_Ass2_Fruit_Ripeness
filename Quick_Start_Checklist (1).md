# Quick Start Checklist - Experiment 1.4

## ⏱️ Total Time: ~45 minutes
- 5 min: Setup
- 35-40 min: Training
- 5 min: Review results

---

## ✅ Pre-Flight Checklist

### 1. Get Your Kaggle Token (5 minutes)
- [ ] Go to https://www.kaggle.com/
- [ ] Log in with your Google account
- [ ] Click profile picture → Settings
- [ ] Scroll to "API" section
- [ ] Click "Create New Token"
- [ ] Download `kaggle.json` (check Downloads folder)

**Need help?** See `Kaggle_API_Token_Guide.md`

---

### 2. Set Up Google Colab (2 minutes)
- [ ] Go to https://colab.research.google.com/
- [ ] Click "File" → "Upload notebook"
- [ ] Upload `experiment_1_4_extended_training_lr_scheduler.ipynb`
- [ ] Click "Runtime" → "Change runtime type"
- [ ] Select **"T4 GPU"** or **"GPU"**
- [ ] Click "Save"

---

### 3. Run the Experiment (35-40 minutes)

#### Section 1: Install Dependencies (~2 min)
- [ ] Click the play button on cell 1
- [ ] Wait for packages to install

#### Section 2: Import Libraries (~10 sec)
- [ ] Run cell 2
- [ ] Verify output shows "Using device: cuda" ✅
- [ ] If it shows "cpu", go back to step 2 and select GPU

#### Section 3: Upload Kaggle Token (~30 sec)
- [ ] Run cell 3
- [ ] Click "Choose Files" button when prompted
- [ ] Select your `kaggle.json` file
- [ ] Wait for "kaggle.json uploaded" message
- [ ] Run the chmod cell

#### Section 4: Download Dataset (~3-5 min)
- [ ] Run the dataset download cell
- [ ] Watch progress bars
- [ ] Wait for "Using train directory" and "Using test directory" messages

#### Section 5: Configure Experiment (~5 sec)
- [ ] Run the hyperparameters cell
- [ ] Verify settings:
  - Epochs: 40 ✅
  - Batch size: 32 ✅
  - LR Scheduler: True ✅

#### Section 6: Load Data (~30 sec)
- [ ] Run the dataset loading cell
- [ ] Check output shows train/val/test sizes

#### Section 7: Define Model (~5 sec)
- [ ] Run the SimpleCNN definition cell

#### Section 8: Initialize Training (~5 sec)
- [ ] Run the model initialization cell
- [ ] Check for "✅ Learning Rate Scheduler enabled" message
- [ ] Note the model parameters (should be ~X.XX M)

#### Section 9: Training Loop (~35-40 min)
- [ ] Run the training cell
- [ ] ☕ **Take a break!** This will take 35-40 minutes
- [ ] Watch for:
  - Progress bars for each epoch
  - "📉 Learning rate reduced" messages
  - "⭐ BEST!" markers on good epochs

#### Section 10: Visualize Results (~10 sec)
- [ ] Run the visualization cell
- [ ] Review all 4 plots
- [ ] Check summary statistics

#### Section 11: Test Set Evaluation (~10 sec)
- [ ] Run test evaluation cell
- [ ] Note final test accuracy

#### Section 12: Save Results (~5 sec)
- [ ] Run the save results cell
- [ ] Verify files were saved

#### Section 13: Compare Experiments (optional)
- [ ] Skip for now (run after multiple experiments)

---

## 📊 What to Look For During Training

### Good Signs ✅
- Training and validation accuracy both increasing
- Loss decreasing smoothly
- Learning rate reduces 1-2 times
- Validation accuracy still improving near epoch 40

### Warning Signs ⚠️
- Large gap between train and val accuracy (>10%)
- Validation loss increasing while training loss decreasing
- Loss becomes NaN (learning rate too high)
- No improvement after epoch 20

---

## 📝 Record These Results

After training completes, fill this in:

```
EXPERIMENT 1.4 RESULTS
=====================
Date: ___________
Time taken: _____ minutes

Best Validation Accuracy: _____%
Best Epoch: _____
Final Test Accuracy: _____%
Train/Val Gap: _____%
Learning Rate Reductions: _____
Final Learning Rate: _____

Observations:
- Did training look smooth? __________
- When did val accuracy peak? __________
- Any overfitting signs? __________
- How does this compare to baseline? __________

Next Steps:
- What experiment should I try next? __________
```

---

## 💾 Download Your Results

Before closing Colab:
- [ ] Click folder icon (left sidebar)
- [ ] Navigate to `experiments/Exp_1.4_Extended_Training_LR_Scheduler/`
- [ ] Right-click each file → Download:
  - [ ] `training_curves.png`
  - [ ] `results.json`
  - [ ] `best_model.pt` (optional, if needed)

---

## 🔄 Compare to Baseline

| Metric | Your Baseline | Experiment 1.4 | Improvement |
|--------|---------------|----------------|-------------|
| Epochs | 20 | 40 | +20 |
| Val Acc | ____% | ____% | +____% |
| Test Acc | ____% | ____% | +____% |
| Gap | ____% | ____% | ____% |

**Target**: 3-5% improvement over baseline

---

## 🚀 Next Experiments (After This One)

Choose based on your results:

### If you got 3-5%+ improvement ✅
**Option A**: Try Experiment 2.3 (SGD optimizer)
- Different optimizer might work even better
- ~40 minutes

**Option B**: Try Experiment 3.2 (more data augmentation)
- Could reduce any overfitting
- ~40 minutes

### If you see overfitting (train >> val) ⚠️
**Must try**: Experiment 3.1 or 3.2 (regularization)
- More dropout or augmentation
- ~40 minutes

### If improvement is small (<2%) 😕
**Option A**: Try Experiment 1.3 (higher learning rate)
- Maybe need more aggressive learning
- ~40 minutes

**Option B**: Try Experiment 4.1 (wider network)
- Maybe need more capacity
- ~40 minutes

---

## 🐛 Common Issues & Fixes

### Issue: "Using device: cpu" instead of "cuda"
**Fix**: 
1. Runtime → Change runtime type
2. Select GPU
3. Restart and run all cells again

### Issue: Kaggle authentication failed
**Fix**:
1. Re-run Section 3
2. Upload kaggle.json again
3. Make sure file is named exactly "kaggle.json"

### Issue: Out of memory
**Fix**:
1. Runtime → Restart runtime
2. Run all cells again from the beginning
3. If persists, batch size might be too large (unlikely with 32)

### Issue: Training very slow (>1 hour)
**Fix**:
1. Check you're using GPU (see first issue)
2. Dataset should be ~15-30k images total
3. Each epoch should take 50-90 seconds

### Issue: Can't find downloaded files
**Fix**:
1. In Colab, click folder icon (left)
2. Files are in `experiments/Exp_1.4.../`
3. Right-click → Download

---

## 📧 For Your Report

Make sure to save:
1. ✅ Training curves image
2. ✅ Final accuracy numbers
3. ✅ Configuration settings
4. ✅ Observations about training behavior
5. ✅ Comparison with baseline

These will go into your Phase 1 section of the report.

---

## ⏭️ After Experiment 1.4

You'll have:
- ✅ Results from extended training
- ✅ Understanding of LR scheduler impact
- ✅ Baseline for further experiments
- ✅ Template for future experiments

**Estimated experiments needed**: 5-8 more for a complete study

**Estimated total time**: 4-6 hours of compute time (can run overnight)

---

## 🎯 Success Criteria

This experiment is successful if:
- ✅ Training completes without errors
- ✅ Test accuracy > baseline accuracy
- ✅ You understand what happened during training
- ✅ Results are saved and documented

Don't worry if improvement is small - that's valuable data too!

---

## 💡 Pro Tips

1. **Run overnight**: Set it up before bed, check results in the morning
2. **Multiple tabs**: Open multiple Colab tabs for parallel experiments
3. **Save often**: Download results immediately after each experiment
4. **Document everything**: Write notes while training runs
5. **Compare systematically**: Use Section 13 after 3+ experiments

---

**Ready to start?** Follow the checklist from top to bottom! 🚀

**Questions?** Check the detailed guides:
- `Kaggle_API_Token_Guide.md` - For token issues
- `Experiment_1.4_Summary.md` - For detailed explanations
- `Hyperparameter_Tuning_Plan.md` - For next experiments
