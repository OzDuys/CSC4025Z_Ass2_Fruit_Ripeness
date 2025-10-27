# Testing kNN Baseline on Unseen Images

This directory contains tools to test the kNN baseline model on your own fruit images.

## 📁 Files

- **`baseline_knn.ipynb`**: Main notebook for training and testing kNN
- **`test_knn_on_image.py`**: Standalone Python script for quick testing
- **`artefacts/knn_model.pkl`**: Saved kNN model (generated after running the notebook)

## 🚀 Quick Start

### Option 1: Use the Notebook (Recommended)

1. Run all cells in `baseline_knn.ipynb` to train and save the model
2. Scroll to **Section 10: Test kNN on Your Own Unseen Images**
3. Edit the `test_image_path` variable to point to your image
4. Run the cell to see predictions with visualization

### Option 2: Use the Command-Line Script

After training the model in the notebook, you can use the standalone script:

```bash
# Test a single image
python test_knn_on_image.py path/to/your/fruit.jpg

# Use different K value
python test_knn_on_image.py path/to/your/fruit.jpg --k 3

# Test all images in a folder
python test_knn_on_image.py --folder path/to/images/

# Save the visualization
python test_knn_on_image.py my_fruit.jpg --save output.png
```

## 📝 Where is the Model Saved?

**Location**: `artefacts/knn_model.pkl`

**What's inside**:
- All training images (flattened to vectors)
- Training labels
- Class names
- Configuration (image size, best K value)

**Why so large?** kNN doesn't learn parameters like neural networks. Instead, it memorizes ALL training data (~5,000+ images). The model file will be 50-200 MB depending on your dataset size.

## 🎯 How kNN Prediction Works

1. **Load your image** → Resize to 64×64 → Flatten to 12,288-dimensional vector
2. **Compare to ALL training images** using Euclidean distance
3. **Find K nearest neighbors** (K=1 works best for this dataset)
4. **Vote**: Most common class among neighbors wins

## 📊 Expected Results

Based on the test set performance:

| K Value | Accuracy | Use Case |
|---------|----------|----------|
| K=1 | 74.4% | **Best overall** (recommended) |
| K=3 | 68.3% | More stable predictions |
| K=5 | 67.2% | Standard baseline |
| K=20 | 63.8% | Too much averaging |

## 🖼️ Example Usage in Notebook

```python
# Load the model (run once)
model_data = load_knn_model('artefacts/knn_model.pkl')

# Test a single image
visualize_prediction(
    "~/Desktop/my_apple.jpg", 
    model_data, 
    k=1,  # Use best K
    show_top_n=5  # Show top 5 predictions
)

# Test multiple images in a folder
predict_folder("~/Desktop/fruit_photos", model_data, k=1, max_images=12)
```

## 🔍 What Gets Displayed

The visualization shows:
1. **Your input image** (original resolution)
2. **Top 5 predictions** with confidence percentages
3. **Bar chart** showing probability distribution
4. **Color coding**:
   - 🟢 Green: Top prediction
   - 🟠 Orange: Medium confidence (50-70%)
   - 🔴 Red: Low confidence (<50%)

## 📌 Tips for Best Results

1. **Use fruit images similar to training data**:
   - Studio-quality photos work best
   - Plain/uniform background
   - Centered fruit
   - Good lighting

2. **Supported fruits**:
   - Apples
   - Bananas
   - Oranges

3. **Ripeness states**:
   - Fresh (ripe)
   - Unripe (green)
   - Rotten (brown spots, decay)

4. **Image formats**: `.jpg`, `.jpeg`, `.png`

## ⚠️ Limitations

- **Slow for many images**: kNN compares to ALL training images (no GPU acceleration)
- **Sensitive to image quality**: Works best with images similar to training data
- **Memory intensive**: Model file is large (~50-200 MB)
- **Not robust to variations**: Unlike CNNs, kNN struggles with different angles, lighting, backgrounds

## 🆚 Comparison with CNN

Your CNN baseline achieves **84.5%** vs kNN's **74.4%** — the ~10 percentage point improvement demonstrates the value of learned hierarchical features over raw pixel distances.

## 🐛 Troubleshooting

**"Model not found"**
- Make sure you've run the full `baseline_knn.ipynb` notebook first
- Check that `artefacts/knn_model.pkl` exists

**"Image not found"**
- Verify the image path is correct
- Use absolute paths if relative paths fail
- On macOS, drag-drop the file to Terminal to get the full path

**"Poor predictions"**
- Try different K values (1, 3, 5)
- Ensure image is of a fruit (apple, banana, or orange)
- Check image quality (avoid blurry, dark, or complex backgrounds)

## 📚 For Your Assignment

Use this to:
1. ✅ Demonstrate kNN baseline performance
2. ✅ Compare with your CNN model
3. ✅ Test on out-of-distribution images (real-world photos)
4. ✅ Analyze where kNN fails but CNN succeeds
5. ✅ Show the value of learned features vs raw pixels

---

**Need help?** Check the notebook comments or refer to the inline documentation in `test_knn_on_image.py`.
