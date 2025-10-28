# Fruit Ripeness Classifier App - Setup Guide

## Quick Start

### 1. Install Requirements
```bash
pip install gradio torch torchvision pillow
```

### 2. Prepare Your Model
Save your trained model from your notebook. Add this to your notebook:
```python
# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names,
    'image_size': IMAGE_SIZE,
}, 'baseline_cnn.pt')
```

Place the `baseline_cnn.pt` file in the same directory as `fruit_classifier_app.py`.

### 3. Run the App
```bash
python fruit_classifier_app.py
```

This will:
- Start a local web server
- Open automatically in your browser
- Generate a public link (valid for 72 hours) that you can access on your phone

### 4. Use on Your Phone
- Copy the "public URL" that appears (looks like: `https://xxxxx.gradio.live`)
- Open it on your phone's browser
- Tap "Upload or capture fruit image" to use your camera
- The model will classify the fruit

## Customisation

### Update Class Names
If your dataset has different labels, modify the `class_names` list in the code.

### Adjust Image Size
If you trained with a different image size, update:
- The `transforms.Resize()` value
- The `nn.Linear()` input size in the classifier

### Add Example Images
Place example images in an `examples/` folder and add their paths to the `examples` parameter.

## Troubleshooting

**"Model file not found"**: Make sure `baseline_cnn.pt` is in the same directory as the script.

**Wrong predictions**: Ensure the preprocessing (normalization values, image size) matches your training setup.

**Can't access on phone**: Make sure both devices are on the same network, or use the public Gradio link (share=True).

## Notes
- The public link expires after 72 hours (free Gradio limitation)
- For a permanent solution, you could deploy to Hugging Face Spaces (also free)
- The app works offline if you don't use share=True (local network only)
