#!/usr/bin/env python3
"""
Standalone script to test the kNN baseline model on unseen images.

Usage:
    python test_knn_on_image.py path/to/your/image.jpg
    python test_knn_on_image.py path/to/your/image.jpg --k 3
    python test_knn_on_image.py --folder path/to/images/
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier


def load_knn_model(model_path):
    """Load the saved kNN model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def predict_single_image(image_path, model_data, k=None):
    """Predict the class of a single image using the saved kNN model."""
    # Extract model components
    X_train_flat = model_data['X_train_flat']
    y_train = model_data['y_train']
    class_names = model_data['class_names']
    image_size = model_data['image_size']
    k = k if k is not None else model_data['best_k']
    
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    original_img = img.copy()
    img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    img_array = np.array(img)
    
    # Flatten and normalize
    img_flat = img_array.reshape(1, -1).astype(np.float32) / 255.0
    
    # Create and fit kNN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    knn.fit(X_train_flat, y_train)
    
    # Predict
    pred_class_idx = knn.predict(img_flat)[0]
    pred_class_name = class_names[pred_class_idx]
    
    # Get prediction probabilities
    pred_proba = knn.predict_proba(img_flat)[0]
    confidence = pred_proba[pred_class_idx]
    
    return pred_class_name, pred_class_idx, confidence, pred_proba, original_img


def visualize_prediction(image_path, model_data, k=None, show_top_n=5, save_path=None):
    """Predict and visualize the result for a single image."""
    # Predict
    pred_class_name, pred_class_idx, confidence, pred_proba, original_img = predict_single_image(
        image_path, model_data, k
    )
    
    class_names = model_data['class_names']
    k = k if k is not None else model_data['best_k']
    
    # Get top N predictions
    top_indices = np.argsort(pred_proba)[::-1][:show_top_n]
    top_classes = [class_names[i] for i in top_indices]
    top_probs = [pred_proba[i] for i in top_indices]
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Show original image
    ax1.imshow(original_img)
    ax1.axis('off')
    ax1.set_title(f'Input Image\n{Path(image_path).name}', fontsize=12, fontweight='bold')
    
    # Show prediction probabilities
    colors = ['green' if i == 0 else 'gray' for i in range(show_top_n)]
    
    ax2.barh(range(show_top_n), top_probs, color=colors, alpha=0.7)
    ax2.set_yticks(range(show_top_n))
    ax2.set_yticklabels(top_classes)
    ax2.set_xlabel('Confidence', fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.set_title(f'Top {show_top_n} Predictions (K={k})', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, prob in enumerate(top_probs):
        ax2.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.suptitle(f'Prediction: {pred_class_name} ({confidence*100:.1f}% confidence)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Image: {Path(image_path).name}")
    print(f"Predicted class: {pred_class_name}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"K neighbors: {k}")
    print(f"\nTop {show_top_n} predictions:")
    for i, (cls, prob) in enumerate(zip(top_classes, top_probs), 1):
        print(f"  {i}. {cls:20s}: {prob*100:.1f}%")
    print(f"{'='*60}")


def predict_folder(folder_path, model_data, k=None, max_images=12):
    """Predict on all images in a folder."""
    folder = Path(folder_path)
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_exts:
        image_files.extend(folder.glob(ext))
    
    if len(image_files) == 0:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images. Processing up to {max_images}...")
    image_files = sorted(image_files)[:max_images]
    
    k = k if k is not None else model_data['best_k']
    
    # Predict all
    print("\nPredictions:")
    print(f"{'Filename':<40s} {'Prediction':<20s} {'Confidence'}")
    print("-" * 75)
    
    for img_path in image_files:
        try:
            pred_class_name, _, confidence, _, _ = predict_single_image(img_path, model_data, k)
            print(f"{img_path.name:<40s} {pred_class_name:<20s} {confidence*100:5.1f}%")
        except Exception as e:
            print(f"{img_path.name:<40s} ERROR: {str(e)[:30]}")


def main():
    parser = argparse.ArgumentParser(
        description="Test kNN baseline model on unseen fruit images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('image_path', nargs='?', help='Path to image file')
    parser.add_argument('--folder', help='Path to folder with multiple images')
    parser.add_argument('--model', default='artefacts/knn_model.pkl', 
                       help='Path to saved kNN model')
    parser.add_argument('-k', type=int, help='Number of neighbors (uses best_k if not specified)')
    parser.add_argument('--top-n', type=int, default=5, help='Show top N predictions')
    parser.add_argument('--save', help='Save visualization to this path')
    parser.add_argument('--max-images', type=int, default=12, 
                       help='Maximum images to process in folder mode')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.folder:
        parser.error("Please provide either an image path or --folder")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("\nMake sure you've run the kNN training notebook first to generate the model.")
        sys.exit(1)
    
    print(f"Loading kNN model from {model_path}...")
    model_data = load_knn_model(model_path)
    print(f"✓ Model loaded")
    print(f"  - Classes: {len(model_data['class_names'])}")
    print(f"  - Best K: {model_data['best_k']}")
    print(f"  - Training samples: {len(model_data['y_train'])}")
    
    # Folder mode
    if args.folder:
        predict_folder(args.folder, model_data, k=args.k, max_images=args.max_images)
        return
    
    # Single image mode
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ Error: Image not found at {image_path}")
        sys.exit(1)
    
    visualize_prediction(image_path, model_data, k=args.k, 
                        show_top_n=args.top_n, save_path=args.save)


if __name__ == '__main__':
    main()
