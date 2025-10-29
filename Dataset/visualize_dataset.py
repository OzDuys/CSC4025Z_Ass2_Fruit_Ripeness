import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Set random seed for reproducibility (optional, remove if you want truly random)
random.seed(42)

base_path = "Dataset/fruit_ripeness_dataset/archive (1)/dataset/dataset/test"
classes = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
    "unripe apple",
    "unripe banana",
    "unripe orange"
]

# Create a 3x9 grid of subplots (3 images per class, 9 classes)
fig, axes = plt.subplots(3, 9, figsize=(27, 9))  # Adjust figsize for horizontal layout

for j, cls in enumerate(classes):  # j for columns (classes)
    class_path = os.path.join(base_path, cls)
    # List all image files
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Select 3 random images
    selected = random.sample(images, 3)
    
    for i in range(3):  # i for rows (image index)
        img_path = os.path.join(class_path, selected[i])
        img = Image.open(img_path)
        axes[i, j].imshow(img)
        axes[i, j].set_title(f"{cls}", fontsize=10)
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
print("Dataset visualization saved as 'dataset_samples.png'")