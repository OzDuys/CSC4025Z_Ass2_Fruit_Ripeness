"""
Fruit Ripeness Classifier - Streamlit App
A simple web interface for classifying fruit ripeness using a trained CNN model.
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# Define your CNN architecture (must match your trained model)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),  # Assumes 224x224 input
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Class labels
class_names = [
    "Apple - Unripe",
    "Apple - Ripe", 
    "Apple - Rotten",
    "Banana - Unripe",
    "Banana - Ripe",
    "Banana - Rotten",
    "Orange - Unripe",
    "Orange - Ripe",
    "Orange - Rotten"
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    """Load the trained model (cached to avoid reloading)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=9)
    
    try:
        checkpoint = torch.load('baseline_cnn.pt', map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        st.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.warning("⚠️ Model file not found. Using untrained model for demo.")
    
    model.to(device)
    model.eval()
    return model, device


def classify_fruit(image, model, device):
    """Classify a fruit image"""
    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Get top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    results = [(class_names[idx], prob.item()) for idx, prob in zip(top3_idx, top3_prob)]
    
    return results


# Streamlit UI
st.set_page_config(
    page_title="Fruit Ripeness Classifier",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎🍌🍊 Fruit Ripeness Classifier")
st.write("Upload a photo of an apple, banana, or orange to classify it as unripe, ripe, or rotten.")

# Load model
model, device = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Take a photo with your phone or upload an existing image"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("Classification Results")
        
        # Classify the image
        with st.spinner("Classifying..."):
            results = classify_fruit(image, model, device)
        
        # Display results
        for i, (class_name, prob) in enumerate(results, 1):
            # Split class name for better formatting
            fruit_type = class_name.split(" - ")[0]
            ripeness = class_name.split(" - ")[1]
            
            # Color-code based on ripeness
            if ripeness == "Unripe":
                color = "🟢"
            elif ripeness == "Ripe":
                color = "🟡"
            else:  # Rotten
                color = "🔴"
            
            st.write(f"**{i}. {color} {class_name}**")
            st.progress(prob)
            st.write(f"{prob*100:.1f}% confidence")
            st.write("")

else:
    st.info("👆 Upload an image to get started!")
    
    # Instructions
    st.markdown("---")
    st.subheader("How to use on your phone:")
    st.markdown("""
    1. Open this app in your phone's browser
    2. Tap "Browse files" or "Choose an image"
    3. Select "Take Photo" or "Camera" to capture a fruit
    4. Wait for the classification results!
    """)

# Footer
st.markdown("---")
st.caption("Powered by PyTorch and Streamlit")