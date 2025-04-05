import os
import torch
import open_clip
from PIL import Image
import numpy as np
import faiss

# Load CLIP model and define device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model.to(device)

# Correct directory paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Create model directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Function to Extract Image Features and Build FAISS Index ---
def extract_and_index():
    image_files = [f for f in os.listdir(DATA_DIR) if f.endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print("⚠️ No images found for indexing.")
        return

    image_features = []
    image_paths = []

    for file in image_files:
        image_path = os.path.join(DATA_DIR, file)
        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model.encode_image(image).cpu().numpy()
                image_features.append(feature)
                image_paths.append(image_path)
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

    if not image_features:
        print("⚠️ No valid features extracted.")
        return

    image_features = np.vstack(image_features).astype('float32')
    faiss.normalize_L2(image_features)

    # Create FAISS index
    dimension = image_features.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(image_features)

    # Save index and paths
    faiss.write_index(index, os.path.join(MODEL_DIR, "image_index.faiss"))
    np.save(os.path.join(MODEL_DIR, "image_paths.npy"), np.array(image_paths))
    print("✅ Feature extraction and indexing complete!")

if __name__ == "__main__":
    extract_and_index()
