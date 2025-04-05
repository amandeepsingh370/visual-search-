import torch
import open_clip
import faiss
import numpy as np
import os
from PIL import Image

# Load CLIP model and FAISS index
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP Model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model.to(device)

# Define directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "image_index.faiss")
IMAGE_PATHS_PATH = os.path.join(BASE_DIR, "models", "image_paths.npy")

# Load FAISS index
if os.path.exists(MODEL_PATH):
    index = faiss.read_index(MODEL_PATH)
else:
    raise FileNotFoundError(f"❌ FAISS index not found at: {MODEL_PATH}")

# Load image paths
if os.path.exists(IMAGE_PATHS_PATH):
    image_paths = np.load(IMAGE_PATHS_PATH)
else:
    raise FileNotFoundError(f"❌ Image paths file not found at: {IMAGE_PATHS_PATH}")

# Normalize embeddings before search
def normalize_features(features):
    faiss.normalize_L2(features)

# Search by text query
def search_by_text(query, top_k=5):
    if not query:
        return []
    text = open_clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).cpu().numpy()
    normalize_features(text_features)
    D, I = index.search(text_features.astype("float32"), top_k)

    results = [{"path": str(image_paths[i]), "score": float(D[0][j])} for j, i in enumerate(I[0]) if i >= 0]
    return results

# Search by image
def search_by_image(image_path, top_k=5):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found at: {image_path}")

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy()
    normalize_features(image_features)
    D, I = index.search(image_features.astype('float32'), top_k)

    results = [{"path": str(image_paths[i]), "score": float(D[0][j])} for j, i in enumerate(I[0]) if i >= 0]
    return results
