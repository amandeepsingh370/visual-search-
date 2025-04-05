import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
import torch
import open_clip
from PIL import Image

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model.to(device)

# Load Index and Image Paths
index_path = os.path.join(MODEL_DIR, "image_index.faiss")
paths_path = os.path.join(MODEL_DIR, "image_paths.npy")

if not os.path.exists(index_path) or not os.path.exists(paths_path):
    print("❌ Index or paths not found. Please re-run feature extraction.")
    exit()

index = faiss.read_index(index_path)
image_paths = np.load(paths_path)

# --- Manual Text Search ---
def search_text(query, top_k=5):
    with torch.no_grad():
        text_features = model.encode_text(open_clip.tokenize([query]).to(device)).cpu().numpy()

    faiss.normalize_L2(text_features)
    D, I = index.search(text_features.astype("float32"), top_k)

    results = [
        {"path": str(image_paths[i]), "score": float(D[0][j])}
        for j, i in enumerate(I[0]) if i >= 0 and D[0][j] > 0.1
    ]

    if results:
        print(f"✅ Found {len(results)} relevant results for query '{query}':")
        for res in results:
            print(f"{res['path']} (Score: {res['score']:.4f})")
    else:
        print(f"⚠️ No relevant results found for query '{query}'.")


# --- Test with a Text Query ---
search_text("cat")  # Change to any other relevant term


