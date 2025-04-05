import torch
import open_clip
from PIL import Image
import os
import numpy as np
import faiss
import requests  # To call Unsplash API
from flask import Flask, request, jsonify

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model.to(device)

# Define directories
image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Unsplash API Key (Replace with your actual key)
UNSPLASH_ACCESS_KEY = "dYqZKXdg44DD8SRg0HN7nS2mJfWJn99TM5ta_SQ4QiE"

# --- Function to Query Unsplash ---
def fetch_images_from_unsplash(query, num_results=10):
    url = f"https://api.unsplash.com/search/photos"
    params = {"query": query, "client_id": UNSPLASH_ACCESS_KEY, "per_page": num_results}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error if request fails
        data = response.json()
        return [{"path": item["urls"]["regular"], "score": None} for item in data.get("results", [])]
    except Exception as e:
        print(f"❌ Unsplash API Error: {e}")
        return []

# --- Function to Search FAISS Index ---
def search_faiss(query_features, num_results=20):
    index_path = os.path.join(models_dir, "image_index.faiss")
    paths_path = os.path.join(models_dir, "image_paths.npy")

    if not os.path.exists(index_path) or not os.path.exists(paths_path):
        return []  # No FAISS index found

    index = faiss.read_index(index_path)
    image_paths = np.load(paths_path)

    # Perform the search
    D, I = index.search(query_features.astype("float32"), num_results)

    # Filter results based on similarity threshold
    return [{"path": str(image_paths[i]), "score": float(D[0][j])} for j, i in enumerate(I[0]) if i >= 0 and D[0][j] >= 0.1]

# Initialize Flask app
app = Flask(__name__)

@app.route("/search/text", methods=["POST"])
def search_text():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    with torch.no_grad():
        text_features = model.encode_text(open_clip.tokenize([query]).to(device)).cpu().numpy()
    faiss.normalize_L2(text_features)

    # Search locally first
    results = search_faiss(text_features, num_results=20)

    # If not enough results, fetch from Unsplash
    if len(results) < 5:
        unsplash_results = fetch_images_from_unsplash(query, num_results=10)
        results.extend(unsplash_results)

    return jsonify({"results": results})

@app.route("/search/image", methods=["POST"])
def search_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(file).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor).cpu().numpy()

        faiss.normalize_L2(image_features)

        # Search locally first
        results = search_faiss(image_features, num_results=20)

        # If not enough results, fetch from Unsplash (fallback to generic search term)
        if len(results) < 5:
            unsplash_results = fetch_images_from_unsplash("random", num_results=10)
            results.extend(unsplash_results)

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

import requests

url = "http://127.0.0.1:5000/search/text"
data = {"query": "cat"}

response = requests.post(url, json=data)
print(response.json())  # See if Unsplash results are included