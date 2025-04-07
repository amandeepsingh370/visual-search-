import torch
import open_clip
from PIL import Image
import os
import numpy as np
import faiss
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import threading
from flask import Flask, request, jsonify


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load CLIP model using open_clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model.to(device)

# Define directories
image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# --- Function to Build FAISS Index ---
def build_faiss_index():
    print("Rebuilding FAISS index...")
    image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print("⚠️ No images found in the data folder. Skipping re-indexing.")
        return

    image_features = []
    image_paths = []

    for file in image_files:
        image_path = os.path.join(image_dir, file)
        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model.encode_image(image)
            image_features.append(feature.cpu().numpy())
            image_paths.append(image_path)
        except Exception as e:
            print(f"⚠️ Error processing image {file}: {e}")
            continue

    if not image_features:
        print("⚠️ No valid images processed. Skipping index rebuild.")
        return

    # Create FAISS index for image features
    image_features = np.vstack(image_features).astype('float32')
    faiss.normalize_L2(image_features)
    dimension = image_features.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Use Inner Product (cosine similarity)
    index.add(image_features)

    # Save FAISS index and image paths
    os.makedirs(models_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(models_dir, "image_index.faiss"))
    np.save(os.path.join(models_dir, "image_paths.npy"), np.array(image_paths))
    print("✅ Index rebuilt and saved successfully!")

# --- File Change Handler to Trigger Rebuild ---
class ImageFolderHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith((".jpg", ".jpeg", ".png")):
            print(f"📸 New or modified image detected: {event.src_path}")
            build_faiss_index()

# --- Start Watchdog Observer ---
def start_observer():
    event_handler = ImageFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, path=image_dir, recursive=False)
    observer.start()
    print(f"👀 Watching '{image_dir}' for changes...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Initialize Flask app
app = Flask(__name__)
SIMILARITY_THRESHOLD = 0.85  # Increased for better filtering

@app.route("/search/text", methods=["POST"])
def search_text():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    with torch.no_grad():
        text_features = model.encode_text(open_clip.tokenize([query]).to(device)).cpu().numpy()
    faiss.normalize_L2(text_features)

    index_path = os.path.join(models_dir, "image_index.faiss")
    paths_path = os.path.join(models_dir, "image_paths.npy")

    if not os.path.exists(index_path) or not os.path.exists(paths_path):
        return jsonify({"error": "Index or paths not found. Please re-index the images."}), 500

    index = faiss.read_index(index_path)
    image_paths = np.load(paths_path)

    D, I = index.search(text_features.astype("float32"), 20)

    results = [
        {"path": str(image_paths[i]), "score": float(D[0][j])}
        for j, i in enumerate(I[0]) if i >= 0 and D[0][j] >= 0.1
    ]

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

        index_path = os.path.join(models_dir, "image_index.faiss")
        paths_path = os.path.join(models_dir, "image_paths.npy")

        if not os.path.exists(index_path) or not os.path.exists(paths_path):
            return jsonify({"error": "Index or paths not found. Please re-index the images."}), 500

        index = faiss.read_index(index_path)
        image_paths = np.load(paths_path)

        D, I = index.search(image_features.astype("float32"), 20)

        results = [
            {"path": str(image_paths[i]), "score": float(D[0][j])}
            for j, i in enumerate(I[0]) if i >= 0 and D[0][j] >= 0.1
        ]

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 500

# --- Main Entry Point ---
if __name__ == "__main__":
    build_faiss_index()
    threading.Thread(target=start_observer, daemon=True).start()
    print("🚀 Auto-reindexing enabled! Waiting for changes...")
    app.run(host="0.0.0.0", port=5000, debug=True)
