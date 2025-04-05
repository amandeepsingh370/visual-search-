import faiss
import numpy as np

index_path = "models/image_index.faiss"
paths_path = "models/image_paths.npy"

index = faiss.read_index(index_path)
image_paths = np.load(paths_path)

print(f"Total images indexed: {len(image_paths)}")
print("First 5 images:", image_paths[:5])


import requests

UNSPLASH_ACCESS_KEY = "dYqZKXdg44DD8SRg0HN7nS2mJfWJn99TM5ta_SQ4QiE"
query = "cats"
url = f"https://api.unsplash.com/search/photos?query={query}&per_page=5&client_id={UNSPLASH_ACCESS_KEY}"

response = requests.get(url)
print(response.json())
