import requests

# Define API URLs
BASE_URL = "http://127.0.0.1:5000"

# --- Check Text Search API ---
def check_text_search():
    url = f"{BASE_URL}/search/text"
    data = {"query": "cat"}
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            print(f"✅ Text search successful! Found {len(results)} results.")
        else:
            print("⚠️ Text search returned no results.")
    else:
        print(f"❌ Text search API failed. Status: {response.status_code}, Error: {response.text}")

# --- Check Image Search API ---
def check_image_search():
    url = f"{BASE_URL}/search/image"
    image_path = "C:/Users/neha/Downloads/visual_search_project/app/data/sample1.jpg"  # Change to a valid image path
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            print(f"✅ Image search successful! Found {len(results)} results.")
        else:
            print("⚠️ Image search returned no results.")
    else:
        print(f"❌ Image search API failed. Status: {response.status_code}, Error: {response.text}")

if __name__ == "__main__":
    check_text_search()
    check_image_search()

import requests

url = "http://127.0.0.1:5000/search/text"
data = {"query": "cat"}

response = requests.post(url, json=data)
print(response.json())  # See if Unsplash results are included


