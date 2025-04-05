import streamlit as st
import requests
import os
from PIL import Image
import base64

# --- Flask API Endpoint ---
API_URL_TEXT = "http://127.0.0.1:5000/search/text"
API_URL_IMAGE = "http://127.0.0.1:5000/search/image"

# --- Page Config ---
st.set_page_config(
    page_title="Visual Search with CLIP",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "num_results_to_show" not in st.session_state:
    st.session_state.num_results_to_show = 5  # Start with 5 results

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 8px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        border: 2px solid #4CAF50;
        border-radius: 10px;
    }
    .stFileUploader>div>div>div>button {
        background-color: #0073e6;
        color: white;
    }
    .download-btn a {
        color: #1E90FF;
        text-decoration: none;
        font-size: 14px;
        margin-top: 5px;
    }
    .download-btn a:hover {
        color: #00bfff;
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar Info ---
with st.sidebar:
    st.image("https://clipdrop.co/static/logo.png", width=100)
    st.title("💡 About Visual Search")
    st.write(
        "This Visual Search app uses **OpenAI CLIP** to find similar images based on text or image queries. "
        "It compares embeddings from the search input with a FAISS-indexed image database and Unsplash API."
    )
    st.write("🔗 [Learn More about CLIP](https://openai.com/research/clip)")

# --- Main Title ---
st.title("🔎 Visual Search with CLIP")
st.markdown("<p style='color: black; font-size: 16px;'>Find similar results using either text or an image!</p>", unsafe_allow_html=True)

# --- Tabs for Search ---
tab1, tab2 = st.tabs(["📄 Search with Text", "🖼️ Search with Image"])

# --- Function to Generate Download Link ---
def get_image_download_link(img_path, filename):
    """Returns a download link for local images."""
    try:
        with open(img_path, "rb") as img_file:
            img_data = img_file.read()
        b64 = base64.b64encode(img_data).decode()
        return f'<a href="data:file/jpg;base64,{b64}" download="{filename}" class="download-btn">📥 Download</a>'
    except Exception as e:
        return ""

# --- Display Results in Grid ---
def display_results(results):
    """Displays search results in a grid, handling both local and Unsplash images."""
    cols = st.columns(3)  # 3 columns layout
    for i, result in enumerate(results):
        img_path = result["path"]
        score = result["score"]

        # Detect if it's a URL (Unsplash) or local image
        is_unsplash = img_path.startswith("http")

        try:
            if is_unsplash:
                # Display Unsplash image directly from URL
                with cols[i % 3]:
                    st.image(img_path, caption=f"Source: Unsplash", use_container_width=True)
                    st.markdown(f'<a href="{img_path}" target="_blank">🌍 View on Unsplash</a>', unsafe_allow_html=True)
            else:
                # Local image handling
                if not os.path.isabs(img_path):
                    img_path = os.path.join(os.getcwd(), "app", "data", os.path.basename(img_path))

                image = Image.open(img_path)
                with cols[i % 3]:  # Display in columns
                    st.image(image, caption=f"Score: {score:.2f}", use_container_width=True)
                    st.markdown(get_image_download_link(img_path, os.path.basename(img_path)), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Error loading image: {img_path}. Error: {e}")

# --- Text Search ---
with tab1:
    query = st.text_input("💡 Enter a search query", placeholder="e.g., cat, beach, sunset")

    if st.button("🔍 Search with Text"):
        if query:
            with st.spinner("Searching... Please wait!"):
                try:
                    response = requests.post(API_URL_TEXT, json={"query": query})
                    if response.status_code == 200:
                        results = response.json().get("results", [])
                        if results:
                            st.success(f"✅ Found {len(results)} results!")
                            display_results(results)
                        else:
                            st.warning("⚠️ No results found!")
                    else:
                        st.error("❌ Error while searching. Check API response!")
                except Exception as e:
                    st.error(f"❌ Exception: {e}")

# --- Image Search ---
with tab2:
    uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file and st.button("🔎 Search with Image"):
        with st.spinner("Searching... Please wait!"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(API_URL_IMAGE, files=files)

                if response.status_code == 200:
                    results = response.json().get("results", [])
                    if results:
                        st.success(f"✅ Found {len(results)} results!")
                        display_results(results)
                    else:
                        st.warning("⚠️ No results found!")
                else:
                    st.error("❌ Error while searching. Check API response!")
            except Exception as e:
                st.error(f"❌ Exception: {e}")
