"""Streamlit frontend dashboard for Seismic Fault Recognition API."""

import streamlit as st
import requests
from PIL import Image
import io
import time

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Seismic Fault Recognition",
    page_icon="🌍",
    layout="wide",
)

# --- Helper Functions ---
@st.cache_data(ttl=60)
def fetch_status():
    try:
        r = requests.get(f"{API_BASE_URL}/")
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        return None

@st.cache_data(ttl=60)
def fetch_models():
    try:
        r = requests.get(f"{API_BASE_URL}/models")
        return [m["name"] for m in r.json()] if r.status_code == 200 else []
    except requests.exceptions.ConnectionError:
        return []

@st.cache_data(ttl=60)
def fetch_recipes():
    try:
        r = requests.get(f"{API_BASE_URL}/recipes")
        return r.json() if r.status_code == 200 else []
    except requests.exceptions.ConnectionError:
        return []

# --- Sidebar Navigation ---
st.sidebar.title("🌍 Seismic AI Hub")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation",
    ["System Dashboard", "Run Inference", "Volume Slicer Viewer", "Interactive 3D Viewer"]
)

status = fetch_status()
if not status:
    st.sidebar.error("🔴 API Server Offline. Please start the FastAPI backend.")
else:
    st.sidebar.success(f"🟢 API Online (v{status.get('version')})")

st.sidebar.markdown("---")
st.sidebar.info("Developed with Streamlit & FastAPI")

# --- Page: System Dashboard ---
if app_mode == "System Dashboard":
    st.title("System Dashboard")
    st.markdown("Overview of the Seismic Fault Recognition platform and available resources.")

    if status:
        col1, col2, col3 = st.columns(3)
        col1.metric("API Status", status["status"].upper())
        col2.metric("Models Loaded", status["models_available"])
        col3.metric("Recipes Registered", status["recipes_available"])

        st.subheader("Neural Network Architectures")
        models = fetch_models()
        if models:
            for m in models:
                st.markdown(f"- `{m}`")

        st.subheader("Experiment Recipes")
        recipes = fetch_recipes()
        if recipes:
            st.dataframe(recipes, use_container_width=True)
    else:
        st.warning("Cannot connect to the backend server to fetch data.")

# --- Page: Run Inference ---
elif app_mode == "Run Inference":
    st.title("🚀 Run 3D Inference")
    st.markdown("Submit a 3D seismic volume for fault segmentation via the distributed task queue.")

    models = fetch_models()

    with st.form("inference_form"):
        st.subheader("Job Configuration")

        input_path = st.text_input("Input Volume Path (.npz, .npy, .zarr)", value="/path/to/data/test_volume.npz")
        checkpoint_path = st.text_input("Model Checkpoint Path (.pth)", value="/path/to/checkpoints/best_model.pth")

        col1, col2 = st.columns(2)
        with col1:
            selected_model = st.selectbox("Model Architecture", models if models else ["Offline"])
            overlap = st.slider("Sliding Window Overlap", min_value=0.0, max_value=0.75, value=0.25, step=0.05)
        with col2:
            st.markdown("Advanced Settings (Chunks & ROI)")
            chunk_size = st.text_input("Chunk Size (Z,Y,X)", value="256, 256, 256")
            roi_size = st.text_input("ROI Size (Z,Y,X)", value="128, 128, 128")

        submit_btn = st.form_submit_button("Submit Job")

    if submit_btn:
        if not status:
            st.error("Cannot submit. API is offline.")
        else:
            try:
                c_size = [int(x.strip()) for x in chunk_size.split(",")]
                r_size = [int(x.strip()) for x in roi_size.split(",")]

                payload = {
                    "input_path": input_path,
                    "model_variant": selected_model,
                    "checkpoint_path": checkpoint_path,
                    "chunk_size": c_size,
                    "roi_size": r_size,
                    "overlap": overlap
                }

                with st.spinner("Submitting to Celery queue..."):
                    resp = requests.post(f"{API_BASE_URL}/inference/submit", json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    st.success("Job successfully dispatched!")
                    st.json(data)
                    st.info(f"You can track this job ID: `{data['job_id']}`")
                else:
                    st.error(f"Error submitting job: {resp.text}")
            except Exception as e:
                st.error(f"Failed to parse inputs or connect: {e}")

# --- Page: Volume Slicer Viewer ---
elif app_mode == "Volume Slicer Viewer":
    st.title("👁️ Volume Slicer Viewer")
    st.markdown("Interactively extract and view 2D slices from massive 3D seismic volumes without loading them fully into RAM.")

    volume_path = st.text_input("Volume Path", value="")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Slice Controls")
        axis = st.radio("Slicing Axis", options=["Depth (Z)", "Inline (Y)", "Crossline (X)"])
        axis_map = {"Depth (Z)": "d", "Inline (Y)": "h", "Crossline (X)": "w"}

        index = st.number_input("Slice Index", min_value=0, value=50, step=1)

    with col2:
        st.subheader("Visualization")
        if volume_path:
            if not status:
                st.error("API is offline.")
            else:
                with st.spinner("Fetching slice from API..."):
                    try:
                        url = f"{API_BASE_URL}/volume/slice/{volume_path}/{axis_map[axis]}/{index}"

                        resp = requests.get(url)
                        if resp.status_code == 200:
                            image = Image.open(io.BytesIO(resp.content))
                            st.image(image, caption=f"Axis: {axis} | Index: {index}", use_container_width=True)
                        else:
                            st.error(f"Failed to fetch slice: {resp.json().get('detail', resp.text)}")
                    except Exception as e:
                        st.error(f"Error fetching image: {e}")
        elif not volume_path:
            st.info("Enter a valid local path to a .npz or .zarr file to view slices.")

# --- Page: Interactive 3D Viewer ---
elif app_mode == "Interactive 3D Viewer":
    st.title("🌐 Interactive 3D Viewer")
    st.markdown("Render a 3D point cloud of a segmented fault mask. Select a crop region if the file is too large.")
    import streamlit.components.v1 as components

    volume_path = st.text_input("Mask Volume Path (.npz, .npy, .zarr)", value="")

    st.subheader("Crop Region")
    st.markdown("Define the bounding box (start and end indices) to extract a sub-volume for rendering.")

    c1, c2, c3 = st.columns(3)
    with c1:
        z_start = st.number_input("Depth (Z) Start", value=0, step=64)
        z_end = st.number_input("Depth (Z) End", value=256, step=64)
    with c2:
        y_start = st.number_input("Inline (Y) Start", value=0, step=64)
        y_end = st.number_input("Inline (Y) End", value=256, step=64)
    with c3:
        x_start = st.number_input("Crossline (X) Start", value=0, step=64)
        x_end = st.number_input("Crossline (X) End", value=256, step=64)

    st.subheader("Rendering Settings")
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Probability Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
    with col2:
        max_points = st.number_input("Max Points to Render", min_value=10_000, max_value=500_000, value=100_000, step=10_000)

    if st.button("Render 3D Point Cloud"):
        if not volume_path:
            st.warning("Please enter a volume path.")
        elif not status:
            st.error("API is offline.")
        else:
            with st.spinner("Fetching data and generating 3D HTML..."):
                try:
                    # Construct URL with crop parameters
                    url = f"{API_BASE_URL}/volume/3d/{volume_path}?threshold={threshold}&max_points={max_points}&z_start={z_start}&z_end={z_end}&y_start={y_start}&y_end={y_end}&x_start={x_start}&x_end={x_end}"
                    resp = requests.get(url)

                    if resp.status_code == 200:
                        st.success("Render complete!")
                        # Display the HTML string directly in an iframe
                        components.html(resp.text, height=600, scrolling=True)
                    else:
                        st.error(f"Failed to render 3D view: {resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"Error connecting to 3D endpoint: {e}")
