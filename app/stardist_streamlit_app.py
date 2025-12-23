# ============================================
# Streamlit StarDist App — Interactive Microscopy Segmentation
# ============================================

import os, io, zipfile, urllib.request, tempfile, shutil, hashlib
import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image
import tifffile
from skimage import exposure, morphology, filters, transform, measure
from csbdeep.utils import normalize
from stardist.models import StarDist2D

# -----------------------
# Configuration
# -----------------------

# Require user upload (no private local paths)
DEFAULT_INPUT_PATH = None

# Store models inside repo
MODEL_BASEDIR = os.path.join(os.getcwd(), "models")
MODEL_NAME = "2D_versatile_fluo"
MODEL_ZIP_URL = (
    "https://github.com/stardist/stardist-models/releases/"
    "download/v0.1/python_2D_versatile_fluo.zip"
)
REQUIRED_FILES = {"config.json", "weights_best.h5"}

# -----------------------
# Utilities
# -----------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _find_model_root(search_dir):
    for root, _, files in os.walk(search_dir):
        if REQUIRED_FILES.issubset(set(files)):
            return root
    return None

def _copy_tree(src, dst):
    ensure_dir(dst)
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out_dir = os.path.join(dst, rel) if rel != "." else dst
        ensure_dir(out_dir)
        for f in files:
            shutil.copy2(os.path.join(root, f), os.path.join(out_dir, f))

# -----------------------
# Model loading
# -----------------------

@st.cache_resource(show_spinner=False)
def prepare_model(basedir: str, name: str, url: str) -> StarDist2D:
    final_dir = os.path.join(basedir, name)
    ensure_dir(final_dir)

    if not all(os.path.isfile(os.path.join(final_dir, f)) for f in REQUIRED_FILES):
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        with tempfile.TemporaryDirectory() as tmpdir:
            zpath = os.path.join(tmpdir, "model.zip")
            with open(zpath, "wb") as f:
                f.write(data)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(tmpdir)
            model_root = _find_model_root(tmpdir)
            if model_root is None:
                raise RuntimeError("StarDist model files not found after extraction.")
            _copy_tree(model_root, final_dir)

    return StarDist2D(None, name=name, basedir=basedir)

@st.cache_resource(show_spinner=False)
def get_model():
    return prepare_model(MODEL_BASEDIR, MODEL_NAME, MODEL_ZIP_URL)

# -----------------------
# Image loading
# -----------------------

@st.cache_data(show_spinner=False)
def read_image_from_bytes(b: bytes, filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".tif", ".tiff"]:
        arr = tifffile.imread(io.BytesIO(b))
    else:
        img = Image.open(io.BytesIO(b))
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        arr = np.array(img)
    return arr

def load_image(uploaded_file):
    if uploaded_file is None:
        return None, None
    raw = uploaded_file.read()
    img = read_image_from_bytes(raw, uploaded_file.name)
    return img, uploaded_file.name

# -----------------------
# Channels & preprocessing
# -----------------------

def select_channel(img: np.ndarray, mode: str):
    if img.ndim == 2:
        return img, "Gray"
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        means = [img[..., i].mean() for i in range(img.shape[-1])]
        mapping = {"Auto": int(np.argmax(means)), "Red": 0, "Green": 1, "Blue": 2}
        idx = mapping.get(mode, int(np.argmax(means)))
        names = ["Red", "Green", "Blue", "Alpha"]
        return img[..., idx], names[idx]
    a = img
    while a.ndim > 2:
        a = a.max(axis=0)
    return a, "Collapsed"

def preprocess_for_model(gray2d,
                         use_tophat=True, tophat_r=8,
                         use_clahe=True, clahe_clip=0.01,
                         gauss_sigma=0.7,
                         p_low=1, p_high=99.8):
    img = gray2d.astype(np.float32)
    if use_tophat:
        img = morphology.white_tophat(img, morphology.disk(tophat_r))
    if use_clahe:
        vmin, vmax = np.percentile(img, (p_low, p_high))
        img = np.clip((img - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        img = exposure.equalize_adapthist(img, clip_limit=clahe_clip).astype(np.float32)
    if gauss_sigma > 0:
        img = filters.gaussian(img, sigma=gauss_sigma, preserve_range=True)
    return normalize(img, p_low, p_high)

def make_green_overlay(labels, alpha=0.35):
    h, w = labels.shape
    ov = np.zeros((h, w, 4), dtype=np.float32)
    mask = labels > 0
    ov[..., 1] = mask.astype(np.float32)
    ov[..., 3] = mask.astype(np.float32) * alpha
    return ov

# -----------------------
# StarDist inference
# -----------------------

@st.cache_data(show_spinner=False)
def run_stardist(model_input, prob_thresh, nms_thresh, tiles):
    model = get_model()
    labels, _ = model.predict_instances(
        model_input,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=tiles
    )
    return labels.astype(np.int32)

def adaptive_tiles(h, w):
    m = max(h, w)
    if m <= 1024:
        return None
    if m <= 1536:
        return (2, 2)
    if m <= 3072:
        return (4, 4)
    return (6, 6)

# -----------------------
# App UI
# -----------------------

st.set_page_config(page_title="StarDist Microscopy Segmentation", layout="wide")
st.title("🔬 StarDist Segmentation — Streamlit App")

with st.sidebar:
    st.header("Image Input")
    uploader = st.file_uploader(
        "Upload microscopy image (.tif / .png / .jpg)",
        type=["tif", "tiff", "png", "jpg", "jpeg"]
    )
    st.caption("Upload an image to begin analysis.")

    st.header("Segmentation Parameters")
    channel = st.radio("Channel", ["Auto", "Red", "Green", "Blue"])
    prob = st.slider("Probability threshold", 0.0, 1.0, 0.5, 0.01)
    nms = st.slider("NMS threshold", 0.0, 1.0, 0.3, 0.01)

# -----------------------
# Main logic
# -----------------------

img, img_label = load_image(uploader)

if img is None:
    st.info("Please upload an image to start.")
    st.stop()

gray_raw, ch_name = select_channel(img, channel)
gray_proc = preprocess_for_model(gray_raw)

tiles = adaptive_tiles(*gray_proc.shape)
labels = run_stardist(gray_proc, prob, nms, tiles)

overlay = make_green_overlay(labels)
bg = (gray_raw - gray_raw.min()) / max(gray_raw.ptp(), 1e-6)

fig = px.imshow(bg, color_continuous_scale="gray", origin="upper")
fig.add_layout_image(
    dict(
        source=Image.fromarray((overlay * 255).astype(np.uint8), mode="RGBA"),
        x=0, y=0, sizex=bg.shape[1], sizey=bg.shape[0],
        xref="x", yref="y", sizing="stretch", layer="above"
    )
)
fig.update_layout(title=f"Segmentation overlay (channel: {ch_name})",
                  coloraxis_showscale=False)

st.plotly_chart(fig, use_container_width=True)

n_objects = labels.max()
st.metric("Detected objects", int(n_objects))
