import os
import torch
import streamlit as st
from paint_core.segmentation import SegmentationEngine, sam_model_registry
from app_config.constants import PerformanceConfig

# --- CONSTANTS ---
MODEL_TYPE = PerformanceConfig.SAM_MODEL_TYPE
CHECKPOINT_PATH = PerformanceConfig.SAM_CHECKPOINT_PATH

def _load_sam_model(path, type_name):
    """Load the model weights from disk. No caching."""
    if not os.path.exists(path):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry[type_name](checkpoint=path)
    model.to(device=device)

    # CPU quantization for speed
    if device == "cpu":
        try:
            print(f"DEBUG: Applying dynamic quantization to {type_name} model...")
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            print("DEBUG: Quantization complete.")
        except Exception as e:
            print(f"WARNING: Quantization failed: {e}")

    return model

def get_sam_engine(checkpoint_path=CHECKPOINT_PATH, model_type=MODEL_TYPE):
    """
    Get the SAM engine stored in session state.
    Model weights are loaded once per session and stored in session_state.
    The engine (which holds image embeddings) is per-session and never cached globally.
    """
    # Load model weights once per session (heavy but unavoidable)
    if "sam_model" not in st.session_state or st.session_state["sam_model"] is None:
        model = _load_sam_model(checkpoint_path, model_type)
        st.session_state["sam_model"] = model

    model = st.session_state["sam_model"]
    if model is None:
        return None

    # Always create a fresh engine per session
    if "sam_engine" not in st.session_state or st.session_state["sam_engine"] is None:
        from paint_core.segmentation import SegmentationEngine
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.session_state["sam_engine"] = SegmentationEngine(model_instance=model, device=device)

    return st.session_state["sam_engine"]

def ensure_model_exists():
    """Download weights automatically if missing."""
    import requests
    import time
    if not os.path.exists(CHECKPOINT_PATH):
        with st.status("⚠️ AI model not found. Downloading automatically...", expanded=True) as status:
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            try:
                progress_bar = st.progress(0)
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 512 * 1024
                with open(CHECKPOINT_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = min(1.0, downloaded / total_size)
                                progress_bar.progress(percent)
                            status.write(f"📥 Downloaded {downloaded//(1024*1024)}MB...")

                if os.path.getsize(CHECKPOINT_PATH) < 35 * 1024 * 1024:
                    status.update(label="❌ Download incomplete or corrupt.", state="error")
                    os.remove(CHECKPOINT_PATH)
                    st.stop()

                status.update(label="✅ Model weights verified!", state="complete")
                time.sleep(1)
                # Reset engine so it reloads from new weights
                for key in ["sam_model", "sam_engine"]:
                    if key in st.session_state:
                        del st.session_state[key]
                from paint_utils.state_manager import preserve_sidebar_state
                preserve_sidebar_state()
                st.rerun()
            except Exception as e:
                status.update(label=f"❌ Failed to download model: {e}", state="error")
                if os.path.exists(CHECKPOINT_PATH): os.remove(CHECKPOINT_PATH)
                st.stop()
