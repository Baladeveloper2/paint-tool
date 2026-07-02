"""
Performance utilities for image processing.
Cache-free version: all cache logic has been removed.
"""

import numpy as np
import gc
from typing import Optional, List, Dict, Any
from app_config.constants import MAX_IMAGE_DIMENSION


def optimize_mask_storage(masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return masks unchanged (sparse compression is handled elsewhere)."""
    return masks


def estimate_memory_usage() -> Dict[str, float]:
    """Estimate current memory usage of session state."""
    import streamlit as st
    usage = {"images": 0.0, "masks": 0.0, "total": 0.0}

    for key in ["image", "image_original"]:
        if key in st.session_state and st.session_state[key] is not None:
            img = st.session_state[key]
            if isinstance(img, np.ndarray):
                usage["images"] += img.nbytes / (1024 * 1024)

    if "masks" in st.session_state:
        from scipy import sparse
        for mask_data in st.session_state["masks"]:
            if "mask" in mask_data and mask_data["mask"] is not None:
                m = mask_data["mask"]
                if sparse.issparse(m):
                    usage["masks"] += (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / (1024 * 1024)
                else:
                    usage["masks"] += m.nbytes / (1024 * 1024)

    usage["total"] = usage["images"] + usage["masks"]
    return usage


def resize_image_smart(image: np.ndarray, max_dim: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """
    Intelligently resize image with aspect ratio preservation.
    Only resizes if image exceeds max dimension.
    """
    import cv2

    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    if h > w:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    else:
        new_w = max_dim
        new_h = int(h * (max_dim / w))

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
