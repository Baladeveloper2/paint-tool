import cv2
import numpy as np
import streamlit as st
from scipy import sparse
from app_config.constants import ColorizerConfig

# ─────────────────────────────────────────────────────────────────────────────
# PER-IMAGE SCENE PRIORS (cached once, reused per click)
# ─────────────────────────────────────────────────────────────────────────────
def _get_scene_priors(image_rgb: np.ndarray):
    """
    Returns (img_lab, L_base, L_detail, gray) for image_rgb.
    Uses Edge-Aware Guided Filtering for Intrinsic Image Decomposition.
    """
    h, w = image_rgb.shape[:2]
    key_id  = id(image_rgb)
    key_dim = (h, w)

    if (st.session_state.get("_scene_id")  == key_id and
            st.session_state.get("_scene_dim") == key_dim and
            "_scene_priors" in st.session_state):
        return st.session_state["_scene_priors"]

    # 1. Convert to LAB float32 (L: 0-100, A/B: -128 to 127)
    img_f = image_rgb.astype(np.float32) / 255.0
    img_lab = cv2.cvtColor(img_f, cv2.COLOR_RGB2Lab)
    
    L_channel = img_lab[:, :, 0]
    
    # 2. Guided Filter for Intrinsic Decomposition (Base/Detail separation)
    # The base contains illumination (lighting, shadows), detail contains texture (plaster, brick).
    if hasattr(cv2, 'ximgproc'):
        L_base = cv2.ximgproc.guidedFilter(
            guide=np.uint8(L_channel * 2.55), 
            src=np.uint8(L_channel * 2.55), 
            radius=15, eps=100
        ).astype(np.float32) / 2.55
    else:
        # Fallback to bilateral if ximgproc is missing
        L_base = cv2.bilateralFilter(L_channel, 15, 20, 20)
        
    L_detail = L_channel - L_base
    
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    priors = (img_lab, L_base, L_detail, gray)
    st.session_state["_scene_priors"] = priors
    st.session_state["_scene_id"]     = key_id
    st.session_state["_scene_dim"]    = key_dim

    # Invalidate layer composite cache whenever base image changes
    st.session_state["_layer_cache"]     = None
    st.session_state["_layer_cache_len"] = 0
    st.session_state["_layer_cache_lid"] = None

    return priors

# ─────────────────────────────────────────────────────────────────────────────
class ColorTransferEngine:

    @staticmethod
    def hex_to_rgb(hex_color: str):
        h = hex_color.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    @st.cache_data
    def get_target_lab(color_hex: str):
        """Cached full LAB L,a,b for a hex color."""
        rgb = ColorTransferEngine.hex_to_rgb(color_hex)
        px  = np.array([[[*rgb]]], dtype=np.uint8)
        lab = cv2.cvtColor(px.astype(np.float32)/255.0, cv2.COLOR_RGB2Lab)
        return float(lab[0, 0, 0]), float(lab[0, 0, 1]), float(lab[0, 0, 2])

    @staticmethod
    def apply_color(image_rgb, mask, target_color_hex,
                    intensity=1.0, seed_point=None, use_adaptive=False):
        return ColorTransferEngine.composite_multiple_layers(
            image_rgb,
            [{'mask': mask, 'color': target_color_hex}]
        )

    @staticmethod
    def composite_multiple_layers(image_rgb: np.ndarray, masks_data: list) -> np.ndarray:
        """
        True Physically Realistic Paint Rendering using LAB Intrinsic Image Decomposition.
        Implements Albedo Replacement and Global Illumination Normalization.
        """
        if not masks_data:
            return image_rgb.copy()

        h, w = image_rgb.shape[:2]

        # ── Scene priors (cached per image) ─────────────────────────────────
        img_lab, L_base, L_detail, gray_guide = _get_scene_priors(image_rgb)

        # We DO NOT accumulate colors! Every layer replaces the albedo directly from the RAW image.
        # This prevents color stacking/mutation when repainting.
        curr_lab = img_lab.copy()

        # ── Composite each layer ──────────────────────────────────────────────
        for data in masks_data:
            mask      = data.get('mask')
            color_hex = data.get('color')
            if mask is None or not color_hex:
                continue

            if sparse.issparse(mask):
                mask = mask.toarray()

            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

            mask_bin = (mask > 0).astype(np.float32)

            # Refinement
            refinement = data.get('refinement', 0)
            if refinement != 0:
                k  = abs(refinement) * 2 + 1
                rk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mask_bin = (cv2.dilate(mask_bin, rk) if refinement > 0 else cv2.erode(mask_bin, rk))

            # ── Edge-Aware Poisson-style Alpha Mask ──────────────────────────────
            user_soft = data.get('softness', 0)
            if hasattr(cv2, 'ximgproc'):
                radius = 5 + (user_soft * 3)
                mask_soft = cv2.ximgproc.guidedFilter(
                    guide=gray_guide, 
                    src=mask_bin, 
                    radius=radius, eps=10.0
                )
            else:
                blur_sigma = max(1, user_soft * 2 + 2)
                mask_soft = cv2.GaussianBlur(mask_bin, (blur_sigma*2+1, blur_sigma*2+1), blur_sigma)
                
            mask_soft = np.clip(mask_soft, 0.0, 1.0)
            mask_soft_3ch = mask_soft[:, :, np.newaxis]

            # ── True Color Transfer (Intrinsic Decomposition) ────────────────────
            tgt_L, tgt_A, tgt_B = ColorTransferEngine.get_target_lab(color_hex)
            
            # Global Illumination Normalization
            # Instead of anchoring to the 80th percentile of the whole image,
            # we anchor to the median illumination of ONLY the painted wall.
            # This ensures the wall receives ONE CONTINUOUS uniform shade.
            wall_pixels = L_base[mask_bin > 0]
            if len(wall_pixels) > 0:
                wall_anchor = float(np.median(wall_pixels))
            else:
                wall_anchor = float(np.percentile(L_base, 80))
            
            # The l_shift represents the natural lighting variation (shadows, highlights)
            l_shift = L_base - wall_anchor
            
            # Normalize the wall luminance tightly to the target color
            # We compress the lighting variance slightly so shadows don't break the color
            compressed_shift = np.where(l_shift < 0, l_shift * 0.7, l_shift)
            new_L_base = tgt_L + compressed_shift
            
            # ── Apply Finishes ──────────────────────────────────────────────────
            finish = data.get('finish', 'Standard')
            mod_detail = L_detail.copy()
            
            if finish == 'Matte':
                mod_detail *= 0.6
                new_L_base = np.where(new_L_base > 85, 85 + (new_L_base - 85)*0.3, new_L_base)
            elif finish == 'Gloss':
                mod_detail *= 1.3
                new_L_base = np.where(new_L_base > 75, new_L_base * 1.1, new_L_base)
            elif finish == 'Satin':
                mod_detail *= 0.9

            final_L = np.clip(new_L_base + mod_detail, 0.0, 100.0)
            
            # TRUE ALBEDO REPLACEMENT: We replace A and B entirely, removing the old wall color
            final_A = np.full_like(curr_lab[:, :, 1], tgt_A)
            final_B = np.full_like(curr_lab[:, :, 2], tgt_B)
            
            painted_lab = np.stack([final_L, final_A, final_B], axis=-1)

            # ── Absolute Overwrite Blending ──────────────────────────────────────
            # Because we operate on img_lab directly, when we apply mask_soft_3ch, 
            # we seamlessly overwrite any previous paint layers with the new paint layer 
            # while feathering the edges flawlessly into the original background!
            curr_lab = (painted_lab * mask_soft_3ch) + (curr_lab * (1.0 - mask_soft_3ch))

        # Convert back to RGB
        final_rgb_f = cv2.cvtColor(curr_lab, cv2.COLOR_Lab2RGB)
        final_rgb = np.clip(final_rgb_f * 255.0, 0, 255).astype(np.uint8)

        return final_rgb
