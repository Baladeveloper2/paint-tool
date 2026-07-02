import cv2
import numpy as np
from PIL import Image
import streamlit as st
from scipy import sparse
from app_config.constants import ColorizerConfig

# Try to import adaptive processing, but make it optional
try:
    from .adaptive_processing import (
        get_adaptive_blur_kernel,
        apply_bilateral_blur,
        classify_object,
        get_object_params,
        ObjectType
    )
    ADAPTIVE_AVAILABLE = True
except ImportError as e:
    # Fallback if adaptive module not available
    ADAPTIVE_AVAILABLE = False
    import logging
    logging.warning(f"Adaptive processing not available: {e}")

class ColorTransferEngine:
    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert HEX string to RGB tuple.
        
        Args:
            hex_color: Hex color string (e.g., '#FF0000' or 'FF0000')
            
        Returns:
            Tuple[int, int, int]: RGB values (0-255)
            
        Raises:
            ValueError: If hex_color is invalid
        """
        if not isinstance(hex_color, str):
            raise ValueError(f"hex_color must be a string, got {type(hex_color)}")
        
        hex_color = hex_color.lstrip('#')
        
        if len(hex_color) != 6:
            raise ValueError(f"hex_color must be  6 characters (got {len(hex_color)}): {hex_color}")
        
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError as e:
            raise ValueError(f"Invalid hex color '{hex_color}': {e}")

    @staticmethod
    def apply_color(image_rgb, mask, target_color_hex, intensity=1.0, seed_point=None, use_adaptive=False):
        """
        Apply color with optional adaptive processing (DISABLED by default).
        
        Args:
            image_rgb: NumPy array (H, W, 3) in RGB format
            mask: Boolean or uint8 mask array (H, W)
            target_color_hex: Target color as hex string
            intensity: Blending intensity (0.0-1.0)
            seed_point: Optional (x, y) click point for object classification
            use_adaptive: If True, uses adaptive blur (slower, experimental)
            
        Returns:
            NumPy array: Colored image
            
        Raises:
            ValueError: If inputs are invalid
            
        Note:
            Adaptive processing adds edge detection and object classification,
            which is slower. Default legacy mode is optimized for speed
            and texture preservation.
        """
        # Input validation
        if not isinstance(image_rgb, np.ndarray):
            raise ValueError(f"image_rgb must be numpy array, got {type(image_rgb)}")
        
        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"image_rgb must be (H, W, 3), got {image_rgb.shape}")
        
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"mask must be numpy array, got {type(mask)}")
        
        if len(mask.shape) != 2:
            raise ValueError(f"mask must be 2D array, got shape {mask.shape}")
        
        if mask.shape[:2] != image_rgb.shape[:2]:
            raise ValueError(f"mask shape {mask.shape} doesn't match image {image_rgb.shape[:2]}")
        
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"intensity must be between 0 and 1, got {intensity}")
        
        # Ensure input is standard format
        image_rgb = image_rgb.astype(np.uint8)
        
        # 1. Adaptive Blur Selection
        mask_float = mask.astype(np.float32)
        
        if use_adaptive and ADAPTIVE_AVAILABLE:
            try:
                # Detect optimal blur kernel based on edge density
                blur_kernel = get_adaptive_blur_kernel(mask, image_rgb)
                
                # Optionally detect if texture preservation needed
                if seed_point is not None:
                    try:
                        obj_type = classify_object(mask, image_rgb, seed_point)
                        params = get_object_params(obj_type)
                        
                        if params['use_bilateral']:
                            # Use bilateral filter for textured surfaces
                            mask_soft = apply_bilateral_blur(mask_float, preserve_edges=True)
                        else:
                            # Standard Gaussian blur
                            mask_soft = cv2.GaussianBlur(mask_float, blur_kernel, 0)
                    except Exception:
                        # Fallback to adaptive kernel if classification fails
                        mask_soft = cv2.GaussianBlur(mask_float, blur_kernel, 0)
                else:
                    # No seed point - use adaptive kernel only
                    mask_soft = cv2.GaussianBlur(mask_float, blur_kernel, 0)
            except Exception as e:
                # Fallback to legacy blur on any error
                import logging
                logging.warning(f"Adaptive blur failed, using legacy: {e}")
                mask_soft = cv2.GaussianBlur(mask_float, ColorizerConfig.BLUR_KERNEL_SIZE, 0)
        else:
            # Legacy mode: fixed blur
            mask_soft = cv2.GaussianBlur(mask_float, ColorizerConfig.BLUR_KERNEL_SIZE, 0)
        
        mask_3ch = np.stack([mask_soft] * 3, axis=-1)
        
        # 2. Prepare Target Color
        target_rgb = ColorTransferEngine.hex_to_rgb(target_color_hex)
        
        # 3. LAB Color Transfer with Caching
        img_float = image_rgb.astype(np.float32) / 255.0
        
        # Cache LAB conversion if possible
        cache_key = f"lab_conversion_{id(image_rgb)}"
        if cache_key in st.session_state:
            img_lab = st.session_state[cache_key]
        else:
            img_lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)
            # Cache for reuse (helps with multiple layers)
            st.session_state[cache_key] = img_lab
        
        L, A, B = cv2.split(img_lab)
        
        # Target color in LAB (cached via get_target_lab)
        target_L, target_a, target_b = ColorTransferEngine.get_target_lab(target_color_hex)
        
        # Apply Smart Luminance Shift (matches composite_multiple_layers logic)
        mask_float = mask.astype(np.float32)
        valid_mask = mask_float > 0.1
        if np.any(valid_mask):
            ref_L = np.mean(L[valid_mask])
        else:
            ref_L = 75.0
        ref_L = np.clip(ref_L, 10.0, 95.0)
        
        new_L = np.clip(L * (target_L / ref_L), 0, 100)
        
        # Swap A/B channels and use shifted L
        new_lab = cv2.merge([new_L, np.full_like(A, target_a), np.full_like(B, target_b)])
        recolored_rgb = cv2.cvtColor(new_lab, cv2.COLOR_Lab2RGB)
        
        # 4. Blend based on mask
        result_float = (recolored_rgb * mask_3ch) + (img_float * (1.0 - mask_3ch))
        
        result_uint8 = np.clip(result_float * 255.0, 0, 255).astype(np.uint8)
        
        return result_uint8

    @staticmethod
    @st.cache_data
    def get_target_lab(color_hex):
        """Pre-calculate and cache the LAB L/A/B channels for a hex color."""
        rgb = ColorTransferEngine.hex_to_rgb(color_hex)
        pixel = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.uint8)
        lab = cv2.cvtColor(pixel.astype(np.float32)/255.0, cv2.COLOR_RGB2Lab)
        return float(lab[0, 0, 0]), float(lab[0, 0, 1]), float(lab[0, 0, 2])

    @staticmethod
    def composite_multiple_layers(image_rgb, masks_data):
        """
        ULTRA-STABLE Single-Pass Compositor with Smart Caching.
        
        Optimized to handle 'Add Layer' operations incrementally.
        only re-calculates the new layer on top of the cached previous state.
        """
        if not masks_data:
            return image_rgb.copy()

        h, w = image_rgb.shape[:2]
        
        # --- CACHING LOGIC ---
        # We need to decide: Start from scratch OR Start from cached state?
        
        # 1. Access Base LAB (Always needed for L-channel reference)
        l_cache_key = "global_base_lab"
        
        if (l_cache_key not in st.session_state or 
            st.session_state.get("lab_cache_id") != id(image_rgb) or
            st.session_state.get("lab_cache_dim") != (h, w)):
            
            img_f = image_rgb.astype(np.float32, copy=False) / 255.0
            img_lab = cv2.cvtColor(img_f, cv2.COLOR_RGB2Lab)
            L, A, B = cv2.split(img_lab)
            st.session_state[l_cache_key] = (L, A, B)
            st.session_state["lab_cache_id"] = id(image_rgb)
            st.session_state["lab_cache_dim"] = (h, w)
            
            # Reset composite cache if base image changed
            st.session_state["comp_cache_state"] = None
            st.session_state["comp_cache_len"] = 0
            st.session_state["comp_cache_last_id"] = None
        
        # Load Base
        base_L, base_A, base_B = st.session_state[l_cache_key]
        
        # 2. Check for Incremental Update
        cached_state = st.session_state.get("comp_cache_state")
        cached_len = st.session_state.get("comp_cache_len", 0)
        
        start_index = 0
        curr_A = base_A.copy()
        curr_B = base_B.copy()
        curr_L_mod = base_L.copy() 

        can_use_cache = False
        
        if cached_state is not None and len(masks_data) > cached_len:
            if cached_len > 0:
                last_cached_mask = masks_data[cached_len-1]
                if id(last_cached_mask) == st.session_state.get("comp_cache_last_id"):
                     can_use_cache = True
            else:
                can_use_cache = True
        
        if can_use_cache:
            c_L, c_A, c_B = cached_state
            curr_L_mod = c_L.copy()
            curr_A = c_A.copy()
            curr_B = c_B.copy()
            start_index = cached_len

        # 3. Cumulative A/B/L Blending
        for i in range(start_index, len(masks_data)):
            data = masks_data[i]
            mask = data['mask']
            color_hex = data.get('color')
            if not color_hex: continue
            
            # ⚡ MEMORY OPTIMIZATION: Decompress if sparse
            if sparse.issparse(mask):
                mask = mask.toarray()
            
            target_L, target_a, target_b = ColorTransferEngine.get_target_lab(color_hex)
            
            # Robust preparation
            if mask.shape[:2] != (h, w):
                mask_uint8 = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask_f = mask_uint8.astype(np.float32)
            else:
                mask_f = mask.astype(np.float32)

            if mask_f.max() > 1.0: mask_f /= 255.0
            
            # BLUR / REFINEMENT
            refinement = data.get('refinement', 0)
            if refinement != 0:
                k_size = abs(refinement) * 2 + 1
                refine_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                if refinement > 0: mask_f = cv2.dilate(mask_f, refine_kernel)
                else: mask_f = cv2.erode(mask_f, refine_kernel)

            user_soft = data.get('softness', 0)
            if user_soft > 0:
                # Scaled to be more intuitive: 1=5x5, 2=11x11, 3=17x17...
                # This makes '2' a significantly smoother blend as requested.
                k_size = (user_soft * 6) - 1
                blur_val = (k_size, k_size)
            else:
                # Default (0) is now a crisp 3x3 blur for sharper architectural lines
                blur_val = (3, 3)
            
            kernel = np.ones(ColorizerConfig.DILATION_KERNEL_SIZE, np.uint8)
            mask_dilated = cv2.dilate(mask_f, kernel, iterations=ColorizerConfig.DILATION_ITERATIONS)
            mask_soft = cv2.GaussianBlur(mask_dilated, blur_val, 0)
            
            # --- SMART LUMINANCE FINISH ---
            # All finishes now respect target_L to prevent color mismatch.
            finish = data.get('finish', 'Standard')
            
            # Reference L for normalization
            # Calculate the actual average lightness of the masked area
            # This ensures that dark objects and light objects both get painted
            # to the EXACT same target lightness.
            valid_mask = mask_f > 0.1
            if np.any(valid_mask):
                ref_L = np.mean(base_L[valid_mask])
            else:
                ref_L = 75.0
            ref_L = np.clip(ref_L, 10.0, 95.0)
            
            if finish == 'Matte':
                # Flatter lighting, very opaque feel, matches target_L closely
                layer_L = np.clip(target_L + (base_L - ref_L) * 0.4, 0, 100)
            elif finish == 'Gloss':
                # High contrast highlights, keeps deep shadows, punchy
                layer_L = np.clip(target_L + (base_L - ref_L) * 1.5, 0, 100)
            elif finish == 'Satin':
                # Medium contrast, classic paint look
                layer_L = np.clip(target_L + (base_L - ref_L) * 1.1, 0, 100)
            elif finish == 'Texture':
                # Multiply approach: Best for rough surfaces (bricks, stones)
                layer_L = np.clip(base_L * (target_L / ref_L), 0, 100)
            else: # 'Standard'
                # HYBRID BLENDING: Multiply for dark colors, Shift for light colors.
                # This prevents 'black hole' effects for dark paints while keeping
                # exact color matches for lighter paints.
                
                # Calculate weights (darker target = more multiply)
                # target_L is 0-100.
                mult_weight = np.clip((70.0 - target_L) / 50.0, 0, 1.0)
                
                L_mult = base_L * (target_L / ref_L)
                L_shift = target_L + (base_L - ref_L) * 0.9
                
                layer_L = np.clip(L_mult * mult_weight + L_shift * (1.0 - mult_weight), 0, 100)
            
            curr_L_mod = (layer_L * mask_soft) + (curr_L_mod * (1.0 - mask_soft))
            curr_A = (target_a * mask_soft) + (curr_A * (1.0 - mask_soft))
            curr_B = (target_b * mask_soft) + (curr_B * (1.0 - mask_soft))

        # 4. Save Cache
        st.session_state["comp_cache_state"] = (curr_L_mod.copy(), curr_A.copy(), curr_B.copy())
        st.session_state["comp_cache_len"] = len(masks_data)
        if masks_data:
            st.session_state["comp_cache_last_id"] = id(masks_data[-1])
        else:
            st.session_state["comp_cache_last_id"] = None

        # 5. Final Conversion & Safety Check
        # Enforce consistent depth (float32) and size to prevent cv2.merge crashes
        curr_L_mod = curr_L_mod.astype(np.float32, copy=False)
        curr_A = curr_A.astype(np.float32, copy=False)
        curr_B = curr_B.astype(np.float32, copy=False)
        
        final_lab = cv2.merge([curr_L_mod, curr_A, curr_B])
        final_rgb = cv2.cvtColor(final_lab, cv2.COLOR_Lab2RGB)
        
        return np.clip(final_rgb * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_texture(image_rgb, mask, texture_rgb, opacity=0.8):
        """
        Apply a texture with blending to simulate surface material.
        """
        image_rgb = image_rgb.astype(np.uint8)
        
        # 1. Create Smooth Mask
        mask_float = mask.astype(np.float32)
        mask_soft = cv2.GaussianBlur(mask_float, ColorizerConfig.BLUR_KERNEL_SIZE, 0)
        mask_3ch = np.stack([mask_soft] * 3, axis=-1)
        
        # 2. Tile Texture to fill image
        h, w, c = image_rgb.shape
        th, tw, tc = texture_rgb.shape
        
        # Resize texture if too large to keep pattern visible
        if max(th, tw) > ColorizerConfig.MAX_TEXTURE_SIZE:
            scale = ColorizerConfig.MAX_TEXTURE_SIZE / max(th, tw)
            texture_rgb = cv2.resize(texture_rgb, (0, 0), fx=scale, fy=scale)
            th, tw, tc = texture_rgb.shape
            
        tiled_texture = np.zeros_like(image_rgb)
        
        for i in range(0, h, th):
            for j in range(0, w, tw):
                # Calculate available space
                curr_h = min(th, h - i)
                curr_w = min(tw, w - j)
                tiled_texture[i:i+curr_h, j:j+curr_w] = texture_rgb[:curr_h, :curr_w]
                
        # 3. Blend Texture (Multiply/Overlay approach)
        # Simple Approach: Multiply original L with Texture
        
        img_float = image_rgb.astype(np.float32) / 255.0
        tex_float = tiled_texture.astype(np.float32) / 255.0
        
        # Luminosity preservation:
        # Result = Texture * Original_Luminance
        # This makes the texture look shadowed by the room's lighting.
        
        img_lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)
        L, A, B = cv2.split(img_lab)
        
        # Blend: use texture color but keep original lightness structure
        # Optionally mix texture's own lightness with original lightness
        
        # Simplified: Alpha Blend texture over image, but modulated by mask
        # To make it look "on the wall", we ideally want:
        # Out = Texture * (Original_Gray) * 2.0 (Overlay-ish)
        
        gray = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)
        gray_3ch = np.stack([gray] * 3, axis=-1)
        
        # Hard Light / Multiply simulation
        blended = tex_float * gray_3ch * ColorizerConfig.TEXTURE_BRIGHTNESS_BOOST # Boost brightness slightly
        
        blended = np.clip(blended, 0, 1.0)
        
        # 4. Composite
        # Result = (Blended * Mask * Opacity) + (Original * (1 - Mask*Opacity))
        
        final_mask = mask_3ch * opacity
        output = (blended * final_mask) + (img_float * (1.0 - final_mask))
        
        return np.clip(output * 255.0, 0, 255).astype(np.uint8)
