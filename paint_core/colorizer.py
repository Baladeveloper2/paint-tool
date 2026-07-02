import cv2
import numpy as np
from PIL import Image
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
        
        # 3. LAB Color Transfer (no caching)
        img_float = image_rgb.astype(np.float32) / 255.0
        img_lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)
        
        L, A, B = cv2.split(img_lab)
        
        # Target color in LAB
        target_L, target_a, target_b = ColorTransferEngine.get_target_lab(target_color_hex)
        
        # Apply Smart Luminance Shift
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
    def get_target_lab(color_hex):
        """Calculate the LAB L/A/B channels for a hex color."""
        rgb = ColorTransferEngine.hex_to_rgb(color_hex)
        pixel = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.uint8)
        lab = cv2.cvtColor(pixel.astype(np.float32)/255.0, cv2.COLOR_RGB2Lab)
        return float(lab[0, 0, 0]), float(lab[0, 0, 1]), float(lab[0, 0, 2])

    @staticmethod
    def composite_multiple_layers(image_rgb, masks_data):
        """
        Single-Pass Compositor. No caching. Recomputes from scratch each call
        to ensure deterministic, stale-free rendering.
        """
        if not masks_data:
            return image_rgb.copy()

        h, w = image_rgb.shape[:2]

        # Convert base image to LAB (fresh, no session cache)
        img_f = image_rgb.astype(np.float32, copy=False) / 255.0
        img_lab = cv2.cvtColor(img_f, cv2.COLOR_RGB2Lab)
        base_L, base_A, base_B = cv2.split(img_lab)

        curr_L_mod = base_L.copy()
        curr_A = base_A.copy()
        curr_B = base_B.copy()

        # Cumulative A/B/L Blending
        for data in masks_data:
            mask = data['mask']
            color_hex = data.get('color')
            if not color_hex: continue

            # Decompress sparse masks
            if sparse.issparse(mask):
                mask = mask.toarray()

            target_L, target_a, target_b = ColorTransferEngine.get_target_lab(color_hex)

            # Resize if needed
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
            blur_val = ((user_soft * 6) - 1, (user_soft * 6) - 1) if user_soft > 0 else (3, 3)

            kernel = np.ones(ColorizerConfig.DILATION_KERNEL_SIZE, np.uint8)
            mask_dilated = cv2.dilate(mask_f, kernel, iterations=ColorizerConfig.DILATION_ITERATIONS)
            mask_soft = cv2.GaussianBlur(mask_dilated, blur_val, 0)

            # Finish-based luminance blending
            finish = data.get('finish', 'Standard')
            valid_mask = mask_f > 0.1
            ref_L = np.mean(base_L[valid_mask]) if np.any(valid_mask) else 75.0
            ref_L = np.clip(ref_L, 10.0, 95.0)

            if finish == 'Matte':
                layer_L = np.clip(base_L * (target_L / ref_L) * 0.9 + 10, 0, 100)
            elif finish == 'Gloss':
                layer_L = np.clip(base_L * (target_L / ref_L) * 1.2 - 5, 0, 100)
            else:  # Standard, Texture, Satin
                layer_L = np.clip(base_L * (target_L / ref_L), 0, 100)

            curr_L_mod = (layer_L * mask_soft) + (curr_L_mod * (1.0 - mask_soft))
            curr_A = (target_a * mask_soft) + (curr_A * (1.0 - mask_soft))
            curr_B = (target_b * mask_soft) + (curr_B * (1.0 - mask_soft))

        # Final conversion
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
