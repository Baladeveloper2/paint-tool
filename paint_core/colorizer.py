import cv2
import numpy as np
from PIL import Image
from scipy import sparse
from app_config.constants import ColorizerConfig

def guided_filter_mask(I, p, r=4, eps=0.01):
    """
    Applies Guided Filter to mask p using guide image I to align edges precisely.
    """
    if len(I.shape) == 3:
        I_gray = cv2.cvtColor((I * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        I_gray = I.astype(np.float32)
    
    p = p.astype(np.float32)
    
    mean_I = cv2.boxFilter(I_gray, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(I_gray * p, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I_gray * I_gray, -1, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    
    q = mean_a * I_gray + mean_b
    return np.clip(q, 0.0, 1.0)

class ColorTransferEngine:
    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert HEX string to RGB tuple."""
        if not isinstance(hex_color, str):
            raise TypeError("hex_color must be a string")
        if not hex_color.startswith('#'):
            raise ValueError("Invalid hex color: missing # prefix")
        if len(hex_color) != 7:
            raise ValueError(f"hex_color must be 7 characters (got {len(hex_color)})")
        
        hex_stripped = hex_color[1:]
        try:
            return tuple(int(hex_stripped[i:i+2], 16) for i in (0, 2, 4))
        except ValueError as e:
            raise ValueError(f"Invalid hex color '{hex_color}': {e}")

    @staticmethod
    def get_target_lab(color_hex):
        """Calculate the LAB L/A/B channels for a hex color."""
        rgb = ColorTransferEngine.hex_to_rgb(color_hex)
        pixel = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.uint8)
        lab = cv2.cvtColor(pixel.astype(np.float32)/255.0, cv2.COLOR_RGB2Lab)
        return float(lab[0, 0, 0]), float(lab[0, 0, 1]), float(lab[0, 0, 2])

    @staticmethod
    def apply_color(image_rgb, mask, target_color_hex, intensity=1.0, seed_point=None, use_adaptive=False):
        """
        Apply color using premium LAB blending.
        """
        # Validate inputs
        if not isinstance(image_rgb, np.ndarray) or not isinstance(mask, np.ndarray):
            raise ValueError("Inputs must be numpy arrays")
            
        # Decompress sparse masks
        if sparse.issparse(mask):
            mask = mask.toarray()
            
        # Ensure mask matches image dimension
        if mask.shape[:2] != image_rgb.shape[:2]:
            raise ValueError("mask dimensions must match image dimensions")
            
        if not mask.any():
            return image_rgb.copy()
            
        # Clamp intensity to [0.0, 1.0]
        intensity = max(0.0, min(float(intensity), 1.0))
            
        image_rgb = image_rgb.astype(np.uint8)
        h, w = image_rgb.shape[:2]
        
        mask_f = mask.astype(np.float32)
        
        # Align mask to physical edges using Guided Filter
        img_f = image_rgb.astype(np.float32) / 255.0
        mask_aligned = guided_filter_mask(img_f, mask_f, r=6, eps=0.01)
        
        # Feather mask slightly
        mask_soft = cv2.GaussianBlur(mask_aligned, (3, 3), 0)
        mask_3ch = np.stack([mask_soft] * 3, axis=-1) * intensity
        
        # Convert image to LAB
        img_lab = cv2.cvtColor(img_f, cv2.COLOR_RGB2Lab)
        L, A, B = cv2.split(img_lab)
        
        # Target color in LAB
        target_L, target_a, target_b = ColorTransferEngine.get_target_lab(target_color_hex)
        
        # Premium Additive Luminance Blending to preserve texture/lighting
        valid_mask = mask_aligned > 0.05
        ref_L = np.mean(L[valid_mask]) if np.any(valid_mask) else 75.0
        ref_L = np.clip(ref_L, 10.0, 95.0)
        
        # Additive details preservation: L_new = target_L + (L_orig - ref_L) * contrast_scale
        new_L = np.clip(target_L + (L - ref_L) * 0.85, 0.0, 100.0)
        
        new_lab = cv2.merge([new_L, np.full_like(A, target_a), np.full_like(B, target_b)])
        recolored_rgb = cv2.cvtColor(new_lab, cv2.COLOR_Lab2RGB)
        
        # Blend
        result_float = (recolored_rgb * mask_3ch) + (img_f * (1.0 - mask_3ch))
        return np.clip(result_float * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def composite_multiple_layers(image_rgb, masks_data):
        """
        Single-Pass Compositor utilizing premium LAB blending and edge-aligned masks.
        """
        if not masks_data:
            return image_rgb.copy()

        h, w = image_rgb.shape[:2]
        img_f = image_rgb.astype(np.float32) / 255.0
        img_lab = cv2.cvtColor(img_f, cv2.COLOR_RGB2Lab)
        base_L, base_A, base_B = cv2.split(img_lab)

        curr_L_mod = base_L.copy()
        curr_A = base_A.copy()
        curr_B = base_B.copy()

        for data in masks_data:
            mask = data['mask']
            color_hex = data.get('color')
            if not color_hex:
                continue

            if sparse.issparse(mask):
                mask = mask.toarray()

            if mask.shape[:2] != (h, w):
                mask_uint8 = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask_f = mask_uint8.astype(np.float32)
            else:
                mask_f = mask.astype(np.float32)

            if mask_f.max() > 1.0:
                mask_f /= 255.0

            # Guided Filter mask alignment to avoid bleeding on other elements
            mask_aligned = guided_filter_mask(img_f, mask_f, r=6, eps=0.01)

            # Apply user softness / refinement
            refinement = data.get('refinement', 0)
            if refinement != 0:
                k_size = abs(refinement) * 2 + 1
                refine_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                if refinement > 0:
                    mask_aligned = cv2.dilate(mask_aligned, refine_kernel)
                else:
                    mask_aligned = cv2.erode(mask_aligned, refine_kernel)

            user_soft = data.get('softness', 0)
            blur_val = ((user_soft * 4) | 1, (user_soft * 4) | 1) if user_soft > 0 else (3, 3)
            mask_soft = cv2.GaussianBlur(mask_aligned, blur_val, 0)

            # Target LAB coordinates
            target_L, target_a, target_b = ColorTransferEngine.get_target_lab(color_hex)

            # Get reference brightness
            valid_mask = mask_aligned > 0.05
            ref_L = np.mean(base_L[valid_mask]) if np.any(valid_mask) else 75.0
            ref_L = np.clip(ref_L, 10.0, 95.0)

            # Finish adjusters
            finish = data.get('finish', 'Standard')
            if finish == 'Matte':
                layer_L = np.clip(target_L + (base_L - ref_L) * 0.75, 0, 100)
            elif finish == 'Gloss':
                layer_L = np.clip(target_L + (base_L - ref_L) * 1.1 + 5, 0, 100)
            else:  # Standard / Satin
                layer_L = np.clip(target_L + (base_L - ref_L) * 0.85, 0, 100)

            # Blend channels
            curr_L_mod = (layer_L * mask_soft) + (curr_L_mod * (1.0 - mask_soft))
            curr_A = (target_a * mask_soft) + (curr_A * (1.0 - mask_soft))
            curr_B = (target_b * mask_soft) + (curr_B * (1.0 - mask_soft))

        # Re-merge to RGB
        final_lab = cv2.merge([curr_L_mod.astype(np.float32), curr_A.astype(np.float32), curr_B.astype(np.float32)])
        final_rgb = cv2.cvtColor(final_lab, cv2.COLOR_Lab2RGB)

        return np.clip(final_rgb * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_texture(image_rgb, mask, texture_rgb, opacity=0.8):
        """
        Apply a texture overlay respecting base lighting details.
        """
        image_rgb = image_rgb.astype(np.uint8)
        h, w = image_rgb.shape[:2]
        
        if sparse.issparse(mask):
            mask = mask.toarray()
            
        mask_f = mask.astype(np.float32)
        img_f = image_rgb.astype(np.float32) / 255.0
        mask_aligned = guided_filter_mask(img_f, mask_f, r=6, eps=0.01)
        mask_soft = cv2.GaussianBlur(mask_aligned, (5, 5), 0)
        mask_3ch = np.stack([mask_soft] * 3, axis=-1)
        
        # Tile texture
        th, tw = texture_rgb.shape[:2]
        if max(th, tw) > ColorizerConfig.MAX_TEXTURE_SIZE:
            scale = ColorizerConfig.MAX_TEXTURE_SIZE / max(th, tw)
            texture_rgb = cv2.resize(texture_rgb, (0, 0), fx=scale, fy=scale)
            th, tw = texture_rgb.shape[:2]
            
        tiled_texture = np.zeros_like(image_rgb)
        for i in range(0, h, th):
            for j in range(0, w, tw):
                curr_h = min(th, h - i)
                curr_w = min(tw, w - j)
                tiled_texture[i:i+curr_h, j:j+curr_w] = texture_rgb[:curr_h, :curr_w]
                
        tex_float = tiled_texture.astype(np.float32) / 255.0
        gray = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)
        gray_3ch = np.stack([gray] * 3, axis=-1)
        
        # Multiply blending
        blended = np.clip(tex_float * gray_3ch * ColorizerConfig.TEXTURE_BRIGHTNESS_BOOST, 0.0, 1.0)
        final_mask = mask_3ch * opacity
        output = (blended * final_mask) + (img_f * (1.0 - final_mask))
        return np.clip(output * 255.0, 0, 255).astype(np.uint8)
