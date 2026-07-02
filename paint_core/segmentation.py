import numpy as np
import torch
import cv2
import logging
from mobile_sam import sam_model_registry, SamPredictor
from app_config.constants import SegmentationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SegmentationEngine:
    def __init__(self, checkpoint_path=None, model_type="vit_b", device=None, model_instance=None):
        """
        Initialize the SAM model.
        Args:
            checkpoint_path: Path to weights (if loading new).
            model_type: SAM architecture type.
            device: 'cuda' or 'cpu'.
            model_instance: Pre-loaded sam_model_registry instance (optional).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if model_instance is not None:
             self.sam = model_instance
        elif checkpoint_path:
             # OPTIMIZATION: Force vit_t if filename suggests MobileSAM
             if "mobile_sam" in checkpoint_path and model_type != "vit_t":
                 logger.warning(f"Model type override: Detected MobileSAM weights but requested {model_type}. Forcing 'vit_t'.")
                 model_type = "vit_t"
                 
             logger.info(f"Loading SAM model ({model_type}) on {self.device}...")
             self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
             self.sam.to(device=self.device)
        else:
             raise ValueError("Either checkpoint_path or model_instance must be provided.")

        self.predictor = SamPredictor(self.sam)
        self.is_image_set = False

    def set_image(self, image_rgb):
        """
        Process the image and compute embeddings.
        Args:
            image_rgb: NumPy array (H, W, 3) in RGB format.
        """
        # OPTIMIZATION: Check if image is already set to avoid expensive re-encoding
        if self.is_image_set and hasattr(self, 'image_rgb') and self.image_rgb is not None:
            if image_rgb.shape == self.image_rgb.shape and np.array_equal(image_rgb, self.image_rgb):
                logger.info("Image already set. Skipping embedding computation.")
                print("DEBUG: SAM Engine - Image already set.")
                return

        logger.info("Computing image embeddings...")
        print(f"DEBUG: SAM Engine {id(self)} - Computing embeddings...")
        self.predictor.set_image(image_rgb)
        self.is_image_set = True
        self.image_rgb = image_rgb
        
        # --- PRE-COMPUTE FEATURES FOR FASTER CLICKS ---
        # 1. Grayscale
        self.image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        self.image_u16 = image_rgb.astype(np.uint16)
        
        # 2. Gaussian Blur (for small objects/edge detection)
        k_size = SegmentationConfig.GAUSSIAN_KERNEL_SIZE
        self.image_blurred = cv2.GaussianBlur(self.image_gray, k_size, 0)
        
        # 3. Laplacian Edges (base)
        edges = cv2.Laplacian(self.image_blurred, cv2.CV_16S, ksize=3)
        self.image_edges_map = cv2.convertScaleAbs(edges)
        
        # 4. Strict Canny Edges (for architectural boundaries)
        img_blur2 = cv2.GaussianBlur(self.image_gray, (5, 5), 0)
        self.canny_edges = cv2.Canny(img_blur2, 30, 100)
        
        # 5. Shadow Vision (CLAHE)
        # Compute CLAHE equalized image to reveal details hidden in dark shadows (balconies/interiors)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.image_clahe = clahe.apply(self.image_gray)
        img_clahe_blur = cv2.GaussianBlur(self.image_clahe, (5, 5), 0)
        self.shadow_edges = cv2.Canny(img_clahe_blur, 50, 150)
        
        print(f"DEBUG: SAM Engine {id(self)} - is_image_set = True ✅ (Features Pre-computed)")
        logger.info("Embeddings and features computed.")
        
        # 6. Build permanent architectural exclusion mask (windows, glass, doors, vegetation, sky, ground)
        self.exclusion_mask = self._build_exclusion_mask(image_rgb)
        print(f"DEBUG: Exclusion mask built. Excluded {np.sum(self.exclusion_mask)} pixels.")

    def _build_exclusion_mask(self, image_rgb):
        """
        Build a comprehensive permanent exclusion mask for all non-wall surfaces.
        This mask is computed ONCE per image and subtracted from every wall mask.

        Excluded categories:
          - Vegetation (trees, grass, plants, leaves) - HSV green/yellow-green hues
          - Sky - bright low-saturation blue or overexposed white
          - Glass / windows - high local variance + blue tint OR high specularity
          - Architectural openings - dense edge rectangles (windows, doors, grills, balconies)
          - Ground / road - low-position brownish or grey flat regions
          - Dark specular reflections on glass - very dark uniform low-variance patches

        Returns:
            Binary uint8 mask (H, W) where 1 = excluded (never paint here)
        """
        h, w = image_rgb.shape[:2]
        exclusion = np.zeros((h, w), dtype=np.uint8)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.int32)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        # ── 1. VEGETATION (green / yellow-green hues) ──────────────────────
        veg = ((H > 25) & (H < 90) & (S > 35) & (V > 20)).astype(np.uint8)
        # Dilate slightly to cover leaf shadows that blend into wall color
        veg = cv2.dilate(veg, np.ones((7, 7), np.uint8), iterations=2)
        exclusion = np.maximum(exclusion, veg)

        # ── 2. SKY (light blue or overexposed near-white) ───────────────────
        sky_blue  = ((H > 95) & (H < 135) & (S > 20) & (V > 150)).astype(np.uint8)
        sky_white = ((S < 25) & (V > 230)).astype(np.uint8)
        sky = cv2.dilate(np.maximum(sky_blue, sky_white), np.ones((5, 5), np.uint8), iterations=1)
        exclusion = np.maximum(exclusion, sky)

        # ── 3. DEEP WATER / POOL / REFLECTIVE FLOOR (uniform dark blue) ────
        deep_water = ((H > 90) & (H < 130) & (S > 60) & (V < 120)).astype(np.uint8)
        exclusion = np.maximum(exclusion, deep_water)

        # ── 4. GROUND / ROAD (bottom 15% of image that is brownish/grey) ───
        ground_zone_y = int(h * 0.85)
        ground_region = image_rgb[ground_zone_y:, :]
        ground_hsv = hsv[ground_zone_y:, :, :]
        # Brown / beige / grey at bottom
        is_ground_color = (
            ((ground_hsv[:, :, 0] < 30) | (ground_hsv[:, :, 0] > 150)) &
            (ground_hsv[:, :, 1] < 60)
        ).astype(np.uint8)
        exclusion[ground_zone_y:][is_ground_color == 1] = 1

        # ── 5. ARCHITECTURAL OPENINGS via edge-density rectangles ──────────
        # Windows, doors, grills and balconies have very dense Laplacian edges
        _, hi_edges = cv2.threshold(self.image_edges_map, 35, 255, cv2.THRESH_BINARY)
        # Close nearby edges into solid blobs
        close_k = np.ones((45, 45), np.uint8)
        hi_closed = cv2.morphologyEx(hi_edges, cv2.MORPH_CLOSE, close_k)
        # Also use Canny-based shadow edges
        combined_hi = cv2.bitwise_or(hi_closed, self.shadow_edges)
        close_k2 = np.ones((35, 35), np.uint8)
        combined_closed = cv2.morphologyEx(combined_hi, cv2.MORPH_CLOSE, close_k2)

        contours, _ = cv2.findContours(combined_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        opening_mask = np.zeros((h, w), dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            # Skip regions that span > 90% of image width (whole facade) or are at very top/bottom
            if rw > w * 0.90 or rh > h * 0.90:
                continue
            # Compute edge density inside the bounding box
            roi_edges = hi_edges[ry:ry+rh, rx:rx+rw]
            edge_density = np.sum(roi_edges > 0) / max(rw * rh, 1)
            # Only mark as opening if edge-dense (windows/grills) or medium-dense (doors/frames)
            if edge_density > 0.08:
                # Shrink bounding rect slightly to keep the wall frame itself paintable
                pad = 4
                cv2.rectangle(
                    opening_mask,
                    (max(0, rx + pad), max(0, ry + pad)),
                    (min(w, rx + rw - pad), min(h, ry + rh - pad)),
                    1, -1
                )
        exclusion = np.maximum(exclusion, opening_mask)

        # ── 6. GLASS / SPECULAR REFLECTIONS ────────────────────────────────
        # Glass appears as high local variance patches that are bright-ish (reflections)
        gray_f = self.image_gray.astype(np.float32)
        local_mean = cv2.GaussianBlur(gray_f, (21, 21), 0)
        local_sq   = cv2.GaussianBlur(gray_f ** 2, (21, 21), 0)
        local_var  = np.clip(local_sq - local_mean ** 2, 0, None)
        # Glass: high variance (varied reflections) AND moderately bright
        glass_hi_var = ((local_var > 300) & (gray_f > 60)).astype(np.uint8)
        # Also mark very bright specular spots (over-exposed glass glare)
        specular = ((V > 240) & (S < 40)).astype(np.uint8)
        glass_mask = cv2.dilate(
            np.maximum(glass_hi_var, specular),
            np.ones((5, 5), np.uint8), iterations=1
        )
        # Only add glass exclusion INSIDE architectural opening zones to avoid excluding shiny walls
        glass_in_openings = (glass_mask & opening_mask).astype(np.uint8)
        glass_in_openings = cv2.dilate(glass_in_openings, np.ones((9, 9), np.uint8), iterations=1)
        exclusion = np.maximum(exclusion, glass_in_openings)

        # ── 7. FINAL CLEANUP ────────────────────────────────────────────────
        # Small exclusion islands (< 100 px) can be spurious noise — remove them
        num_lab, lab_img, stats_ex, _ = cv2.connectedComponentsWithStats(
            exclusion, connectivity=8
        )
        clean = np.zeros_like(exclusion)
        for i in range(1, num_lab):
            if stats_ex[i, cv2.CC_STAT_AREA] >= 100:
                clean[lab_img == i] = 1
        return clean

    def generate_mask(self, point_coords=None, point_labels=None, box_coords=None, level=None, is_wall_only=False, cleanup=True, is_wall_click=False):
        print(f"DEBUG: Entering generate_mask v4.3.1 (UUID: {id(self)})")
        is_small_object = False 
        area_ratio = 0.0        
        aspect_ratio = 0.0      
        if self.predictor is None:
            return None
        """
        Generate a mask for a given point or box.
        Args:
            point_coords: List of [x, y] or NumPy array.
            point_labels: List of labels (1 for foreground, 0 for background).
            box_coords: [x1, y1, x2, y2]
            level: int (0, 1, 2) or None. 
                   0=Fine Details, 1=Sub-segment, 2=Whole Object. 
                   If None, auto-selects highest score.
            is_wall_only: bool. If True, uses stricter wall-specific thresholds.
            cleanup: bool. If True, removes disconnected components to prevent leaks.
        """
        if not self.is_image_set:
            raise RuntimeError("Image not set. Call set_image() first.")

        # Prepare inputs
        sam_point_coords = None
        sam_point_labels = None
        sam_box = None

        if point_coords is not None:
            # Check input structure
            # Case 1: Single point [x, y] -> wrap to [[x, y]]
            # Case 2: List of points [[x, y], ...] -> use as is
            
            arr = np.array(point_coords)
            if arr.ndim == 1:
                sam_point_coords = np.array([point_coords])
            else:
                 sam_point_coords = arr
            
            if point_labels is None:
                # We have N points, so we need N labels
                sam_point_labels = np.array([1] * len(sam_point_coords))
            else:
                sam_point_labels = np.array(point_labels)
        
        if box_coords is not None:
            sam_box = np.array(box_coords)

        with torch.inference_mode():
            masks, scores, logits = self.predictor.predict(
                point_coords=sam_point_coords,
                point_labels=sam_point_labels,
                box=sam_box,
                multimask_output=True # Generate multiple masks and choose best
            )

        # Handle batch dimension if present (MobileSAM/TinySAM might return (1, 3, H, W))
        if len(masks.shape) == 4:
            masks = masks[0]
        if len(scores.shape) == 2:
            scores = scores[0]

        # Select best mask
        if level is not None and 0 <= level < 3:
            # User forced a specific level
            if level == 1:
                # For Small Objects, we usually want Index 0 (most granular)
                # But if Index 0 is tiny (e.g. noise), fallback to Index 1 (Sub-segment)
                area0 = np.sum(masks[0])
                area1 = np.sum(masks[1])
                if area0 < SegmentationConfig.MIN_MASK_AREA_PIXELS * 10 and area1 > area0 * 2: # Heuristic for "too small"
                    best_mask = masks[1]
                else:
                    best_mask = masks[0]
            elif level == 0:
                # --- INTELLIGENT STANDARD WALLS MODE ---
                if box_coords is not None:
                    # Box Mode: Always want the largest/whole object (Index 2)
                    best_mask = masks[2] if scores[2] > SegmentationConfig.SAM_MIN_SCORE else masks[1]
                elif is_wall_click:
                    # --- STRICT ARCHITECTURAL WALL MODE ---
                    best_mask = masks[0]
                    print(f"DEBUG: Strict Wall Mode - Using smallest safest mask (Index 0).")
                else:
                    # --- STANDARD POINT CLICK MODE ---
                    h, w = masks[0].shape
                    image_area = h * w
                    
                    best_mask = masks[0]  # Default fallback
                    best_door_score = -1
                    
                    for idx in range(3):
                        mask_area = np.sum(masks[idx])
                        if mask_area == 0: continue
                        
                        mask_coords = np.argwhere(masks[idx] > 0)
                        if len(mask_coords) == 0: continue
                        y_coords, x_coords = mask_coords[:, 0], mask_coords[:, 1]
                        mask_height = np.max(y_coords) - np.min(y_coords) + 1
                        mask_width = np.max(x_coords) - np.min(x_coords) + 1
                        
                        area_ratio = mask_area / image_area
                        aspect_ratio = mask_height / max(mask_width, 1)
                        
                        door_score = 0
                        if area_ratio < 0.20:
                            if aspect_ratio > 1.3: door_score += 2
                            if area_ratio < 0.08: door_score += 3
                            if aspect_ratio > 1.8: door_score += 3

                        print(f"Mask {idx}: area={area_ratio*100:.1f}%, aspect={aspect_ratio:.2f}, score={scores[idx]:.2f}, door_score={door_score}")
                        
                        if door_score > best_door_score and (best_door_score != 0 or door_score > 4):
                            best_door_score = door_score
                            best_mask = masks[idx]
                    
                    if best_door_score >= 5:
                        mask_area = np.sum(best_mask)
                        area_ratio = mask_area / image_area
                        
                        if area_ratio < 0.05:
                            kernel_size, iterations = 3, 1
                        elif area_ratio < 0.15:
                            kernel_size, iterations = 3, 1
                        else:
                            kernel_size, iterations = 3, 1
                        
                        if area_ratio > 0.005:
                            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                            best_mask = cv2.erode(best_mask.astype(np.uint8), erode_kernel, iterations=iterations).astype(bool)
                    
                    if best_door_score < 3:
                        area0, area1, area2 = np.sum(masks[0]), np.sum(masks[1]), np.sum(masks[2])
                        if area0 < SegmentationConfig.MIN_MASK_AREA_PIXELS:
                            best_idx = np.argmax(scores)
                            best_mask = masks[best_idx]
                        else:
                            best_mask = masks[0]
                            
                # Ultimate fallback to guarantee best_mask is always defined
                if 'best_mask' not in locals() or best_mask is None:
                    print("DEBUG: best_mask undefined, falling back to highest score mask.")
                    best_idx = np.argmax(scores)
                    best_mask = masks[best_idx]
            else:
                best_mask = masks[level]
        else:
            # Heuristic: Favor 'Fine Detail' (Index 0) for Point Clicks
            # Previously we favored Index 1, which caused "wrong object" selection for thin walls.
            if box_coords is not None:
                best_mask = masks[2] if scores[2] > SegmentationConfig.SAM_MIN_SCORE else masks[1]
            else:
                # Point Mode: We want the EXACT part user clicked.
                # Index 0 is usually the most granular (e.g., just the side strip).
                # Index 1 often merges neighbors (e.g., side strip + main wall).
                
                # Only skip Index 0 if it's basically noise
                area0 = np.sum(masks[0])
                if area0 < SegmentationConfig.MIN_MASK_AREA_PIXELS: 
                     best_idx = np.argmax(scores)
                     best_mask = masks[best_idx]
                else:
                     best_mask = masks[0]
        
        if cleanup:
            h, w = best_mask.shape
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            
            # Use a reference point for connectivity filtering
            ref_x, ref_y = None, None
            if point_coords is not None and len(point_coords) > 0:
                pos_indices = np.where(sam_point_labels == 1)[0]
                if len(pos_indices) > 0:
                    idx = pos_indices[-1]
                    ref_x, ref_y = int(sam_point_coords[idx][0]), int(sam_point_coords[idx][1])
            elif box_coords is not None:
                ref_x = int((box_coords[0] + box_coords[2]) / 2)
                ref_y = int((box_coords[1] + box_coords[3]) / 2)

            if ref_x is not None:
                is_small_object = False  # Initialize safely for Box mode
                # --- ADAPTIVE FILTERING ---
                # Calculate reference color (from click point or box center)
                y1, y2 = max(0, ref_y-1), min(h, ref_y+2)
                x1, x2 = max(0, ref_x-1), min(w, ref_x+2)
                seed_patch = self.image_rgb[y1:y2, x1:x2]
                
                # Use MEDIAN instead of MEAN for seed color.
                seed_color = np.median(seed_patch, axis=(0, 1))
                
                img_u16 = self.image_u16
                
                if box_coords is not None:
                     # BOX MODE: Enhanced Logic (Mask-Based Seed + Color Diff)
                     mask_indices = np.where(mask_uint8 > 0)
                     if len(mask_indices[0]) > 0:
                         seed_patch = self.image_rgb[mask_indices]
                         seed_color = np.median(seed_patch, axis=0) 
                     
                     # --- VIBRANT COLOR AWARENESS (Box Mode) ---
                     # For box mode, we still use a broad check but stricter for different hues
                     diff_r = np.abs(img_u16[:,:,0] - seed_color[0])
                     diff_g = np.abs(img_u16[:,:,1] - seed_color[1])
                     diff_b = np.abs(img_u16[:,:,2] - seed_color[2])
                     color_diff = np.maximum(np.maximum(diff_r, diff_g), diff_b)
                     
                     valid_mask = (color_diff < SegmentationConfig.COLOR_DIFF_BOX_MODE).astype(np.uint8)
                     
                     # Enable Edge Detection to snap to lines
                     _, edge_barrier = cv2.threshold(self.image_edges_map, SegmentationConfig.EDGE_THRESHOLD_BOX_MODE, 255, cv2.THRESH_BINARY_INV)
                     edge_barrier = (edge_barrier / 255).astype(np.uint8)
                     
                     mask_refined = (mask_uint8 & valid_mask & edge_barrier)
                     
                     # If validation killed the mask (e.g. wrong seed), fallback to original SAM mask
                     if np.sum(mask_refined) < (np.sum(mask_uint8) * 0.1):
                         mask_refined = mask_uint8 
                else:
                    # Point Click Mode Logic
                    # Check if the SAM mask implies a very small/thin object
                    h, w = mask_uint8.shape
                    mask_area_px = np.sum(mask_uint8)
                    
                    # Increased threshold from 1% to 3% to capture vertical wall strips/pillars
                    # These "medium" objects also need the edge-barrier disabled to paint fully.
                    is_small_object = mask_area_px < (h * w * SegmentationConfig.SMALL_OBJECT_THRESHOLD)
                    
                    std_dev = np.std(seed_color)
                    
                    if level == 0:
                        if is_small_object:
                            # --- VIBRANT WALL REFINEMENT (Small/Detached Objects) ---
                            s_max, s_min = np.max(seed_color), np.min(seed_color)
                            saturation = (s_max - s_min) / (s_max + 1)
                            is_vibrant = saturation > 0.3
                            
                            # 🌈 HUE-TOLERANT COLOR MATCHING
                            diff_r = np.abs(img_u16[:,:,0] - seed_color[0].astype(np.int16))
                            diff_g = np.abs(img_u16[:,:,1] - seed_color[1].astype(np.int16))
                            diff_b = np.abs(img_u16[:,:,2] - seed_color[2].astype(np.int16))
                            rgb_diff = np.maximum(np.maximum(diff_r, diff_g), diff_b)
                            img_hsv = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
                            seed_hsv = cv2.cvtColor(np.uint8([[seed_color]]), cv2.COLOR_RGB2HSV)[0,0].astype(np.int16)
                            hue_diff = np.abs(img_hsv[:,:,0] - seed_hsv[0])
                            hue_diff = np.minimum(hue_diff, 180 - hue_diff)
                            color_diff = (0.7 * rgb_diff) + (0.3 * (hue_diff * 2))
                            
                            tol = SegmentationConfig.COLOR_DIFF_WALL_MODE if is_wall_only else SegmentationConfig.COLOR_DIFF_SMALL_OBJECT
                            if is_vibrant: tol += 15
                            
                            valid_mask = (color_diff < tol).astype(np.uint8) 
                            edge_thresh = SegmentationConfig.EDGE_THRESHOLD_WALL_MODE if is_wall_only else SegmentationConfig.EDGE_THRESHOLD_SMALL_OBJECT
                            _, edge_barrier = cv2.threshold(self.image_edges_map, edge_thresh, 255, cv2.THRESH_BINARY_INV)
                            edge_barrier = (edge_barrier / 255).astype(np.uint8)
                            mask_refined = (mask_uint8 & valid_mask & edge_barrier)
                        else:
                            # --- DISTANCE-DECAYING TOLERANCE ---
                            # Logic for Large Walls (Standard or Wall Click)
                            diff_r = np.abs(img_u16[:,:,0] - seed_color[0].astype(np.int16))
                            diff_g = np.abs(img_u16[:,:,1] - seed_color[1].astype(np.int16))
                            diff_b = np.abs(img_u16[:,:,2] - seed_color[2].astype(np.int16))
                            rgb_diff = np.maximum(np.maximum(diff_r, diff_g), diff_b)
                            img_hsv = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
                            seed_hsv = cv2.cvtColor(np.uint8([[seed_color]]), cv2.COLOR_RGB2HSV)[0,0].astype(np.int16)
                            hue_diff = np.abs(img_hsv[:,:,0] - seed_hsv[0])
                            color_diff = (0.4 * rgb_diff) + (0.6 * (hue_diff * 2))

                            Y, X = np.ogrid[:h, :w]
                            dist_from_click = np.sqrt((X - ref_x)**2 + (Y - ref_y)**2)
                            decay_factor = np.clip(1.0 - (dist_from_click / SegmentationConfig.DECAY_DISTANCE_MAX), SegmentationConfig.DECAY_FACTOR_MIN, 1.0)
                            
                            # Wall Click Mode uses dynamic tolerance based on seed brightness to avoid bleeding in dark areas
                            s_max, s_min = np.max(seed_color), np.min(seed_color)
                            
                            if is_wall_click:
                                # Adaptive tolerance: Dark areas (like shadows/interiors) get strict tolerance
                                brightness = s_max / 255.0
                                # Scale base_tol from 45 (pitch black) up to 130 (bright sun)
                                base_tol = 45 + (85 * brightness)
                            else:
                                base_tol = SegmentationConfig.COLOR_DIFF_WALL_MODE if is_wall_only else 95
                                
                            if (s_max - s_min) / (s_max + 1) > 0.3: base_tol += 10

                            tol = base_tol * decay_factor 
                            valid_gate = (color_diff < tol).astype(np.uint8)
                            
                            # EDGE BARRIER: Wall Click ignores small brick edges (Threshold 35+)
                            edge_thresh = 35 if is_wall_click else (SegmentationConfig.EDGE_THRESHOLD_WALL_MODE if is_wall_only else SegmentationConfig.EDGE_THRESHOLD_STANDARD_WALL)
                            _, edge_barrier = cv2.threshold(self.image_edges_map, edge_thresh, 255, cv2.THRESH_BINARY_INV)
                            
                            # 🛡️ BALANCED BARRIER: Thinner for house textures if in Wall Mode
                            if is_wall_click or is_wall_only:
                                kernel = np.ones((3,3), np.uint8)
                                edge_barrier = cv2.erode((edge_barrier/255).astype(np.uint8), kernel, iterations=1)
                            else:
                                edge_barrier = (edge_barrier / 255).astype(np.uint8)
                            
                            # 🏛️ REFINEMENT STRATEGY PICKER
                            if not is_wall_click and not is_wall_only:
                                # Standard Precise Mode: Stay within SAM boundaries
                                mask_refined = (mask_uint8 & valid_gate & edge_barrier)
                            else:
                                # ============================================================
                                # STRICT ARCHITECTURAL WALL MODE
                                # Uses the pre-computed exclusion_mask (windows, glass, doors,
                                # vegetation, sky, ground) built at set_image() time.
                                # ============================================================
                                
                                # Strict architectural edge barrier (Canny + CLAHE edges)
                                combined_edges = cv2.bitwise_or(self.canny_edges, self.shadow_edges)
                                strict_edge_barrier = (combined_edges == 0).astype(np.uint8)
                                
                                # Start with SAM's finest mask constrained to architectural edges
                                mask_refined = (mask_uint8 & strict_edge_barrier).astype(np.uint8)
                                
                                # Absolute exclusion: subtract all non-wall regions
                                mask_refined[self.exclusion_mask == 1] = 0
                                
                                # Tiny closing only to fill sub-pixel SAM fragmentation (NOT to bridge gaps)
                                sm_kernel = np.ones((3, 3), np.uint8)
                                mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, sm_kernel)
                                
                                # Re-enforce edges and exclusions after closing
                                mask_refined = (mask_refined & strict_edge_barrier).astype(np.uint8)
                                mask_refined[self.exclusion_mask == 1] = 0
                                
                                print(f"DEBUG: Wall mask after exclusion: {np.sum(mask_refined)} pixels remain")

                            
                    elif level == 1:
                        # Level 1 Sub-segment: Calculate color_diff here too
                        diff_r = np.abs(img_u16[:,:,0] - seed_color[0])
                        diff_g = np.abs(img_u16[:,:,1] - seed_color[1])
                        diff_b = np.abs(img_u16[:,:,2] - seed_color[2])
                        color_diff = np.maximum(np.maximum(diff_r, diff_g), diff_b)
                        valid_mask = (color_diff < SegmentationConfig.INTENSITY_DIFF_LEVEL_1).astype(np.uint8) 
                        mask_refined = (mask_uint8 & valid_mask)
                    else:
                        diff_r = np.abs(img_u16[:,:,0] - seed_color[0])
                        diff_g = np.abs(img_u16[:,:,1] - seed_color[1])
                        diff_b = np.abs(img_u16[:,:,2] - seed_color[2])
                        color_diff = np.maximum(np.maximum(diff_r, diff_g), diff_b)
                        valid_mask = (color_diff < SegmentationConfig.INTENSITY_DIFF_LEVEL_2).astype(np.uint8)
                        mask_refined = (mask_uint8 & valid_mask)

                    # Ensure click point is always preserved
                    if ref_x is not None and ref_y is not None:
                         cv2.circle(mask_refined, (ref_x, ref_y), SegmentationConfig.CLICK_PRESERVE_RADIUS, 1, -1) 

                # --- SELECTIVE HOLE FILLING ---
                if level == 0 or box_coords is not None:
                    # Skip erosion/closing for Small Objects to prevent deleting thin lines
                    # CRITICAL: Also skip for level 0 to preserve eroded safety margins for doors/windows
                    if not (level == 0 and is_small_object):
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, SegmentationConfig.MORPH_KERNEL_SIZE)
                        mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel)
                    
                    # Selective internal hole filling using HIERARCHY
                    # RETR_CCOMP returns hierarchy [Next, Previous, First_Child, Parent]
                    # If a contour has a parent (hierarchy[i][3] != -1), it's an internal hole.
                    cnts, hierarchy = cv2.findContours(mask_refined, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    out_mask = np.copy(mask_refined)

                    if hierarchy is not None:
                        hierarchy = hierarchy[0] # flatten
                        for i, c in enumerate(cnts):
                            parent_idx = hierarchy[i][3]
                            if parent_idx != -1: # It is a hole (has a parent)
                                area = cv2.contourArea(c)
                                # REDUCED THRESHOLD: from 0.005 (0.5%) to 0.001 (0.1%) 
                                # to prevent filling intentional architectural holes.
                                if area < (h * w * 0.001): 
                                    # 🖼️ ULTRA PROTECT: Does the hole contain details?
                                    hole_roi = np.zeros_like(mask_refined)
                                    cv2.drawContours(hole_roi, [c], -1, 1, thickness=-1)
                                    avg_edge = cv2.mean(self.image_edges_map, mask=hole_roi)[0]
                                    
                                    # Increased sensitivity: only fill if edge energy is VERY low (< 4.0)
                                    if avg_edge < 4.0: 
                                        cv2.drawContours(out_mask, [c], -1, 1, thickness=-1)
                    mask_refined = out_mask
                elif level == 1:
                    # Level 1 (Small Objects): No hole filling or closing to preserve lattice/mesh details
                    pass

                if np.sum(mask_refined) > SegmentationConfig.MIN_MASK_AREA_PIXELS:
                    mask_uint8 = mask_refined
            
            # Connectivity filtering and Object Recovery
            if ref_x is not None:
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=SegmentationConfig.CONNECTED_COMPONENTS_CONNECTIVITY)
                if num_labels > 1:
                    if box_coords is not None:
                        # --- BOX MODE: MULTI-COMPONENT RECOVERY ---
                        # In Photoshop style, if you box an object with holes (like a perforated wall),
                        # we want to keep ALL pieces of that object that are inside the box.
                        recovered_mask = np.zeros_like(best_mask)
                        bx1, by1, bx2, by2 = box_coords
                        
                        for i in range(1, num_labels):
                            cx, cy = centroids[i]
                            # If centroid is inside box, or it's the largest component
                            if (bx1 < cx < bx2 and by1 < cy < by2) or stats[i, cv2.CC_STAT_AREA] > (h*w*0.05):
                                recovered_mask |= (labels_im == i)
                        
                        if np.any(recovered_mask):
                            best_mask = recovered_mask
                    else:
                        # Point Click Mode: Keep only the target component
                        # Ensure coordinates are integers for array indexing
                        ix = int(max(0, min(ref_x, w - 1)))
                        iy = int(max(0, min(ref_y, h - 1)))
                        
                        target_label = labels_im[iy, ix]
                        if target_label != 0:
                            # Use intelligent filter to remove small/far disconnected objects (pots, artifacts)
                            best_mask = self._filter_small_components(mask_uint8, ref_x, ref_y, target_label, labels_im, stats, centroids)
                        else:
                            max_area = 0
                            max_label = 1
                            for i in range(1, num_labels):
                                if stats[i, cv2.CC_STAT_AREA] > max_area:
                                    max_area = stats[i, cv2.CC_STAT_AREA]
                                    max_label = i
                            best_mask = (labels_im == max_label)
        
        # === FINAL ABSOLUTE EXCLUSION PASS (wall mode) ===
        # After all connectivity filtering, apply exclusion mask one last time.
        # This guarantees that no window, glass, vegetation, or sky pixel survives.
        if is_wall_click or is_wall_only:
            if hasattr(self, 'exclusion_mask') and self.exclusion_mask is not None:
                if isinstance(best_mask, np.ndarray):
                    bm = best_mask.astype(np.uint8)
                    bm[self.exclusion_mask == 1] = 0
                    best_mask = bm.astype(bool)
                    area = np.sum(best_mask)
                    print(f"DEBUG: Final wall mask area after exclusion = {area} px")

        return best_mask

    def _filter_small_components(self, mask, click_x, click_y, target_label, labels_im, stats, centroids):
        """
        Internal helper to remove disconnected components that are too small or too far from click.
        Helps prevent painting unintended pots/decorations when walls are selected.
        """
        num_labels = len(stats)
        h, w = mask.shape
        
        # Get clicked component stats
        main_area = stats[target_label, cv2.CC_STAT_AREA]
        
        # Build cleaned mask
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(1, num_labels):
            component_area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            
            # Distance from click
            dist = np.sqrt((cx - click_x)**2 + (cy - click_y)**2)
            
            # Keep component if:
            # 1. It's exactly the clicked component
            # 2. OR it's a reasonably large piece (>=10% of main) AND close enough
            if i == target_label:
                clean_mask[labels_im == i] = 1
            elif (component_area >= main_area * SegmentationConfig.MIN_COMPONENT_RATIO and 
                  dist < SegmentationConfig.MAX_COMPONENT_DISTANCE):
                clean_mask[labels_im == i] = 1
        
        return clean_mask.astype(bool)

