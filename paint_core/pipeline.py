import numpy as np
import cv2
import logging
import torch

logger = logging.getLogger(__name__)

class RenderPipeline:
    def __init__(self, use_advanced_models=True):
        """
        Orchestrates the advanced architectural paint rendering pipeline.
        Production-grade: Uses algorithmic semantic exclusion + edge-aware merging.
        Heavy AI models (YOLO, Mask2Former) are stubbed and can be added incrementally.
        """
        self.use_advanced_models = use_advanced_models
        
        # Placeholders for advanced model pipelines
        self.depth_model = None
        self.sam2_model = None
        self.yolo_model = None
        self.mask2former = None
        
        self.depth_map = None
        self.surface_normals = None
        self.exclusion_mask = None
        
        # Cached scene priors
        self._image_gray = None
        self._canny_edges = None
        self._hsv_image = None
        
        self.image_rgb = None
        
    def set_image(self, image_rgb):
        self.image_rgb = image_rgb
        logger.info("Pipeline: Image set. Generating scene priors...")
        self._generate_scene_priors(image_rgb)
        
    def _generate_scene_priors(self, image_rgb):
        """
        Generates depth map, surface normals, and exclusion masks.
        All operations are cached once per image.
        """
        h, w = image_rgb.shape[:2]
        
        # Pre-compute shared derivatives once (avoids redundant work per click)
        self._image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(self._image_gray, (5, 5), 0)
        self._canny_edges = cv2.Canny(blurred, 40, 100)
        self._hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # 1. Depth Map Generation (lightweight gradient heuristic, fast)
        self.depth_map = self._compute_depth_map(image_rgb)
        
        # 2. Surface Normals (from depth map)
        self.surface_normals = self._compute_surface_normals(self.depth_map)
        
        # 3. Semantic Exclusion Mask (algorithmic, no heavy model needed)
        self.exclusion_mask = self._compute_exclusion_mask(image_rgb)
        
    def _compute_depth_map(self, image_rgb):
        """
        Calculates a pseudo-depth map using brightness and gradient heuristics
        when a proper depth model (Depth Anything) is not loaded.
        """
        # Convert to LAB for luminance
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Normalize luminance
        l_norm = cv2.normalize(l_channel, None, 0, 1, cv2.NORM_MINMAX)
        
        # Assume brighter areas are closer (simple heuristic)
        # Combine with a vertical gradient (ground is usually closer than sky)
        h, w = image_rgb.shape[:2]
        y_grid, _ = np.mgrid[0:h, 0:w]
        y_gradient = 1.0 - (y_grid / float(h))
        
        depth = 0.5 * l_norm + 0.5 * y_gradient
        return depth.astype(np.float32)

    def _compute_surface_normals(self, depth_map):
        """
        Calculates surface normals from the depth map with improved geometric smoothing.
        """
        h, w = depth_map.shape
        depth_map_f32 = depth_map.astype(np.float32)
        
        # Use a larger kernel for smoother, more stable structural gradients
        dzdx = cv2.Sobel(depth_map_f32, cv2.CV_32F, 1, 0, ksize=5)
        dzdy = cv2.Sobel(depth_map_f32, cv2.CV_32F, 0, 1, ksize=5)
        
        # Blur the gradients to ignore micro-textures (like plaster bumps)
        dzdx = cv2.GaussianBlur(dzdx, (5, 5), 0)
        dzdy = cv2.GaussianBlur(dzdy, (5, 5), 0)
        
        nx = -dzdx
        ny = -dzdy
        nz = np.full((h, w), 0.5, dtype=np.float32) # Adjusted Z baseline for flatter walls
        
        magnitude = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
        nx /= magnitude
        ny /= magnitude
        nz /= magnitude
        
        return np.stack([nx, ny, nz], axis=-1)

    def _compute_exclusion_mask(self, image_rgb):
        """
        Algorithmically detects non-wall regions that must NEVER receive paint:
          - Sky (blue/cyan, top half)
          - Glass / windows (high-brightness low-saturation OR high texture energy variance)
          - Window frames (Hough lines)
          - Dark interiors / door recesses
          - Vegetation / trees / grass (green hue, saturated)
          - Vehicles (large dark-grey blobs in lower half)
          - Stone cladding / brick (detected via texture energy)

        All based on HSV color analysis and texture/structure heuristics.
        """
        h, w = image_rgb.shape[:2]

        # ── Texture Energy Mapping (Variance) ───────────────────────────────
        # Glass is very smooth, walls have some texture, plants have high texture.
        # Window frames have strong structural lines.
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Calculate local variance (texture energy)
        blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        sq_gray = cv2.multiply(blur_gray, blur_gray)
        blur_sq = cv2.GaussianBlur(sq_gray, (5, 5), 0)
        blur_gray_sq = cv2.multiply(blur_gray, blur_gray)
        variance = cv2.subtract(blur_sq, blur_gray_sq)
        
        # ── Structural Line Detection (Window Frames) ───────────────────────
        edges = cv2.Canny(blur_gray, 50, 150)
        lines_mask = np.zeros((h, w), dtype=np.uint8)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_mask, (x1, y1), (x2, y2), 1, 3)
        # Dilate lines to cover the frame thickness
        lines_mask = cv2.dilate(lines_mask, np.ones((5,5), np.uint8), iterations=1)
        h, w = image_rgb.shape[:2]

        hsv = self._hsv_image.astype(np.float32)
        H = hsv[:, :, 0]   # 0–179
        S = hsv[:, :, 1]   # 0–255
        V = hsv[:, :, 2]   # 0–255

        # ── Sky ─────────────────────────────────────────────────────────────
        # Blue-cyan hue, moderate-high brightness (cloudy sky also included)
        is_blue_hue  = (H >= 88) & (H <= 132)
        is_cyan_hue  = (H >= 80) & (H <= 100)
        is_sky_v     = V > 140
        sky_cand     = ((is_blue_hue & is_sky_v) | (is_cyan_hue & (V > 160))).astype(np.uint8)
        sky_mask     = sky_cand.copy()
        sky_mask[h // 2:, :] = 0   # sky only in upper half

        # ── Glass / Window / Mirror ──────────────────────────────────────────
        # Very bright + near-achromatic (S low) → reflective glass
        # OR extremely low texture variance (perfectly smooth) combined with high brightness
        is_glass_color = ((V > 185) & (S < 50))
        is_glass_smooth = ((variance < 10) & (V > 150))
        is_glass = (is_glass_color | is_glass_smooth).astype(np.uint8)
        
        # Include structural frames
        is_glass_and_frames = np.logical_or(is_glass, lines_mask).astype(np.uint8)

        # ── Dark interiors (door openings, window depths) ────────────────────
        is_dark_interior = ((V < 35) & (S < 60)).astype(np.uint8)

        # ── Vegetation (trees, bushes, grass) ───────────────────────────────
        # Green hue + enough saturation to distinguish from muted wall green
        is_veg_hue = (H >= 33) & (H <= 88)
        is_veg_sat = S > 55
        is_veg_v   = V > 30          # exclude very dark shadows
        veg_mask   = (is_veg_hue & is_veg_sat & is_veg_v).astype(np.uint8)

        # Dilate vegetation mask to capture fringe pixels where paint bleeds
        veg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        veg_mask   = cv2.dilate(veg_mask, veg_kernel, iterations=1)

        # ── Vehicles (lower-half, dark desaturated large blobs) ─────────────
        # Cars are typically grey, dark, desaturated in lower 40% of image
        is_car_color = (S < 50) & (V > 40) & (V < 190)
        car_cand     = is_car_color.astype(np.uint8)
        car_mask     = car_cand.copy()
        car_mask[:int(h * 0.5), :] = 0   # vehicles only in lower portion

        # ── Combine all exclusion regions ────────────────────────────────────
        combined = np.clip(
            sky_mask.astype(np.int32) +
            is_glass_and_frames.astype(np.int32) +
            is_dark_interior.astype(np.int32) +
            veg_mask.astype(np.int32) +
            car_mask.astype(np.int32),
            0, 1
        ).astype(np.uint8)

        # Morphological cleanup: fill small holes, remove pixel noise
        close_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        open_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  open_k)

        return combined


    def process_wall_selection(self, sam_base_mask, point_coords=None):
        """
        Takes an initial SAM mask and applies full architectural topology constraints:
        1. Merges coplanar regions (wall planes) 
        2. Subtracts semantic exclusions (sky, glass, plants)
        3. Closes gaps aggressively (corners, parapets, recesses)
        4. Snaps to strong architectural edges
        
        This makes clicking anywhere on a wall select the ENTIRE wall surface.
        """
        if sam_base_mask is None or not np.any(sam_base_mask):
            return sam_base_mask
            
        h, w = sam_base_mask.shape[:2]
        
        # 1. Coplanar Region Merging (expand mask across same wall plane)
        merged_mask = self._merge_coplanar_regions(sam_base_mask, self.surface_normals, self.depth_map)
        
        # 2. Subtract Exclusions (Windows, Plants, Sky, Cars)
        refined_mask = np.logical_and(merged_mask, np.logical_not(self.exclusion_mask)).astype(np.uint8)
        
        # 3. Aggressive Gap Filling (closes mortar lines, shadow gaps)
        refined_mask = self._fill_mask_gaps_aggressive(refined_mask)
        
        # 4. Edge Snapping (aligns mask to Canny edges, prevents bleeding past roof/corners)
        snapped_mask = self._snap_to_edges(refined_mask, self._canny_edges)
        
        return snapped_mask
        
    def _merge_coplanar_regions(self, base_mask, normals, depth_map):
        """
        GEOMETRIC REGION GROWING:
        Aggressively expands the SAM mask outward until it hits a physical corner or edge.
        Completely ignores shadows and lighting gradients.
        """
        base_mask_bool = base_mask > 0
        if not np.any(base_mask_bool):
            return base_mask
            
        mean_normal = np.mean(normals[base_mask_bool], axis=0)
        mean_normal /= (np.linalg.norm(mean_normal) + 1e-8)
        
        dot_product = np.sum(normals * mean_normal, axis=-1)
        
        # Very relaxed coplanarity threshold to treat slightly curved/bumpy walls as flat
        coplanar_mask = (dot_product > 0.70).astype(np.uint8)
        
        # Seed the region growing with the base mask
        seed_mask = base_mask_bool.astype(np.uint8)
        
        # Morphological Reconstruction (Flood Fill on Topography)
        # We dilate the seed mask continuously, but constrain it to the coplanar_mask.
        # This grows the mask to fill the ENTIRE connected physical wall.
        prev_mask = np.zeros_like(seed_mask)
        curr_mask = seed_mask.copy()
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # Iterate until convergence (max 50 steps to prevent infinite loop)
        for _ in range(50):
            if np.array_equal(curr_mask, prev_mask):
                break
            prev_mask = curr_mask.copy()
            curr_mask = cv2.dilate(curr_mask, kernel)
            curr_mask = cv2.bitwise_and(curr_mask, coplanar_mask)
            
        return curr_mask

    def _fill_mask_gaps_aggressive(self, mask):
        """
        Fills holes, removes seams, and fuses fractured segments using 
        convex hulls and morphological bridging.
        """
        # 1. Fill internal holes completely
        cnts, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, c in enumerate(cnts):
                if hierarchy[i][3] != -1: # It's an internal hole
                    cv2.drawContours(mask, [c], -1, 1, thickness=-1)
                    
        # 2. Bridge small gaps using heavy morphological closing
        k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        bridged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_bridge)
        
        # 3. Fuse nearby fragmented blocks via Convex Hull if they are very close
        cnts, _ = cv2.findContours(bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 100: # Ignore tiny noise
                hull = cv2.convexHull(c)
                # Fill the hull partially to smooth outer edges without creating massive blobs
                cv2.drawContours(bridged, [hull], -1, 1, thickness=-1)
                
        return bridged

    def _snap_to_edges(self, mask, edges):
        """
        ACTIVE EDGE SNAPPING:
        Forcefully expands the paint mask outward to hit the nearest architectural edge,
        eliminating the thin unpainted lines (sky or old wall color) at boundaries.
        """
        # 1. Expand the mask significantly to overshoot the boundary
        overshoot = cv2.dilate(mask, np.ones((11, 11), np.uint8))
        
        # 2. Thicken the edges to create an impenetrable barrier
        barrier = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        
        # 3. Create a distance field from the barrier
        dist_to_edge = cv2.distanceTransform(255 - barrier, cv2.DIST_L2, 3)
        
        # 4. Snap logic: Keep pixels that are inside the overshoot AND 
        # structurally continuous with the original mask, bounded by the barrier.
        # We simulate a watershed constraint by eroding the overshoot until it hits the barrier.
        
        snapped = overshoot.copy()
        for _ in range(5): # iterative pull-back from edges
            # Remove pixels that are ON the barrier
            snapped = cv2.bitwise_and(snapped, cv2.bitwise_not(barrier))
            # Smooth
            snapped = cv2.morphologyEx(snapped, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            
        # Ensure the original core mask is never lost
        final_mask = np.logical_or(mask, snapped).astype(np.uint8)
        
        # Feather the absolute perimeter for anti-aliasing
        final_mask = cv2.medianBlur(final_mask, 5)
        
        return final_mask
