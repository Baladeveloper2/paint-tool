import hashlib
import threading
import time
import numpy as np
import torch
import cv2
import logging
import traceback
import sys
from ultralytics import YOLO
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import Sam2Processor, Sam2Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def guided_filter(I, p, r, eps):
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
    return q

class AIEngine:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, device=None):
        with cls._lock:
            if cls._instance is None:
                logger.info("Lifecycle: AI Engine Created")
                cls._instance = cls(device)
            else:
                logger.info("Lifecycle: AI Engine Reused")
        return cls._instance

    def __init__(self, device=None):
        if AIEngine._instance is not None:
            raise Exception("This class is a singleton! Use get_instance().")
            
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_lock = threading.Lock()
        self.current_image_hash = None
        
        self.yolo = None
        self.mask2former_processor = None
        self.mask2former_model = None
        self.sam2_processor = None
        self.sam2_model = None
        
        self.is_image_set = False
        self.image_rgb = None
        self.wall_base_mask = None
        self.exclusion_mask = None
        self.wall_planes = {}
        
        self._load_models()
        
    def _load_models(self):
        logger.info(f"Loading Production AI selection pipeline on {self.device}...")
        start_time = time.time()
        try:
            import logging as hf_logging
            hf_logging.getLogger("transformers").setLevel(hf_logging.ERROR)
            
            self.yolo = YOLO("yolov8n-seg.pt")
            
            self.mask2former_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
            self.mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
            self.mask2former_model.to(self.device)
            self.mask2former_model.eval()
            
            self.sam2_processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")
            self.sam2_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
            self.sam2_model.to(self.device)
            self.sam2_model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"PERFORMANCE: Model Load Time: {load_time:.2f} seconds")
            
            assert self.yolo is not None, "assert models_loaded failed"
            assert self.mask2former_model is not None, "assert models_loaded failed"
            assert self.sam2_model is not None, "assert predictor_exists failed"
            
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb = traceback.extract_tb(exc_traceback)[-1]
            logger.error(f"Failed to load models: {e}")
            logger.error(f"Filename: {tb.filename}, Function: {tb.name}, Line: {tb.lineno}")
            print("--- FULL TRACEBACK ---")
            traceback.print_exc(file=sys.stdout)
            print("----------------------")
            sys.exit(1)

    def set_image(self, image_rgb):
        if not isinstance(image_rgb, np.ndarray) or len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be a 3-channel numpy array")
            
        img_hash = hashlib.md5(image_rgb.tobytes()).hexdigest()
        if self.current_image_hash == img_hash and self.is_image_set:
            logger.info("Lifecycle: Embedding Reused")
            return
            
        with self.inference_lock:
            start_time = time.time()
            self.is_image_set = False
            self.image_rgb = image_rgb.copy()
            self.wall_base_mask = None
            self.exclusion_mask = None
            self.wall_planes = {}
            
            h, w = image_rgb.shape[:2]
            
            try:
                # YOLO
                logger.info("Lifecycle: YOLO Detect Wall Started")
                yolo_start = time.time()
                yolo_results = self.yolo(image_rgb, verbose=False)
                yolo_mask = np.zeros((h, w), dtype=np.uint8)
                if yolo_results and yolo_results[0].masks is not None:
                    for mask_obj in yolo_results[0].masks.data:
                        m = mask_obj.cpu().numpy()
                        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        yolo_mask = np.maximum(yolo_mask, (m_resized > 0.5).astype(np.uint8))
                logger.info(f"PERFORMANCE: YOLO Time: {time.time() - yolo_start:.2f} seconds")
                logger.info("Lifecycle: YOLO Completed")
                
                # Mask2Former
                inputs = self.mask2former_processor(images=image_rgb, return_tensors="pt")
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                    
                with torch.no_grad():
                    outputs = self.mask2former_model(**inputs)
                    
                semantic_segmentation = self.mask2former_processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(h, w)]
                )[0]
                semantic_map = semantic_segmentation.cpu().numpy()
                
                self.wall_base_mask = (semantic_map == 0) | (semantic_map == 1) | (semantic_map == 25) | (semantic_map == 43) | (semantic_map == 105)
                ade_exclude_mask = (~self.wall_base_mask).astype(np.uint8)
                self.exclusion_mask = np.maximum(yolo_mask, ade_exclude_mask)
                
                # Boundaries & Watershed
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 30, 90)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
                structural_lines_mask = np.zeros_like(edges)
                
                if lines is not None:
                    for line in lines:
                        coords = line.flatten()
                        if len(coords) >= 4:
                            x1, y1, x2, y2 = map(int, coords[:4])
                            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                            is_horizontal = (angle < 15) or (angle > 165)
                            is_vertical = (75 < angle < 105)
                            if is_horizontal or is_vertical:
                                cv2.line(structural_lines_mask, (x1, y1), (x2, y2), 255, thickness=4)
                
                edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dilated_edges = cv2.dilate(edges, edge_kernel, iterations=1)
                combined_boundaries = np.maximum(dilated_edges, structural_lines_mask)
                
                split_wall = self.wall_base_mask.copy().astype(np.uint8)
                split_wall[self.exclusion_mask > 0] = 0
                split_wall[combined_boundaries > 0] = 0
                
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(split_wall, connectivity=8)
                markers = np.zeros((h, w), dtype=np.int32)
                markers[self.exclusion_mask > 0] = 1
                markers[0, :] = 1
                markers[-1, :] = 1
                markers[:, 0] = 1
                markers[:, -1] = 1
                
                current_marker_id = 2
                self.wall_plane_ids = {}
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= 150:
                        markers[labels_im == i] = current_marker_id
                        self.wall_plane_ids[current_marker_id] = f"WallPlane_{current_marker_id-1:03d}"
                        current_marker_id += 1
                        
                smooth_img = cv2.bilateralFilter(image_rgb, 9, 50, 50)
                cv2.watershed(smooth_img, markers)
                
                for marker_id, plane_id in self.wall_plane_ids.items():
                    plane_mask = (markers == marker_id)
                    if np.sum(plane_mask) > 100:
                        plane_mask = cv2.morphologyEx(plane_mask.astype(np.uint8), cv2.MORPH_CLOSE, edge_kernel) > 0
                        plane_mask[self.exclusion_mask > 0] = False
                        y_indices, x_indices = np.where(plane_mask)
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            x1, y1 = np.min(x_indices), np.min(y_indices)
                            x2, y2 = np.max(x_indices), np.max(y_indices)
                            self.wall_planes[plane_id] = {
                                "mask": plane_mask,
                                "bbox": [int(x1), int(y1), int(x2), int(y2)]
                            }
                
                self.image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                self.image_edges_map = cv2.convertScaleAbs(cv2.Laplacian(self.image_gray, cv2.CV_16S, ksize=3))
                
                self.current_image_hash = img_hash
                self.is_image_set = True
                
                assert self.is_image_set, "assert embedding_exists failed"
                embed_time = time.time() - start_time
                logger.info(f"PERFORMANCE: Embedding Time: {embed_time:.2f} seconds")
                
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb = traceback.extract_tb(exc_traceback)[-1]
                logger.error(f"set_image failed: {e}")
                logger.error(f"Filename: {tb.filename}, Function: {tb.name}, Line: {tb.lineno}")
                print("--- FULL TRACEBACK ---")
                traceback.print_exc(file=sys.stdout)
                print("----------------------")
                sys.exit(1)

    def generate_mask(self, point_coords=None, point_labels=None, box_coords=None, level=None, is_wall_only=False, cleanup=True, is_wall_click=False):
        if not self.is_image_set:
            raise RuntimeError("must call set_image first")
            
        with self.inference_lock:
            total_click_start = time.time()
            logger.info("Lifecycle: Click Started")
            
            try:
                h, w = self.image_rgb.shape[:2]
                
                if point_coords is not None:
                    pts = np.array(point_coords)
                    if pts.ndim == 1:
                        pts = np.array([point_coords])
                    for pt in pts:
                        px, py = pt[0], pt[1]
                        if px < 0 or px >= w or py < 0 or py >= h:
                            raise ValueError("out of image bounds")
                            
                if box_coords is not None:
                    bx1, by1, bx2, by2 = box_coords
                    if bx1 < 0 or bx1 >= w or by1 < 0 or by1 >= h or bx2 < 0 or bx2 >= w or by2 < 0 or by2 >= h:
                        raise ValueError("out of image bounds")
                    if bx2 < bx1 or by2 < by1:
                        raise ValueError("Invalid box coordinates")

                ref_x, ref_y = None, None
                if point_coords is not None:
                    arr = np.array(point_coords)
                    if arr.ndim == 1:
                        ref_x, ref_y = int(point_coords[0]), int(point_coords[1])
                    else:
                        ref_x, ref_y = int(arr[-1][0]), int(arr[-1][1])
                elif box_coords is not None:
                    ref_x = int((box_coords[0] + box_coords[2]) / 2)
                    ref_y = int((box_coords[1] + box_coords[3]) / 2)

                clicked_plane_mask = None
                clicked_plane_id = None
                clicked_bbox = None
                
                if ref_x is not None and ref_y is not None:
                    for plane_id, plane_data in self.wall_planes.items():
                        p_mask = plane_data["mask"]
                        if p_mask[ref_y, ref_x]:
                            clicked_plane_mask = p_mask
                            clicked_plane_id = plane_id
                            clicked_bbox = plane_data["bbox"]
                            break
                            
                    if clicked_plane_id is None and self.wall_base_mask[ref_y, ref_x]:
                        clicked_plane_mask = self.wall_base_mask
                        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
                            self.wall_base_mask.astype(np.uint8), connectivity=8
                        )
                        ref_label = labels_im[ref_y, ref_x]
                        if ref_label > 0:
                            clicked_plane_mask = (labels_im == ref_label)
                            x, y, bw, bh, area = stats[ref_label]
                            clicked_bbox = [x, y, x + bw, y + bh]

                crop_image = self.image_rgb
                crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, w, h
                
                if clicked_bbox is not None:
                    clicked_bbox = [float(clicked_bbox[0]), float(clicked_bbox[1]), float(clicked_bbox[2]), float(clicked_bbox[3])]
                    bx1, by1, bx2, by2 = clicked_bbox
                    bw_box = bx2 - bx1
                    bh_box = by2 - by1
                    pad_x = int(bw_box * 0.10)
                    pad_y = int(bh_box * 0.10)
                    
                    crop_x1 = int(max(0, bx1 - pad_x))
                    crop_y1 = int(max(0, by1 - pad_y))
                    crop_x2 = int(min(w, bx2 + pad_x))
                    crop_y2 = int(min(h, by2 + pad_y))
                    
                    crop_image = self.image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                if point_coords is not None:
                    shifted_pts = []
                    for pt in pts:
                        shifted_pts.append([float(pt[0] - crop_x1), float(pt[1] - crop_y1)])
                    input_points = [[shifted_pts]]
                    
                    if point_labels is None:
                        input_labels = [[[int(1)] * len(shifted_pts)]]
                    else:
                        input_labels = [[[int(l) for l in point_labels]]]
                        
                    inputs = self.sam2_processor(
                        images=crop_image,
                        input_points=input_points,
                        input_labels=input_labels,
                        return_tensors="pt"
                    )
                elif box_coords is not None:
                    shifted_box = [
                        float(box_coords[0] - crop_x1),
                        float(box_coords[1] - crop_y1),
                        float(box_coords[2] - crop_x1),
                        float(box_coords[3] - crop_y1)
                    ]
                    input_boxes = [[[shifted_box]]]
                    inputs = self.sam2_processor(
                        images=crop_image,
                        input_boxes=input_boxes,
                        return_tensors="pt"
                    )
                else:
                    return np.zeros((h, w), dtype=bool)

                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                
                sam_start = time.time()
                with torch.no_grad():
                    outputs = self.sam2_model(**inputs)
                    
                masks_post = self.sam2_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"]
                )[0][0]
                
                masks_np = masks_post.cpu().numpy()
                
                logger.info(f"PERFORMANCE: SAM Time: {time.time() - sam_start:.2f} seconds")
                logger.info("Lifecycle: SAM Completed")
                
                if masks_np is None or masks_np.size == 0:
                    return None
                    
                if level is not None and 0 <= level < 3:
                    sam_crop_mask = masks_np[level]
                else:
                    scores = outputs.iou_scores[0][0].cpu().numpy()
                    best_idx = np.argmax(scores)
                    sam_crop_mask = masks_np[best_idx]
                    
                sam_mask = np.zeros((h, w), dtype=bool)
                sam_mask[crop_y1:crop_y2, crop_x1:crop_x2] = sam_crop_mask
                
                sam_mask_area = np.sum(sam_mask)
                total_area = h * w
                rejected = False
                
                overlap_with_exclusions = np.sum(sam_mask & (self.exclusion_mask > 0))
                if overlap_with_exclusions > (0.02 * sam_mask_area):
                    rejected = True
                    
                if clicked_bbox is not None:
                    bw_box = clicked_bbox[2] - clicked_bbox[0]
                    bh_box = clicked_bbox[3] - clicked_bbox[1]
                    obj_area = bw_box * bh_box
                    if sam_mask_area > total_area * 0.25 and obj_area < total_area * 0.25:
                        rejected = True
                        
                refined_mask = sam_mask
                if rejected or (clicked_plane_mask is not None and sam_mask_area < 0.20 * np.sum(clicked_plane_mask)):
                    refined_mask = clicked_plane_mask
                elif clicked_plane_mask is not None:
                    refined_mask = sam_mask & clicked_plane_mask
                    
                refined_mask = refined_mask.astype(np.uint8)
                refined_mask[self.exclusion_mask > 0] = 0
                
                # Hole filling removed to preserve windows and doors
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
                
                if cleanup:
                    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
                        refined_mask, connectivity=8
                    )
                    if ref_x is not None and ref_y is not None and 0 <= ref_x < w and 0 <= ref_y < h:
                        ref_label = labels_im[int(ref_y), int(ref_x)]
                        if ref_label > 0:
                            refined_mask = (labels_im == ref_label).astype(np.uint8)
                
                final_mask = refined_mask > 0
                
                # Check assertion
                assert final_mask is not None and np.sum(final_mask) > 0, "assert mask_generated failed"
                logger.info("Lifecycle: Mask Generated")
                
                total_click_time = time.time() - total_click_start
                logger.info(f"PERFORMANCE: Total Click Time: {total_click_time:.2f} seconds")
                
                return final_mask
                
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb = traceback.extract_tb(exc_traceback)[-1]
                logger.error(f"generate_mask failed: {e}")
                logger.error(f"Filename: {tb.filename}, Function: {tb.name}, Line: {tb.lineno}")
                print("--- FULL TRACEBACK ---")
                traceback.print_exc(file=sys.stdout)
                print("----------------------")
                sys.exit(1)
