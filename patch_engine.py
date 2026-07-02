import re
import os

filepath = 'paint_core/segmentation.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add imports
if 'import hashlib' not in content:
    content = content.replace('import numpy as np', 'import hashlib\nimport threading\nimport time\nimport numpy as np')

# 2. Update __init__
init_pattern = r'def __init__\(self, checkpoint_path=\"DEFAULT_INTERNAL\", model_type=\"vit_b\", device=None, model_instance=None\):'
init_repl = 'def __init__(self, checkpoint_path="DEFAULT_INTERNAL", model_type="vit_b", device=None, model_instance=None, models_dict=None):'
content = re.sub(init_pattern, init_repl, content)

if 'self.inference_lock = threading.Lock()' not in content:
    device_pattern = r'self\.device = device\s+logger\.info'
    device_repl = 'self.device = device\n        \n        self.inference_lock = threading.Lock()\n        self.current_image_hash = None\n        \n        logger.info'
    content = re.sub(device_pattern, device_repl, content)

# 3. Update __init__ models loading
yolo_pattern = r'self\.yolo = YOLO\(\"yolov8n-seg\.pt\"\)'
yolo_repl = '''if models_dict is not None:
                self.yolo = models_dict['yolo']
                self.mask2former_processor = models_dict['m2f_processor']
                self.mask2former_model = models_dict['m2f_model']
                self.sam2_processor = models_dict['sam2_processor']
                self.sam2_model = models_dict['sam2_model']
            else:
                self.yolo = YOLO("yolov8n-seg.pt")'''
content = content.replace(yolo_pattern, yolo_repl)

# 4. Update set_image
set_image_pattern = r'def set_image\(self, image_rgb\):.*?logger\.info\(\"Setting image on segmentation engine\.\.\.\"\)'
set_image_repl = '''def set_image(self, image_rgb):
        """
        Process the image through YOLOv8 and Mask2Former to compute static masks.
        """
        if not isinstance(image_rgb, np.ndarray):
            raise ValueError("image_rgb must be 3-channel")
        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be 3-channel")
            
        img_hash = hashlib.md5(image_rgb.tobytes()).hexdigest()
        if self.current_image_hash == img_hash and self.is_image_set:
            logger.info("Image hash match. Skipping redundant set_image computations.")
            return
            
        with self.inference_lock:
            start_time = time.time()
            logger.info("Setting image on segmentation engine...")'''
content = re.sub(set_image_pattern, set_image_repl, content, flags=re.DOTALL)

# Wrap set_image rest
end_set_image_repl = '''self.wall_planes[plane_id] = {
                    "mask": plane_mask,
                    "bbox": [x1, y1, x2, y2]
                }
            
            self.current_image_hash = img_hash
            self.is_image_set = True
            
            embed_time = time.time() - start_time
            logger.info(f"PERFORMANCE: Image Embedding Time: {embed_time:.2f} seconds")'''
content = content.replace('                self.wall_planes[plane_id] = {\n                    "mask": plane_mask,\n                    "bbox": [x1, y1, x2, y2]\n                }', end_set_image_repl)

# 5. Update generate_mask
gen_mask_pattern = r'def generate_mask\(self, point_coords=None, point_labels=None, box_coords=None,\s*level=None, is_wall_only=False, is_wall_click=False, cleanup=True\):.*?if not self\.is_image_set:'
gen_mask_repl = '''def generate_mask(self, point_coords=None, point_labels=None, box_coords=None, 
                      level=None, is_wall_only=False, is_wall_click=False, cleanup=True):
        
        with self.inference_lock:
            start_time = time.time()
            if not self.is_image_set:'''
content = re.sub(gen_mask_pattern, gen_mask_repl, content, flags=re.DOTALL)

# Add performance log to end of generate_mask
if 'PERFORMANCE: Inference Time' not in content:
    content = content.replace('return refined_mask > 0', 'inf_time = time.time() - start_time\n            logger.info(f"PERFORMANCE: Inference Time: {inf_time:.2f} seconds")\n            return refined_mask > 0')
    content = content.replace('return sam_mask', 'inf_time = time.time() - start_time\n                logger.info(f"PERFORMANCE: Inference Time: {inf_time:.2f} seconds")\n                return sam_mask')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('Patch complete.')
