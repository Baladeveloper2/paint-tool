import re

filepath = 'paint_core/segmentation.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Cast bounding box padding and bounds to integers, and add strict assertions and logging before the OpenCV slice
target_crop = '''                pad_x = int(bw * 0.10)
                pad_y = int(bh * 0.10)
            
                crop_x1 = max(0, bx1 - pad_x)
                crop_y1 = max(0, by1 - pad_y)
                crop_x2 = min(w, bx2 + pad_x)
                crop_y2 = min(h, by2 + pad_y)
            
                crop_image = self.image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]'''

replacement_crop = '''                pad_x = int(bw * 0.10)
                pad_y = int(bh * 0.10)
            
                # Explicitly cast to native integers to prevent slice errors
                crop_x1 = int(max(0, bx1 - pad_x))
                crop_y1 = int(max(0, by1 - pad_y))
                crop_x2 = int(min(w, bx2 + pad_x))
                crop_y2 = int(min(h, by2 + pad_y))
                
                # Pre-crop logging as requested
                print(f"✓ Image Shape: {h}x{w}")
                print(f"✓ Crop Coordinates (int): x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}")
                print(f"✓ Crop Size: {crop_x2-crop_x1}x{crop_y2-crop_y1}")
                print(f"✓ Coordinate Types: {[type(crop_x1), type(crop_y1), type(crop_x2), type(crop_y2)]}")
                
                # Failsafe bounds assertions
                assert isinstance(crop_x1, int), "Crop x1 must be integer"
                assert crop_x1 < crop_x2, "Invalid crop width"
                assert crop_y1 < crop_y2, "Invalid crop height"
                assert crop_x1 >= 0 and crop_x2 <= w, "Crop out of horizontal bounds"
                assert crop_y1 >= 0 and crop_y2 <= h, "Crop out of vertical bounds"
            
                crop_image = self.image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]'''
content = content.replace(target_crop, replacement_crop)

# 2. Prevent None returns implicitly crashing the pipeline downstream
target_post_process = '''            masks_post = self.sam2_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"]
            )[0][0]
        
            masks_np = masks_post.cpu().numpy() # shape (3, Crop_H, Crop_W)
        
            # Select best mask level'''

replacement_post_process = '''            masks_post = self.sam2_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"]
            )[0][0]
        
            masks_np = masks_post.cpu().numpy() # shape (3, Crop_H, Crop_W)
            
            if masks_np is None or masks_np.size == 0:
                print("❌ SAM2 returned empty mask")
                return None
        
            # Select best mask level'''
content = content.replace(target_post_process, replacement_post_process)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("segmentation.py integer slice bounding constraints successfully applied.")
