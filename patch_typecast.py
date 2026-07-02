import re

filepath = 'paint_core/segmentation.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Cast bounding box
target_bbox = '''            if clicked_bbox is not None:
                # Pad bounding box by 10%'''
replacement_bbox = '''            if clicked_bbox is not None:
                clicked_bbox = [int(clicked_bbox[0]), int(clicked_bbox[1]), int(clicked_bbox[2]), int(clicked_bbox[3])]
                # Pad bounding box by 10%'''
content = content.replace(target_bbox, replacement_bbox)

# 2. Cast points
target_pts = '''            if point_coords is not None:
                shifted_pts = []
                for pt in pts:
                    shifted_pts.append([pt[0] - crop_x1, pt[1] - crop_y1])'''
replacement_pts = '''            if point_coords is not None:
                shifted_pts = []
                for pt in pts:
                    # Enforce strict standard python int casting for SAM2 prediction
                    shifted_pts.append([int(pt[0]) - int(crop_x1), int(pt[1]) - int(crop_y1)])'''
content = content.replace(target_pts, replacement_pts)

# 3. Cast shifted box
target_shifted_box = '''            elif box_coords is not None:
                shifted_box = [
                    box_coords[0] - crop_x1,
                    box_coords[1] - crop_y1,
                    box_coords[2] - crop_x1,
                    box_coords[3] - crop_y1
                ]'''
replacement_shifted_box = '''            elif box_coords is not None:
                shifted_box = [
                    int(box_coords[0]) - int(crop_x1),
                    int(box_coords[1]) - int(crop_y1),
                    int(box_coords[2]) - int(crop_x1),
                    int(box_coords[3]) - int(crop_y1)
                ]'''
content = content.replace(target_shifted_box, replacement_shifted_box)

# 4. Add type logging and failsafe
target_logs = '''            # Forward pass on SAM2
            for k, v in inputs.items():'''
replacement_logs = '''            # Forward pass on SAM2
            # Type logging as requested
            print(type(ref_x))
            print(type(ref_y))
            if clicked_bbox is not None: print(type(clicked_bbox[0]))
            if point_coords is not None: print(type(input_points[0][0][0][0]))
            
            # Failsafe to guarantee no np.int64 leaks
            for k, v in inputs.items():'''
content = content.replace(target_logs, replacement_logs)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("segmentation.py successfully patched with explicit integer casting and type logging.")
