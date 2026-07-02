import os
import sys
import traceback
import numpy as np
import cv2

from paint_utils.sam_loader import get_sam_engine

def main():
    try:
        print("Initializing engine...")
        engine = get_sam_engine()
        
        # Create a mock image (e.g. 1000x1000 white image)
        image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        print("Setting image...")
        engine.set_image(image)
        
        print("Generating mask...")
        # Simulate the user's click coordinates that triggered the error
        point_coords = [504, 165]
        mask = engine.generate_mask(point_coords=point_coords)
        
        print("Success! Mask generated.")
        
    except Exception as e:
        print("--- FULL TRACEBACK ---")
        traceback.print_exc(file=sys.stdout)
        print("----------------------")

if __name__ == "__main__":
    main()
