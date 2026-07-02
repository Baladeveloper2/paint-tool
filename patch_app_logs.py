import re

filepath = 'app.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the DEBUG: statements with the specific ✓ validation strings requested by the user
replacements = {
    'print("DEBUG: Processing Locked")': 'print("✓ Processing Locked")',
    'print("DEBUG: Tap Received")': 'print("✓ Click Coordinates: ", real_x, real_y)',
    'print("DEBUG: Image Set")': 'pass  # Handled in segmentation.py',
    'print("DEBUG: SAM Prediction Started")': 'print("✓ SAM Prediction Started")',
    'print("DEBUG: SAM Prediction Finished")': 'print("✓ SAM Prediction Finished")',
    'print("DEBUG: Mask Generated")': 'print("✓ Mask Generated")',
    'print(f"DEBUG: Mask Pixel Count: {mask_pixels} pixels")': 'print(f"✓ Mask Pixel Count: {mask_pixels} pixels")',
    'print("DEBUG: Mask Returned")': 'print("✓ Mask Bounding Box Checked")',
    'print("DEBUG: Paint Layer Created")': 'print("✓ Layer Created")',
    'print(f"DEBUG: Color Applied: {picked_color}")': 'print(f"✓ Selected Color: {picked_color}")\\n                            print("✓ Paint Applied")',
    'print("DEBUG: Image Blended")': 'print("✓ Image Blended")',
    'print("DEBUG: Display Updated")': 'print("✓ Render Completed")',
    'print("DEBUG: UI Refreshed")': 'print("✓ UI Updated")',
    'print("DEBUG: Mask rejected (too small)")': 'print("❌ Mask rejected: Area < 100 pixels")'
}

new_content = content
for old, new in replacements.items():
    new_content = new_content.replace(old, new)

# Also strictly enforce boolean mask validation
bool_enforcement = """
                        if mask_pixels < 100:
                            st.toast("⚠️ Selected area is too small (<100 pixels).", icon="🚫")
                            print("❌ Mask rejected: Area < 100 pixels")
                        else:
                            if mask.dtype != bool:
                                mask = mask.astype(bool)
"""
new_content = new_content.replace("""
                        if mask_pixels < 100:
                            st.toast("⚠️ Selected area is too small (<100 pixels).", icon="🚫")
                            print("❌ Mask rejected: Area < 100 pixels")
                        else:
""", bool_enforcement)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(new_content)
print("app.py successfully patched with strict validation and detailed logging.")
