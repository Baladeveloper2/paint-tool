import re

filepath = 'app.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# We want to replace everything from `# --- AI POINT HANDLER ---` down to the end of its `except Exception as e:` block.
# Since line numbers can shift, we use a regex that captures this exact block.
pattern = r'# --- AI POINT HANDLER ---.*?except Exception as e:\s*print\(f\"DEBUG: Tap Setup Error: \{e\}\"\)\s*if \"tap\" in st\.query_params:\s*del st\.query_params\[\"tap\"\]'

replacement = '''# --- AI POINT HANDLER ---
        print(f"DEBUG: Processing Mobile Tap (AI) -> {tap_param}")
        
        if st.session_state.get("ai_processing", False):
            print("DEBUG: Duplicate Tap Ignored")
            if "tap" in st.query_params: 
                del st.query_params["tap"]
            st.stop()
            
        st.session_state["ai_processing"] = True
        
        try:
            with st.spinner("👆 AI is analyzing object..."):
                print("DEBUG: Processing Locked")
                
                parts = tap_param.split(",")
                if len(parts) >= 2:
                    x, y = int(parts[0].strip()), int(parts[1].strip())
                    img = st.session_state["image"]
                    h, w = img.shape[:2]
                    display_width = 800
                    zoom = st.session_state.get("zoom_level", 1.0)
                    pan_x = st.session_state.get("pan_x", 0.5)
                    pan_y = st.session_state.get("pan_y", 0.5)
                    start_x, start_y, view_w, view_h = get_crop_params(w, h, zoom, pan_x, pan_y)
                    scale_factor = display_width / view_w
                    
                    real_x = int(x / scale_factor) + start_x
                    real_y = int(y / scale_factor) + start_y
                    
                    print("DEBUG: Tap Received")
                    
                    # 1. Set Image (Synchronous)
                    if not getattr(sam, "is_image_set", False) or getattr(sam, "image_rgb", None) is None:
                        sam.set_image(img)
                    print("DEBUG: Image Set")
                    
                    # 2. SAM Prediction
                    print("DEBUG: SAM Prediction Started")
                    current_tool = st.session_state.get("selection_tool", "")
                    is_wall_click_mode = "Wall Click" in current_tool
                    is_wall_mode = st.session_state.get("is_wall_only", False)
                    
                    mask = sam.generate_mask(
                        point_coords=[real_x, real_y], 
                        level=st.session_state.get("mask_level", 0), 
                        is_wall_only=is_wall_mode,
                        is_wall_click=is_wall_click_mode
                    )
                    print("DEBUG: SAM Prediction Finished")
                    
                    if mask is not None:
                        print("DEBUG: Mask Generated")
                        mask_pixels = mask.sum() if hasattr(mask, "sum") else 0
                        print(f"DEBUG: Mask Pixel Count: {mask_pixels} pixels")
                        
                        if mask_pixels < 100:
                            st.toast("⚠️ Selected area is too small (<100 pixels).", icon="🚫")
                            print("DEBUG: Mask rejected (too small)")
                        else:
                            print("DEBUG: Mask Returned")
                            picked_color = st.session_state.get("picked_color", "#8FBC8F")
                            print(f"DEBUG: Color Applied: {picked_color}")
                            
                            st.session_state["masks"] = []
                            print("DEBUG: Paint Layer Created")
                            
                            st.session_state["pending_selection"] = {'mask': mask, 'point': (real_x, real_y)}
                            st.session_state["selection_op"] = "Add"
                            cb_apply_pending(increment_canvas=False, silent=True)
                            st.session_state["render_id"] += 1
                            
                            print("DEBUG: Image Blended")
                            print("DEBUG: Display Updated")
                    else:
                        st.toast("Mask generated but paint application failed.", icon="⚠️")
                        print("DEBUG: Mask generation failed or returned None.")
                        
        except Exception as e:
            print(f"DEBUG: Tap Pipeline Error: {e}")
        finally:
            print("DEBUG: UI Refreshed")
            print("DEBUG: Processing Unlocked")
            st.session_state["ai_processing"] = False
            if "tap" in st.query_params: 
                del st.query_params["tap"]
            st.rerun()'''

new_content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(new_content)
print('Synchronous pipeline patched into app.py successfully.')
