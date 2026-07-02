import re

filepath = 'app.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Target exactly the TAP Fast Path (starts with "DEBUG: Processing Mobile Tap (AI)")
# To be safe, we will replace the block inside the `else:` branch of the `async_status` for tap.
# Let's find the exact block for the Tap fast path.
tap_sync_pattern = r'# 🚀 FAST PATH: If embeddings are ready.*?if getattr\(sam, \"is_image_set\", False\) and getattr\(sam, \"image_rgb\", None\) is not None:.*?st\.rerun\(\)'

tap_sync_repl = '''# 🚀 FAST PATH: If embeddings are ready, run synchronously
                    if getattr(sam, "is_image_set", False) and getattr(sam, "image_rgb", None) is not None:
                        print("DEBUG: Click received")
                        print("DEBUG: Image embedding ready")
                        
                        current_tool = st.session_state.get("selection_tool", "")
                        is_wall_click_mode = "Wall Click" in current_tool
                        is_wall_mode = st.session_state.get("is_wall_only", False)
                        
                        mask = sam.generate_mask(
                            point_coords=[real_x, real_y], 
                            level=st.session_state.get("mask_level", 0), 
                            is_wall_only=is_wall_mode,
                            is_wall_click=is_wall_click_mode
                        )
                        
                        if mask is not None:
                            print("DEBUG: Mask generated")
                            mask_pixels = mask.sum() if hasattr(mask, "sum") else 0
                            print(f"DEBUG: Mask size: {mask_pixels} pixels")
                            
                            if mask_pixels < 100:
                                st.toast("⚠️ Selected area is too small (<100 pixels).", icon="🚫")
                                print("DEBUG: Mask rejected (too small)")
                                if "tap" in st.query_params: del st.query_params["tap"]
                                st.rerun()
                                
                            picked_color = st.session_state.get("picked_color", "#8FBC8F")
                            print(f"DEBUG: Color selected: {picked_color}")
                                
                            st.session_state["masks"] = []
                            
                            print("DEBUG: Paint applied")
                            st.session_state["pending_selection"] = {'mask': mask, 'point': (real_x, real_y)}
                            st.session_state["selection_op"] = "Add"
                            cb_apply_pending(increment_canvas=False, silent=True)
                            st.session_state["render_id"] += 1
                            
                            print("DEBUG: Image blended")
                            print("DEBUG: UI updated")
                        else:
                            st.toast("Mask generated but paint application failed.", icon="⚠️")
                            print("DEBUG: Mask generation failed or returned None.")
                        
                        if "tap" in st.query_params: 
                            del st.query_params["tap"]
                        st.rerun()'''

# The pattern needs to be non-greedy to avoid capturing the end of the file
content = re.sub(tap_sync_pattern, tap_sync_repl, content, count=1, flags=re.DOTALL)

# Target exactly the TAP Async success block
tap_async_pattern = r'elif isinstance\(async_status, dict\) and async_status\.get\(\"status\"\) == \"success\":\s*# Success!\s*mask = async_status\.get\(\"mask\"\)\s*if mask is not None:.*?st\.session_state\[\"render_id\"\] \+= 1\s*# Clear Param'

tap_async_repl = '''elif isinstance(async_status, dict) and async_status.get("status") == "success":
             print("DEBUG: Click received")
             mask = async_status.get("mask")
             
             if mask is not None:
                print("DEBUG: Mask generated")
                mask_pixels = mask.sum() if hasattr(mask, "sum") else 0
                print(f"DEBUG: Mask size: {mask_pixels} pixels")
                
                if mask_pixels < 100:
                    st.toast("⚠️ Selected area is too small (<100 pixels).", icon="🚫")
                    print("DEBUG: Mask rejected (too small)")
                    if "tap" in st.query_params: del st.query_params["tap"]
                    st.rerun()
                    
                picked_color = st.session_state.get("picked_color", "#8FBC8F")
                print(f"DEBUG: Color selected: {picked_color}")
                
                st.session_state["masks"] = []
                
                parts = tap_param.split(",")
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

                print("DEBUG: Paint applied")
                st.session_state["pending_selection"] = {'mask': mask, 'point': (real_x, real_y)}
                st.session_state["selection_op"] = "Add"
                cb_apply_pending(increment_canvas=False, silent=True)
                st.session_state["render_id"] += 1
                
                print("DEBUG: Image blended")
                print("DEBUG: UI updated")
             else:
                 st.toast("Mask generated but paint application failed.", icon="⚠️")
                 
             # Clear Param'''

content = re.sub(tap_async_pattern, tap_async_repl, content, count=1, flags=re.DOTALL)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('Patch safely applied to app.py')
