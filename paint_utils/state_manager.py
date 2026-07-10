import streamlit as st
import numpy as np
import cv2
from scipy import sparse
from .performance import cleanup_session_caches, should_trigger_cleanup

def initialize_session_state():
    """Initialize all session state variables with multi-layer safety."""
    defaults = {
        "image": None,          # 640px preview image
        "image_original": None, # Full resolution original
        "front_buffer": None,   # Validated output for display
        "render_error": None,   # Current rendering error state
        "file_name": None,
        "masks": [],
        "masks_redo": [],
        "selection_op": "Add",
        "is_wall_only": False,
        "selection_softness": 0,
        "selection_highlight_opacity": 0.5,
        "zoom_level": 1.0,
        "pan_x": 0.5,
        "pan_y": 0.5,
        "last_click_global": None,
        "mask_level": 0,    # 0, 1, or 2 for granularity
        "selection_tool": "👆 AI Click (Point)",
        "ai_drag_sub_tool": "🆕 Draw New",
        "picked_color": "#8FBC8F",
        "pending_selection": None,
        "pending_boxes": [],
        "render_id": 0,
        "canvas_id": 0,
        "uploader_id": 0,
        "sidebar_p_open": False,
        "last_export": None,
        "selected_layer_idx": None,
        "loop_guarded": False,
        "grayscale_mode": False,     # 🎨 Grayscale Preview Mode
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
    # Restore any states saved before a rerun abort
    restore_sidebar_state()

SIDEBAR_KEYS = [
    "selection_tool", "selection_op", "picked_color", 
    "grayscale_mode", "is_wall_only", "mask_level", 
    "selection_softness", "selection_refinement",
    "selection_highlight_opacity", "fill_selection", "lasso_thickness",
    "selection_finish" 
]

def preserve_sidebar_state():
    """Preserve Streamlit widget states before an abortive rerun."""
    for key in SIDEBAR_KEYS:
        if key in st.session_state:
            st.session_state[f"_saved_{key}"] = st.session_state[key]

def restore_sidebar_state():
    """Restore preserved widget states at the start of a run."""
    for key in SIDEBAR_KEYS:
        if f"_saved_{key}" in st.session_state:
            st.session_state[key] = st.session_state[f"_saved_{key}"]
            del st.session_state[f"_saved_{key}"]

def cb_apply_pending(increment_canvas=True, silent=False):
    if st.session_state.get("pending_selection") is not None:
        new_mask = st.session_state["pending_selection"].copy()
        new_mask.update({
            'color': st.session_state["picked_color"],
            'visible': True,
            'name': f"Layer {len(st.session_state['masks'])+1}",
            'refinement': st.session_state.get("selection_refinement", 0), # Expansion/Contraction (-10 to 10)
            'softness': st.session_state.get("selection_softness", 0),
            'brightness': 0.0, 'contrast': 1.0, 'saturation': 1.0, 'hue': 0.0, 
            'opacity': st.session_state.get("selection_highlight_opacity", 1.0), 
            'finish': st.session_state.get("selection_finish", 'Standard')
        })
        
        # DEBUG: Track operation state
        current_op = st.session_state.get("selection_op")
        num_masks = len(st.session_state["masks"])
        print(f"DEBUG: cb_apply_pending -> Operation: {current_op}, Existing masks: {num_masks}")
        
        # ⚡ MEMORY OPTIMIZATION: Compress mask to sparse matrix for storage
        if not sparse.issparse(new_mask['mask']):
            try:
                new_mask['mask'] = sparse.csc_matrix(new_mask['mask'])
            except Exception as e:
                print(f"WARNING: Sparse compression failed: {e}")
        
        
        # Handle Subtraction Logic
        # Handle Subtraction Logic (Eraser Mode)
        if current_op == "Subtract":
            if st.session_state["masks"]:
                print(f"DEBUG: SUBTRACT mode -> Applying to ALL layers")
                
                total_removed = 0
                cleaned_any = False
                new_selection_mask = new_mask['mask']

                # Iterate through ALL layers to erase from everything
                # Iterate through ALL layers to erase from everything
                for layer in st.session_state["masks"]:
                    if layer.get("visible", True):
                        target_mask = layer['mask']
                        
                        # 🛡️ SAFE DECOMPRESSION: Convert to dense for boolean logic
                        if sparse.issparse(target_mask):
                            target_mask = target_mask.toarray()
                        
                        # Resize if needed (safety check for consistency)
                        dense_new_sel = new_selection_mask
                        if sparse.issparse(dense_new_sel):
                            dense_new_sel = dense_new_sel.toarray()
                            
                        # Resize to match execution context
                        if target_mask.shape != dense_new_sel.shape:
                            resized_new = cv2.resize(dense_new_sel.astype(np.uint8), (target_mask.shape[1], target_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                        else:
                            resized_new = dense_new_sel

                        before_count = np.sum(target_mask)
                        
                        # PERFORM SUBTRACTION (Dense Arrays)
                        layer_mask = target_mask & ~resized_new
                        
                        # ⚡ RE-COMPRESS RESULT
                        layer['mask'] = sparse.csc_matrix(layer_mask)
                        
                        after_count = np.sum(layer_mask)
                        
                        diff = before_count - after_count
                        total_removed += diff
                        if diff > 0:
                            cleaned_any = True

                print(f"DEBUG: Subtraction applied -> Removed {total_removed} pixels total")
                
                # --- USER FEEDBACK (Only if not silent) ---
                if not silent:
                    if not cleaned_any:
                        st.toast("⚠️ selected area didn't overlap with any paint.", icon="ℹ️")
                    else:
                        st.toast("✅ Paint Erased!", icon="🧹")
            else:
                if not silent:
                    st.toast("⚠️ Nothing to erase! The canvas is clean.", icon="✨")
        
        else:
            # ADD Mode (Default)
            merged = False
            if st.session_state["masks"]:
                last_layer = st.session_state["masks"][-1]
                
                # Check if we can merge (same color and properties)
                if (last_layer.get("color") == new_mask["color"] and
                    last_layer.get("finish") == new_mask["finish"] and
                    last_layer.get("opacity") == new_mask["opacity"] and
                    last_layer.get("visible", True) == True):
                    
                    print(f"DEBUG: ADD mode -> Merging with previous layer to prevent seams")
                    
                    # Get masks
                    m1 = last_layer["mask"]
                    m2 = new_mask["mask"]
                    if sparse.issparse(m1): m1 = m1.toarray()
                    if sparse.issparse(m2): m2 = m2.toarray()
                    
                    # Ensure matching dimensions
                    if m1.shape != m2.shape:
                        m2 = cv2.resize(m2.astype(np.uint8), (m1.shape[1], m1.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                        
                    # 1. Union the masks
                    combined = (m1 > 0) | (m2 > 0)
                    
                    # 2. Morphological Closing to fill hairline seams and cracks between adjacent clicks
                    # A 7x7 kernel perfectly seals 1-3 pixel gaps between SAM segments
                    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    closed = cv2.morphologyEx(combined.astype(np.uint8), cv2.MORPH_CLOSE, close_k)
                    
                    # 3. Store back as sparse
                    last_layer["mask"] = sparse.csc_matrix(closed > 0)
                    merged = True
                    
                    # Invalidate layer cache for the merged layer
                    if "flattened_cache" in st.session_state:
                        st.session_state["flattened_cache"] = st.session_state["flattened_cache"][:-1]
                        
            if not merged:
                print(f"DEBUG: ADD mode -> Creating new layer")
                st.session_state["masks"].append(new_mask)
                
            # ── STRICT LAYER MANAGEMENT (No Overpainting) ──
            # Subtract the newly painted area from ALL underlying layers.
            # This ensures each pixel belongs to only ONE paint color, preventing color bleed.
            if st.session_state["masks"]:
                top_layer = st.session_state["masks"][-1]
                top_mask = top_layer["mask"]
                if sparse.issparse(top_mask): top_mask = top_mask.toarray()
                
                # We need to subtract top_mask from all layers except the last one
                filtered_masks = []
                for i in range(len(st.session_state["masks"]) - 1):
                    old_layer = st.session_state["masks"][i]
                    old_mask = old_layer["mask"]
                    if sparse.issparse(old_mask): old_mask = old_mask.toarray()
                    
                    if old_mask.shape != top_mask.shape:
                        old_mask = cv2.resize(old_mask.astype(np.uint8), (top_mask.shape[1], top_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                        
                    # Subtract the new mask
                    new_old_mask = np.logical_and(old_mask > 0, np.logical_not(top_mask > 0))
                    
                    # Only keep the layer if it still has painted pixels
                    if np.any(new_old_mask):
                        old_layer["mask"] = sparse.csc_matrix(new_old_mask)
                        filtered_masks.append(old_layer)
                        
                # Append the top layer back
                filtered_masks.append(top_layer)
                st.session_state["masks"] = filtered_masks
                
                # Invalidate cache
                if "flattened_cache" in st.session_state:
                    st.session_state["flattened_cache"] = []
                st.session_state["_layer_cache"] = None
            
        st.session_state["masks_redo"] = [] # Clear redo stack on new action
        st.session_state["pending_selection"] = None
        st.session_state["pending_boxes"] = []
        st.session_state["render_id"] += 1
        
        # ⚡ OPTIMIZATION: Allow skipping canvas reset for smooth continuous clicking
        if increment_canvas:
            st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
            
        st.session_state["canvas_raw"] = {} # Force clear cached objects
        st.session_state["just_applied"] = True # 🛡️ Guard against object persistence loops


def cb_cancel_pending():
    st.session_state["pending_selection"] = None
    st.session_state["pending_boxes"] = []
    st.session_state["render_id"] += 1
    st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
    st.session_state["canvas_raw"] = {} # Force clear cached objects

def cb_undo():
    """Undo last paint layer with automatic memory cleanup."""
    if st.session_state["masks"]:
        last_mask = st.session_state["masks"].pop()
        st.session_state["masks_redo"].append(last_mask)
        st.session_state["render_id"] += 1
        st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
        
        # Check if cleanup needed after undo
        if should_trigger_cleanup():
            cleanup_session_caches(aggressive=False)

def cb_redo():
    """Redo the last undone paint layer."""
    if st.session_state.get("masks_redo"):
        mask = st.session_state["masks_redo"].pop()
        st.session_state["masks"].append(mask)
        st.session_state["render_id"] += 1
        st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1

def cb_clear_all():
    """Clear all paint layers and perform memory cleanup."""
    st.session_state["masks"] = []
    st.session_state["masks_redo"] = []
    
    # Aggressive cleanup when clearing all
    cleanup_session_caches(aggressive=True)
    st.session_state["render_id"] += 1
    st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1

def cb_delete_layer(idx):
    if st.session_state.get("masks") and 0 <= idx < len(st.session_state["masks"]):
        st.session_state["masks"].pop(idx)
        st.session_state["selected_layer_idx"] = None
        st.session_state["render_id"] += 1
        st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
