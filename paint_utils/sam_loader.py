import os
import torch
import streamlit as st
import time
import logging
from paint_core.ai_engine import AIEngine
from app_config.constants import PerformanceConfig

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
MODEL_TYPE = PerformanceConfig.SAM_MODEL_TYPE
CHECKPOINT_PATH = PerformanceConfig.SAM_CHECKPOINT_PATH

def load_global_models(device_str):
    """
    Requirement 1, 9, 16: Load YOLO, SAM2, Mask2Former and all AI models only once.
    Returns a dictionary of singleton AI model instances.
    """
    # This is deprecated in favor of AIEngine singleton but keeping stub if called
    pass

def get_sam_engine(checkpoint_path=CHECKPOINT_PATH, model_type=MODEL_TYPE):
    """
    Get the production AI selection engine stored in singleton.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AIEngine.get_instance(device=device)

def ensure_model_exists():
    pass
