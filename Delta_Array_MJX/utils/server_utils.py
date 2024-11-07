from fastapi import FastAPI
from pydantic import BaseModel
import torch

from utils.vision_utils import GroundedSAM

class DeltaArrayServer():
    def __init__(self, vlm_args, rl_args, device):
        self.vision_model = GroundedSAM(obj_detection_model=vlm_args['obj_detection_model'], 
                                        segmentation_model=vlm_args['segmentation_model'],
                                        device=device)
        