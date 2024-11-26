import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import time

import cv2
import torch
import requests
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

class GroundedSAM:
    def __init__(self, obj_detection_model, segmentation_model, device="cuda:0"):
        self.device = device
        self.object_detector = pipeline(model=obj_detection_model, task="zero-shot-object-detection", device=device)
        self.object_segmentor = AutoModelForMaskGeneration.from_pretrained(segmentation_model).to(device)
        self.processor = AutoProcessor.from_pretrained(segmentation_model)
        self.lower_green_filter = np.array([35, 50, 50])
        self.upper_green_filter = np.array([85, 255, 255])
                
    def get_boxes(self, results: DetectionResult) -> List[List[List[float]]]:
        boxes = []
        for result in results:
            xyxy = result.box.xyxy
            boxes.append(xyxy)

        return [boxes]
    
    def mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        polygon = largest_contour.reshape(-1, 2).tolist()
        return polygon

    def polygon_to_mask(self, polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=(255,))
        return mask
    
    def refine_masks(self, masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        if polygon_refinement:
            for idx, mask in enumerate(masks):
                shape = mask.shape
                polygon = self.mask_to_polygon(mask)
                mask = self.polygon_to_mask(polygon, shape)
                masks[idx] = mask
        return masks
    
    def compute_yaw_and_com(self, pts):
        centroid = np.mean(pts, axis=0)
        centered_pts = pts - centroid
        cov_matrix = np.cov(centered_pts, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # The principal axis is the eigenvector corresponding to the largest eigenvalue
        principal_axis = eigenvectors[:, -1]
        
        # Calculate the angle (yaw) as the arctangent of (dy/dx) between the principal axis and the x-axis 
        yaw = np.arctan2(principal_axis[1], principal_axis[0])
        return yaw, centroid
    
    def detect_obj_from_labels(self, image, labels:List[str], threshold:float=0.3) -> List[DetectionResult]:
        # TODO: See if this works. If not, add Image.fromarray(image) to the input of the object detector
        labels = [label if label.endswith(".") else label+"." for label in labels]
        results = self.object_detector(image,  candidate_labels=labels, threshold=threshold)
        return [DetectionResult.from_dict(result) for result in results]
    
    def segment_obj_from_bb(self, image, BBs:List[Dict[str, Any]], polygon_refinement=False) -> np.array:
        boxes = self.get_boxes(BBs)
        inputs = self.processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.device)

        outputs = self.object_segmentor(**inputs)
        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0] 

        masks = self.refine_masks(masks, polygon_refinement)
        return masks
        
    def grounded_obj_segmentation(self, image:np.array, labels:List[str], threshold:float=0.5, polygon_refinement=True) -> List[np.array]:
        print("HAKUNA")
        # TODO: See if this works. If not, add Image.fromarray(image) to the input of the object detector
        results = self.detect_obj_from_labels(Image.fromarray(image), labels, threshold)
        masks = self.segment_obj_from_bb(Image.fromarray(image), results, polygon_refinement)
        
        og_mask = masks[0]
        image = cv2.bitwise_and(image, image, mask = og_mask)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        hsv_mask = cv2.inRange(image, self.lower_green_filter, self.upper_green_filter)
        og_mask = cv2.bitwise_and(og_mask, og_mask, mask = hsv_mask)
        
        boundary = cv2.Canny(og_mask,100,200)
        boundary_pts = np.array(np.where(boundary==255)).T
        return boundary_pts.tolist()