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

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def find_chevron_contour(mask, min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    max_area = 0
    best_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area > max_area:
            max_area = area
            best_contour = contour
    return best_contour

def compute_yaw(chevron_contour):
    [vx, vy, x0, y0] = cv2.fitLine(chevron_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan2(vy, vx)  # Yaw in radians
    return angle

def compute_yaw_with_pca(contour, center):
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts_centered = pts - center
    cov = np.cov(pts_centered, rowvar=False)
    eigenvals, eigenvects = np.linalg.eig(cov)
    idx_max = np.argmax(eigenvals)
    principal_axis = eigenvects[:, idx_max]
    
    yaw = np.arctan2(principal_axis[1], principal_axis[0])
    return normalize_angle(yaw + np.pi/2)

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

class VisUtils:
    def __init__(self, obj_detection_model, segmentation_model, device="cuda:0", traditional=True, plane_size=None):
        self.device = device
        
        if not traditional:
            self.object_detector = pipeline(model=obj_detection_model, task="zero-shot-object-detection", device=device)
            self.object_segmentor = AutoModelForMaskGeneration.from_pretrained(segmentation_model).to(device)
            self.processor = AutoProcessor.from_pretrained(segmentation_model)
        
        self.lower_green_filter = np.array([35, 50, 50])
        self.upper_green_filter = np.array([85, 255, 255])
        self.kernel = np.ones((11,11), np.uint8)
        self.lower_red1 = np.array([0, 50, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])
        self.img_size = np.array((1080, 1920))
        self.plane_size = plane_size
        # self.plane_size = np.array([(0.009, -0.376),(0.24200, 0.034)])
        self.delta_plane_x = self.plane_size[1][0] - self.plane_size[0][0]
        self.delta_plane_y = self.plane_size[1][1] - self.plane_size[0][1]
        self.delta_plane = np.array((self.delta_plane_x, -self.delta_plane_y))
        self.traditional = traditional
    
    def find_chevron_contour(self, mask, min_area=10):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        max_area = 0
        best_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area and area > max_area:
                max_area = area
                best_contour = contour
        return best_contour

    def find_chevron_tip_and_base(self, approx_polygon):
        pts = approx_polygon.reshape(-1, 2)
        
        def angle_at(i):
            prev_i = (i - 1) % len(pts)
            next_i = (i + 1) % len(pts)
            v1 = pts[prev_i] - pts[i]
            v2 = pts[next_i] - pts[i]
            
            dot = v1.dot(v2)
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)
            cos_angle = dot / (mag1 * mag2 + 1e-9)
            return np.arccos(np.clip(cos_angle, -1, 1))

        angles = [angle_at(i) for i in range(len(pts))]
        
        i_tip = np.argmin(angles)
        tip_point = pts[i_tip]
        base_points = np.delete(pts, i_tip, axis=0)
        base_center = np.mean(base_points, axis=0)
        return tip_point, base_center

    def compute_yaw(self, chevron_contour):
        [vx, vy, x0, y0] = cv2.fitLine(chevron_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.arctan2(vy, vx)  # Yaw in radians
        return angle

    def approximate_chevron(self, contour, corner_count=6):
        if contour is None:
            return None
        
        peri = cv2.arcLength(contour, True)
        best_approx = None
        eps = 0.04 * peri
        while eps > 0.001 * peri:
            approx = cv2.approxPolyDP(contour, eps, True)
            if len(approx) == corner_count:
                best_approx = approx
                break
            elif len(approx) < corner_count:
                eps *= 0.9
            else:
                eps *= 1.1

        return best_approx

    def get_chevron_yaw(self, hsv, com):
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = mask1 | mask2
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        chevron_contour = self.find_chevron_contour(mask)
        
        approx = self.approximate_chevron(chevron_contour, corner_count=6)
        tip, base_center = self.find_chevron_tip_and_base(approx)
        
        dx = base_center[0] - tip[0]
        dy = base_center[1] - tip[1]
        
        yaw = normalize_angle(-np.arctan2(dy, dx))
        return yaw
                
    
    def convert_world_2_pix(self, vecs):
        result = np.zeros((vecs.shape[0], 2))
        result[:, 0] = (vecs[:, 0] - self.plane_size[0][0]) / self.delta_plane_x * 1080
        result[:, 1] = (vecs[:, 1] - self.plane_size[0][1]) / self.delta_plane_y * 1920 + 1920
        return result
    
    def convert_pix_2_world(self, vecs):
        result = np.zeros((vecs.shape[0], 2))
        result[:, 0] = vecs[:, 0] / 1080 * self.delta_plane_x + self.plane_size[0][0]
        result[:, 1] = (1920 - vecs[:, 1]) / 1920 * self.delta_plane_y + self.plane_size[0][1]
        return result
    
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
        results = self.detect_obj_from_labels(Image.fromarray(image), labels, threshold)
        masks = self.segment_obj_from_bb(Image.fromarray(image), results, polygon_refinement)
        
        og_mask = masks[0]
        image = cv2.bitwise_and(image, image, mask = og_mask)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        hsv_mask = cv2.inRange(image, self.lower_green_filter, self.upper_green_filter)
        og_mask = cv2.bitwise_and(og_mask, og_mask, mask = hsv_mask)
        
        boundary = cv2.Canny(og_mask,100,200)
        # plt.imshow(boundary)
        # plt.show()
        boundary_pts = np.array(np.where(boundary==255)).T
        return boundary_pts
    
    def get_bdpts_and_pose(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if self.traditional:
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            
            _, seg_map = cv2.threshold(mask, 1, 250, cv2.THRESH_BINARY)
            seg = cv2.morphologyEx(seg_map, cv2.MORPH_OPEN, self.kernel)
            seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, self.kernel)
            boundary = cv2.Canny(seg_map,100,200)
            bd_pts = np.array(np.where(boundary==255)).T
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bd_pts = self.grounded_obj_segmentation(img, labels=["green block"], threshold=0.3,  polygon_refinement=True)
        
        bd_pts = self.convert_pix_2_world(bd_pts)
        
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = mask1 | mask2
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        com = np.mean(bd_pts, axis=0)
        return bd_pts, com

    def get_bdpts(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if self.traditional:
            mask = cv2.inRange(hsv, self.lower_green_filter, self.upper_green_filter)
            
            _, seg_map = cv2.threshold(mask, 1, 250, cv2.THRESH_BINARY)
            seg = cv2.morphologyEx(seg_map, cv2.MORPH_OPEN, self.kernel)
            seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, self.kernel)
            boundary = cv2.Canny(seg_map, 100, 200)
            bd_pts_pixel = np.array(np.where(boundary == 255)).T
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bd_pts_pixel = self.grounded_obj_segmentation(
                img_rgb, 
                labels=["green block"], 
                threshold=0.3, 
                polygon_refinement=True
            )

        bd_pts = self.convert_pix_2_world(bd_pts_pixel)
        return bd_pts