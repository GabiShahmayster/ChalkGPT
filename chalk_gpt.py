
import dataclasses
import enum
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import string
import cv2
import numpy as np
import torch
import tqdm
from img2vec_pytorch import Img2Vec
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
from ultralytics.engine.results import Results
from src.mouse_click import select_pixel
from src.base import KeypointData, KeypointMatchingResults
from src.holds_clustering import cluster_images, visualize_clusters
from src.superglue import SuperGlue
from src.superpoint import SuperPoint
from src.video_writer import VideoWriterChalkGpt, add_clock_to_image
from vector_db import FAISSIndex
from video_extractor import extract_video_frames
from shapely.geometry import Polygon, MultiPolygon
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from shapely.geometry import Polygon
import shapely.affinity as affinity


def transform_polygon(polygon, m):
    """
    Apply an affine transformation to a Shapely polygon.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon
        The polygon to transform.
    matrix : list or tuple, optional
        A 6-element list or tuple specifying the affine transformation matrix
        in the order [a, b, d, e, xoff, yoff] for the equation:
        x' = a*x + b*y + xoff
        y' = d*x + e*y + yoff
    Returns:
    --------
    shapely.geometry.Polygon
        The transformed polygon.
    """
    # Apply affine transform using a 6-element matrix
    mat_elements = m[0,0], m[0, 1], m[1, 0], m[1, 1], m[0, 2], m[1, 2]
    try:
        return affinity.affine_transform(polygon, mat_elements)
    except:
        return None

def bbox_to_polygon(top_left, bottom_right):
    """
    Convert a bounding box defined by its top-left and bottom-right corners to a Shapely polygon.

    Parameters:
    -----------
    top_left : tuple
        (x, y) coordinates of the top-left corner of the bounding box.
    bottom_right : tuple
        (x, y) coordinates of the bottom-right corner of the bounding box.

    Returns:
    --------
    shapely.geometry.Polygon
        Shapely polygon representing the bounding box.
    """
    # Extract coordinates
    x_min, y_min = top_left
    x_max, y_max = bottom_right

    # Create polygon coordinates (going clockwise from top-left)
    coords = [
        (x_min, y_min),  # top-left
        (x_max, y_min),  # top-right
        (x_max, y_max),  # bottom-right
        (x_min, y_max),  # bottom-left
        (x_min, y_min)  # closing the loop by returning to top-left
    ]

    # Create and return the polygon
    return Polygon(coords)

class BoundingBox:
    top_left: tuple
    bottom_right: tuple
    polygon: Polygon
    width_px: int = None
    height_px: int = None

    def __init__(self, top_left: tuple, bottom_right: tuple, width_px: int = None, height_px: int = None):
        self.top_left = top_left
        self.bottom_right =bottom_right
        self.width_px = width_px
        self.height_px = height_px
        self.polygon = bbox_to_polygon(top_left=top_left, bottom_right=bottom_right)

    def __post_init__(self):
        self.width_px = self.bottom_right[0] - self.top_left[0]
        self.height_px = self.bottom_right[1] - self.top_left[1]

    def apply_transform(self, T: np.ndarray):
        tl_h: np.ndarray = np.array([self.top_left[0], self.top_left[1], 1.0]).reshape((3, 1))
        br_h: np.ndarray = np.array([self.bottom_right[0], self.bottom_right[1], 1.0]).reshape((3, 1))
        T_h = np.vstack((T, np.array([.0, .0, 1.0]).reshape((1,3))))
        tl = T_h @ tl_h
        br = T_h @ br_h
        self.top_left = tuple(tl[:2].squeeze().astype(int))
        self.bottom_right = tuple(br[:2].squeeze().astype(int))

    @classmethod
    def from_ultralytics_bbox(cls, ultralytics_bbox) -> 'BoundingBox':
        xyxy: Tuple[float] = tuple([int(i) for i in ultralytics_bbox.xyxy.cpu().numpy().squeeze()])
        return BoundingBox(top_left=xyxy[:2],bottom_right=xyxy[2:])

    def get_image_crop(self, im: np.ndarray, buffer: int = 0) -> np.ndarray:
        tl_y = self.top_left[1] - buffer
        tl_x = self.top_left[0] - buffer
        br_y = self.bottom_right[1] + buffer
        br_x = self.bottom_right[0] + buffer
        if tl_y < 0:
            tl_y = self.top_left[1]
        if tl_x < 0:
            tl_x = self.top_left[0]
        if br_y >= im.shape[0]:
            br_y = self.bottom_right[1]
        if br_x >= im.shape[1]:
            br_x = self.bottom_right[0]
        return im[tl_y:br_y,tl_x:br_x]

    def draw_on_image(self,
            im: np.ndarray,
            color: Tuple[int] = (0, 0, 0),  # Default green color
            thickness: int = 2,
            label: str = None,
            font_scale: float = 0.5,
            text_color: Tuple[int] = (255, 255, 255),  # White text
            text_thickness: int = 1
    ) -> np.ndarray:
        """
        Draw a bounding box on an image using OpenCV.

        Args:
            image (np.ndarray): Input image (BGR format)
            bbox (BoundingBox): Bounding box object with top_left and bottom_right coordinates
            color (Tuple[int, int, int]): Box color in BGR format
            thickness (int): Line thickness
            label (str, optional): Label to display above the box
            font_scale (float): Font scale for the label
            text_color (Tuple[int, int, int]): Text color in BGR format
            text_thickness (int): Text thickness

        Returns:
            np.ndarray: Image with drawn bounding box
        """
        # Make a copy to avoid modifying the original image
        img_with_box = im.copy()

        # Extract coordinates
        x1, y1 = self.top_left
        x2, y2 = self.bottom_right

        # Ensure coordinates are integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Draw the bounding box
        img_with_box = cv2.rectangle(
            img_with_box,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        # If a label is provided, draw it
        if label:
            # Calculate text size to create background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                img_with_box,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1  # Fill rectangle
            )

            # Draw text
            cv2.putText(
                img_with_box,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                text_thickness
            )

        return img_with_box

    def get_center(self) -> np.ndarray:
        return 1/2*(np.array(self.top_left)+np.array(self.bottom_right))

class HoldBBox(BoundingBox):
    id: int
    embedding: torch.Tensor
    global_uid: int = -1
    is_near_climber: bool = False
    is_climbed_route: bool = False
    is_labeled: bool = False
    UNABELED_ID:int = -1
    image: np.ndarray = None
    color_name: str = None
    bgr_color: Tuple[int, int, int] = None
    def __init__(self, top_left: tuple, bottom_right: tuple, id: int = None,
                 embedding: torch.Tensor = None, width_px: int = None, height_px: int = None,
                 is_near_climber: bool = False, is_labeled: bool = False):
        BoundingBox.__init__(self,
                             top_left=top_left,
                             bottom_right=bottom_right,
                             width_px=width_px,
                             height_px=height_px)
        self.id = id
        self.embedding = embedding
        self.is_near_climber = is_near_climber
        self.is_labeled = is_labeled

    def __copy__(self):
        return HoldBBox(top_left=self.top_left,
                        bottom_right=self.bottom_right,
                        id=self.id,
                        embedding=self.embedding,
                        width_px=self.width_px,
                        height_px=self.height_px,
                        is_near_climber=self.is_near_climber)

    @classmethod
    def next_id(cls) -> str:
        cls.global_uid += 1
        return cls.global_uid

    def draw_on_image(self,color: Tuple[int, int, int]=(0,0,0),text_color: Tuple[int]=(255,255,255),**kwargs):
        if kwargs.get('label') is None:
            kwargs['label'] = '' if self.id is None else f"{self.id}"
        return BoundingBox.draw_on_image(self, color=color, text_color=text_color, **kwargs)

    @classmethod
    def from_ultralytics_bbox(cls, ultralytics_bbox, id: int = None) -> 'HoldBBox':
        xyxy: Tuple[float] = tuple([int(i) for i in ultralytics_bbox.xyxy.cpu().numpy().squeeze()])
        return HoldBBox(top_left=xyxy[:2],bottom_right=xyxy[2:], id=id)

def get_holds_centers(holds: List[HoldBBox], return_homogeneous: bool = False) -> np.ndarray:
    if return_homogeneous:
        out: np.ndarray = np.ones((3, len(holds)))
        out[:2] = np.array([h.get_center() for h in holds]).T
        return out
    else:
        return np.array([h.get_center() for h in holds]).T

def draw_holds_on_image(im: np.ndarray, holds: List[HoldBBox] = None, color: Tuple[int] = (128, 128, 128),text_color: Tuple[int]=(255,255,255)):
    out = np.array(im)
    for h in holds:
        out = h.draw_on_image(im=out, color=h.bgr_color, text_color=text_color)
    return out

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def get_bounding_box(mask: np.ndarray, pad: int = 0) -> BoundingBox:
    """
    Returns the top-left and bottom-right coordinates of the bounding box
    containing the class in the given boolean mask.

    Args:
        mask (Tensor): A 2D tensor representing a boolean mask.

    Returns:
        BoundingBox: A dataclass containing the top-left and bottom-right coordinates
                      of the bounding box.
    """

    # Get the non-zero indices (i.e., the pixels that belong to the class)
    nonzero_indices = np.nonzero(mask)

    # If there are no non-zero indices, return None
    if len(nonzero_indices[0]) == 0:
        return None

    # Get the minimum and maximum indices along each axis
    top_left_x, top_left_y = nonzero_indices[1].min(), nonzero_indices[0].min()
    bottom_right_x, bottom_right_y = nonzero_indices[1].max(), nonzero_indices[0].max()

    return BoundingBox((max(0, top_left_x-pad), max(0, top_left_y-pad)),
                       (min(mask.shape[1], bottom_right_x+pad), min(mask.shape[0], bottom_right_y+pad)))

@dataclasses.dataclass
class ChalkGptConfig:
    save_to_disk: bool
    try_to_load_from_disk: bool
    matcher_threshold: float
    detector_threshold: float
    device: torch.device
    images_dir: str = None
    video_file: str = None
    superglue_model: str = 'outdoor'
    mask_for_matching: bool = False
    max_frames: int = None
    hold_to_traj_association_distance_px: float = 10
    static_video_shift_thr_px: float = 10.0
    save_video_path: str = None
    video_fps: int = 30

    def __post_init__(self):
        assert self.images_dir is not None or self.video_file is not None, "must provide video/frames path"


def get_opencv_kpt(pt, score) -> cv2.KeyPoint:
    kpt = cv2.KeyPoint()
    kpt.pt = tuple(pt)
    kpt.response = score
    return kpt

@dataclasses.dataclass
class LightGlueResult:
    img_0_kpts: List[cv2.KeyPoint]
    img_1_kpts: List[cv2.KeyPoint]
    matches: List[cv2.DMatch]


def lightglue_to_opencv_matches(input_dict: Dict, lightlue_output: Dict) -> LightGlueResult:
    matches: List[cv2.DMatch] = []
    query_kpts: List[cv2.KeyPoint] = []
    train_kpts: List[cv2.KeyPoint] = []
    matches_indices: np.ndarray = lightlue_output['matches'][0].detach().cpu().numpy()
    detected_kpts0: np.ndarray = input_dict['image0']['keypoints'][0].detach().cpu().numpy()
    detected_kpts1: np.ndarray = input_dict['image1']['keypoints'][0].detach().cpu().numpy()
    scores: np.ndarray = lightlue_output['scores'][0].detach().cpu().numpy()
    for idx, (m, score) in enumerate(zip(matches_indices, scores)):
        match: cv2.DMatch = cv2.DMatch()
        match.queryIdx = idx
        match.trainIdx = idx
        matches.append(match)
        query_kpts.append(get_opencv_kpt(detected_kpts0[m[0]], score))
        train_kpts.append(get_opencv_kpt(detected_kpts1[m[1]], score))
    return LightGlueResult(img_0_kpts=query_kpts,
                           img_1_kpts=train_kpts,
                           matches=matches)

def estimate_relative_pose(matching_results: KeypointMatchingResults) -> np.ndarray:
    INLIER_THRESHOLD: float = 0.1
    maxIters: int = 10000
    confidence = 0.9999
    refineIters: int = 10000
    kpts0: np.ndarray = np.array([np.array(matching_results.query_keypoints[m.queryIdx].pt) for m in matching_results.matches])
    kpts1: np.ndarray = np.array([np.array(matching_results.train_keypoints[m.trainIdx].pt) for m in matching_results.matches])
    mat, inliers = cv2.estimateAffinePartial2D(from_=kpts0, to=kpts1, method=cv2.RANSAC,
                                        ransacReprojThreshold=INLIER_THRESHOLD, maxIters=maxIters, confidence=confidence,
                                        refineIters=refineIters)
    s = np.linalg.norm(mat[0,:2])
    mat[:2,:2] /= s
    return mat

class TrackedObjectType(enum.Enum):
    Climber = 0
    Wall = 1
    Holds = 2

class FrameData:
    id: int
    im_path: str
    holds: List[HoldBBox]
    transform_to_world: np.ndarray
    pose: Results

    def __init__(self, id: int, im_path: str):
        self.id = id
        self.im_path = im_path

    def get_image(self) -> np.ndarray:
        return cv2.imread(self.im_path)
    #
    # def get_holds_near_climber(self):
    #     return [h for h in self.holds if h.is_near_climber]
    #
    # def draw_holds_on_image(self, holds: List[HoldBBox] = None) -> np.ndarray:
    #     if holds is None:
    #         holds = self.holds
    #     out: np.ndarray = self.get_image()
    #     for h in holds:
    #         out = h.draw_on_image(im=out)
    #     return out



class RandomLabelsGenerator:
    @staticmethod
    def generate():
        """Generate a random string of 6 English letters."""
        return ''.join(random.choices(string.ascii_letters, k=6))


def cluster_coordinates(coordinates, eps=0.5, min_samples=5):
    """
    Cluster 2D coordinates using DBSCAN algorithm.

    Parameters:
    coordinates: numpy array of shape (n_samples, 2)
    eps: float, maximum distance between two samples for them to be considered neighbors
    min_samples: int, minimum number of samples in a neighborhood for a point to be a core point

    Returns:
    labels: array of cluster labels (-1 represents noise points)
    n_clusters: number of clusters found
    """
    # Ensure input is numpy array
    coordinates = np.array(coordinates)

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    labels = clustering.labels_

    # Get number of clusters (excluding noise points)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, n_clusters

def is_bbox_occluded_by_mask(mask: Polygon, bbox: BoundingBox, max_iou: float = .8) -> bool:
    iou = mask.intersection(bbox.polygon).area / bbox.polygon.area
    return iou > max_iou

def distance_bbox_from_mask(mask: Polygon, bbox: BoundingBox) -> float:
    return bbox.polygon.distance(mask)

def mask_to_polygon(mask_image, simplify_tolerance=None):
    """
    Convert a binary mask image to a Shapely polygon.

    Parameters:
    -----------
    mask_image : numpy.ndarray
        Binary mask image where non-zero pixels represent the area to convert to polygon.
    simplify_tolerance : float, optional
        Tolerance parameter for simplifying the polygon. If None, no simplification is performed.
        Higher values result in more simplification.

    Returns:
    --------
    shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Shapely polygon(s) representing the mask.
    """
    # Ensure the mask is binary
    if mask_image.dtype != np.bool_:
        mask_image = mask_image.astype(bool)

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask_image.astype(np.uint8),
        cv2.RETR_EXTERNAL,  # Only retrieve the external contours
        cv2.CHAIN_APPROX_SIMPLE  # Compress horizontal, vertical, and diagonal segments
    )

    # Convert contours to polygons
    polygons = []
    for contour in contours:
        # Convert contour to a list of (x, y) tuples
        coords = [(point[0][0], point[0][1]) for point in contour]

        # Create a polygon (must have at least 3 points)
        if len(coords) >= 3:
            poly = Polygon(coords)

            # Simplify polygon if tolerance is provided
            if simplify_tolerance is not None and poly.is_valid:
                poly = poly.simplify(simplify_tolerance, preserve_topology=True)

            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)

    # Return the appropriate geometry
    if len(polygons) == 0:
        return None
    elif len(polygons) == 1:
        return polygons[0]
    else:
        return MultiPolygon(polygons)


def join_images(world_image, raw_image):
    """
    Join two RGB images side by side.

    Args:
        world_image (numpy.ndarray): Left image (taller than image_2)
        raw_image (numpy.ndarray): Right image

    Returns:
        numpy.ndarray: Combined image with image_2 centered vertically
    """
    # Get dimensions of the images
    h1, w1 = world_image.shape[:2]
    h2, w2 = raw_image.shape[:2]

    # Ensure image_1 is taller than image_2
    if h1 < h2:
        raise ValueError("image_1 must be taller than image_2")

    # Calculate padding needed for image_2
    padding_top = (h1 - h2) // 2
    padding_bottom = h1 - h2 - padding_top  # Account for odd difference

    # Create a black canvas for image_2 with padding
    padded_image_2 = np.zeros((h1, w2, 3), dtype=np.uint8)

    # Place image_2 in the center of the padded area
    padded_image_2[padding_top:padding_top + h2, :] = raw_image

    # Concatenate the images horizontally
    combined_image = np.hstack((world_image, padded_image_2))

    return combined_image

class ChalkGpt:
    frames_data: Dict[int, FrameData]

    config: ChalkGptConfig
    predictor: object
    pose_estimator: object
    yolo_holds: object
    # vector_db: FAISSIndex
    random_labels_generator: RandomLabelsGenerator
    static_video: bool
    CLIMBER_OBJECT_ID = 1
    WALL_OBJECT_ID = 2
    HOLDS_OBJECT_ID = 3
    transform_to_world: Dict[int, np.ndarray]
    H_ext: int
    W_ext: int
    holds_world: List[HoldBBox]

    video_writer: VideoWriterChalkGpt

    def   __init__(self, config: ChalkGptConfig):
        self.config = config

        # Load SAM2 from HuggingFace (auto-downloads and caches model)
        # Options: facebook/sam2-hiera-tiny, facebook/sam2-hiera-small,
        #          facebook/sam2-hiera-base-plus, facebook/sam2-hiera-large
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        self.predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-small")

        # Initialize Ultralytics YOLO Pose model (yolov8m-pose for balanced speed/accuracy)
        # Other options: yolov8n-pose.pt (fastest), yolov8s-pose.pt, yolov8l-pose.pt (most accurate)
        # Or YOLOv11: yolo11m-pose.pt
        self.pose_estimator = YOLO("yolov8m-pose.pt")

        # self.img2vec = Img2Vec(cuda=config.device is torch.device('cuda:0'), model='resnet-18')
        # self.vector_db = FAISSIndex(dimension=512, index_type='cosine')

        self.yolo_holds = YOLO("weights/holds/v1/weights/best.pt")
        # self.yolo_shoes = YOLO("weights/shoes/v0/weights/best.pt")
        superpoint_config = {'keypoint_threshold':self.config.detector_threshold}
        superglue_config = {'weights':self.config.superglue_model,
                            'match_threshold':self.config.matcher_threshold}
        self.detector = SuperPoint(config=superpoint_config).to(self.config.device).eval()
        self.matcher = SuperGlue(superglue_config).to(self.config.device).eval()

        self.random_labels_generator = RandomLabelsGenerator()
        self.holds_world = []
        if config.images_dir is None:
            output_dir: str = Path(config.video_file).stem
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                extract_video_frames(config.video_file, output_dir)
            config.images_dir = output_dir

        if config.save_video_path is not None:
            output_path = os.path.join(config.save_video_path, config.images_dir)
            os.makedirs(output_path, exist_ok=True)
            self.video_writer = VideoWriterChalkGpt(output_path=os.path.join(output_path,'chalk_gpt.mp4'))

    def main(self):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            process_frames: bool = True
            os.makedirs('cache', exist_ok=True)
            output_path: str = os.path.join('cache', self.config.images_dir + '.pkl')
            if self.config.try_to_load_from_disk and os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    frame_names = saved_data['frame_names']
                    video_segments = saved_data['video_segments']
                    self.frames_data = saved_data['frames_data']
                    self.relative_motion = saved_data['relative_motion']
                    self.transform_to_world = saved_data['transform_to_world']
                    process_frames = False

            if process_frames:
                frame_names = self.init_frames()
                video_segments = self.process_frames(frame_names=frame_names)
                self.estimate_relative_motion(video_segments=video_segments, frame_names=frame_names)
                save_data: Dict = {"frame_names":frame_names,
                                   "video_segments":video_segments,
                                   "frames_data": self.frames_data,
                                   "relative_motion": self.relative_motion,
                                   "transform_to_world": self.transform_to_world}
                if self.config.save_to_disk:
                    with open(output_path, 'wb') as f:
                        pickle.dump(save_data, f)

            self.define_world_frame()
            self.describe_video()
            self.label_all_holds()
            self.match_holds()
            self.get_holds_color()
            self.identify_route_color()
            self.route_color = 'yellow'
            self.visualize(video_segments=video_segments, frame_names=frame_names)
        if self.video_writer is not None:
            self.video_writer.release()

    def label_all_holds(self):
        T_init_world = self.get_world_frame_placement_transform(return_homogeneous=True)
        all_centers = None
        for frame_id, frame_data in self.frames_data.items():
            temp_holds = [h for h in frame_data.holds if h.is_near_climber]
            centers_frame = get_holds_centers(temp_holds, return_homogeneous=True)
            centers_world = self.transform_to_world[frame_id] @ T_init_world @ centers_frame
            if all_centers is None:
                all_centers = centers_world
            else:
                all_centers = np.hstack((all_centers, centers_world))
        labels, n_clusters = cluster_coordinates(all_centers.T, eps=10.0, min_samples=2)
        centers_final = []
        for lbl in set(labels):
            if lbl==-1:
                continue
            centers_final.append(np.mean(all_centers[:2, labels==lbl], axis=1))
        centers_final = sorted(centers_final, reverse=True, key=lambda x: x[1])
        for center in centers_final:
            self.holds_world.append(HoldBBox(top_left=center-np.array([10,10]), bottom_right=center+np.array([10,10]), id=f'h{HoldBBox.next_id()}'))






    def describe_video(self):
        cam_shift = [np.linalg.norm(t[:2, 2]) for t in self.transform_to_world.values()]
        static_video: bool = True
        for shift in cam_shift:
            if shift > self.config.static_video_shift_thr_px:
                static_video = False
        self.static_video = static_video

    def init_frames(self):
        frame_names = [
            p for p in os.listdir(self.config.images_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        if self.config.max_frames is not None:
            frame_names = frame_names[:self.config.max_frames]

        self.frames_data = {}
        for frame_id, frame_name in enumerate(frame_names):
            self.frames_data[frame_id] = FrameData(id=frame_id,
                                                   im_path=os.path.join(self.config.images_dir, frame_name))
        return frame_names

    def localize_holds_using_yolo(self, inference_state, frame_idx: int, frame_names: List[str]):
        resuts = self.yolo_holds(os.path.join(config.images_dir, frame_names[frame_idx]))[0]
        bboxes = resuts.boxes[resuts.boxes.conf>0.5]
        points = np.empty((len(bboxes), 2), dtype=np.float32)
        labels = self.HOLDS_OBJECT_ID * np.ones(len(bboxes), dtype=np.int32)
        for idx, bbox in enumerate(bboxes):
            xywh = bbox.xywh.cpu().numpy().squeeze()
            points[idx] = xywh[:2]
            # points = np.array([selected_points], dtype=np.float32).squeeze()
            # if len(points.shape) == 1:
            #     points = points.reshape((1,2))
        if len(labels.shape) == 1:
            labels = labels.reshape((1,-1))
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=self.HOLDS_OBJECT_ID,
            points=points,
            labels=labels,
        )

    def manually_select_object(self, inference_state, tracked_obj: TrackedObjectType, frame_idx: int, frame_names: List[str]):
        if tracked_obj is TrackedObjectType.Climber:
            label: str = 'select climber'
            object_id: int = self.CLIMBER_OBJECT_ID
        elif tracked_obj is TrackedObjectType.Wall:
            label: str = 'select wall'
            object_id: int = self.WALL_OBJECT_ID
        elif tracked_obj is TrackedObjectType.Holds:
            label: str = 'select hold'
            object_id: int = self.HOLDS_OBJECT_ID

        selected_points = select_pixel(image=cv2.imread(os.path.join(config.images_dir, frame_names[frame_idx])), label=label)
        """
        segment multipleo objects
        https://github.com/roboflow/rf-segment-anything-2/blob/main/notebooks/video_predictor_example.ipynb
        """
        points = np.array([selected_points], dtype=np.float32).squeeze()
        if len(points.shape) == 1:
            points = points.reshape((1,2))
        labels = object_id * np.ones(len(selected_points), dtype=np.int32)
        if len(labels.shape) == 1:
            labels = labels.reshape((1,-1))
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )

    def find_holds_all_frames(self, frame_names):
        for frame_id, frame_data in tqdm.tqdm(self.frames_data.items(), total=len(self.frames_data)):
            frame_data.holds = self.detect_holds(frame_id=frame_id,im_path=frame_data.im_path)

    def _convert_yolo_pose_output(self, results):
        """
        Convert Ultralytics YOLO Pose output to format compatible with find_holds_near_climber.

        YOLO Pose returns: Results object with:
        - keypoints.xy: (N, 17, 2) - x, y coordinates
        - keypoints.conf: (N, 17) - confidence scores

        Returns: SimpleNamespace with:
        - poses: list of (17, 3) arrays with [x, y, conf]
        """
        from types import SimpleNamespace

        if results is None or len(results) == 0 or results[0].keypoints is None:
            return SimpleNamespace(poses=[])

        result = results[0]
        keypoints = result.keypoints

        if keypoints.xy is None or len(keypoints.xy) == 0:
            return SimpleNamespace(poses=[])

        # Combine keypoints and confidence scores
        poses = []
        for i in range(len(keypoints.xy)):
            kpts_xy = keypoints.xy[i].cpu().numpy()  # (17, 2)
            kpts_conf = keypoints.conf[i].cpu().numpy()  # (17,)
            # Combine to (17, 3) format: [x, y, conf]
            pose = np.concatenate([kpts_xy, kpts_conf[:, np.newaxis]], axis=1)
            poses.append(pose)

        return SimpleNamespace(poses=poses)

    def find_holds_near_climber(self, image: np.ndarray, mask_polygon: Polygon, holds: List[HoldBBox], pose):
        """
        Find holds touched by climber's hands and feet using pose estimation keypoints.

        When pose is available: Sets is_climbed_route=True for holds near hands/feet
        When pose is None: Sets is_near_climber=True as fallback (not used for route)

        COCO keypoint indices:
        - 9: left_wrist, 10: right_wrist (hands)
        - 15: left_ankle, 16: right_ankle (feet)
        """
        if pose is None or len(pose.poses) == 0:
            # Fallback to polygon-based detection (used when pose not available)
            # Only sets is_near_climber, NOT is_climbed_route
            for hold in holds:
                if distance_bbox_from_mask(mask_polygon, hold) < self.config.hold_to_traj_association_distance_px:
                    hold.is_near_climber = True
            return

        # Get the first (most confident) pose
        keypoints = pose.poses[0]
        # Extract hand and foot contact points (COCO format)
        # https: // docs.ultralytics.com / tasks / pose /
        # COCO keypoint indices:
        # 9=left_wrist, 10=right_wrist, 15=left_ankle, 16=right_ankle
        contact_point_indices = [9, 10, 15, 16]#, 15, 16]
        confidence_threshold = 0.3  # Minimum confidence to consider a keypoint

        contact_points = []
        for idx in contact_point_indices:
            if idx < len(keypoints):
                x, y, conf = keypoints[idx]
                if conf > confidence_threshold:
                    contact_points.append(np.array([x, y]))

        if len(contact_points) == 0:
            # No confident keypoints found, fallback to polygon-based
            for hold in holds:
                if distance_bbox_from_mask(mask_polygon, hold) < self.config.hold_to_traj_association_distance_px:
                    hold.is_near_climber = True
            return

        # Check each hold against hand/foot contact points (pose-based detection)
        # This is the high-confidence route detection
        for contact_point in contact_points:
            for hold in holds:
                hold_center = hold.get_center()
                distance = np.linalg.norm(hold_center - contact_point)
                if distance < 3*self.config.hold_to_traj_association_distance_px:
                    hold.is_climbed_route = True  # High confidence: pose-based detection
                    hold.is_near_climber = True   # Also mark as near climber
                    break  # No need to check other holds

    def process_frames(self, frame_names):
        inference_state = self.predictor.init_state(video_path=self.config.images_dir, offload_video_to_cpu=True,
                                                    offload_state_to_cpu=True)
        self.manually_select_object(inference_state, tracked_obj=TrackedObjectType.Climber, frame_idx=0,
                                    frame_names=frame_names)
        self.find_holds_all_frames(frame_names)

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            if self.config.max_frames is not None and out_frame_idx >= self.config.max_frames:
                break
            video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
            for out_obj_id, mask in video_segments[out_frame_idx].items():
                mask = mask.squeeze()
                mask_polygon = mask_to_polygon(mask)

                # Only run pose estimation on sampled frames
                # Apply climber mask to image
                masked_image = np.tile(np.expand_dims(mask, 2), [1, 1, 3]) * self.frames_data[out_frame_idx].get_image()
                # Run YOLO Pose inference
                pose_results = self.pose_estimator(masked_image, conf=0.3, verbose=False)
                self.frames_data[out_frame_idx].pose = pose_results
                # Convert to compatible format
                pose = self._convert_yolo_pose_output(pose_results)
                if out_obj_id == self.CLIMBER_OBJECT_ID:
                    self.find_holds_near_climber(image=self.frames_data[out_frame_idx].get_image(),
                                                 mask_polygon=mask_polygon,
                                                 holds=self.frames_data[out_frame_idx].holds,
                                                 pose=pose)
        return video_segments

    def define_world_frame(self):
        delta_x_per_frame = np.array([t[0, 2] for t in self.transform_to_world.values()])
        delta_y_per_frame = np.array([t[1, 2] for t in self.transform_to_world.values()])
        max_delta_x = delta_x_per_frame[np.argmax(np.abs(delta_x_per_frame))]
        max_delta_y = delta_y_per_frame[np.argmax(np.abs(delta_y_per_frame))]
        self.H_ext = int(abs(max_delta_y))
        self.W_ext = int(abs(max_delta_x))


    def get_world_frame(self, frame_names) -> np.ndarray:
        """
        Define extended frame to contain camera motion
        """
        frame: np.ndarray = cv2.imread(os.path.join(self.config.images_dir, frame_names[self.get_world_frame_idx()]))
        H, W = frame.shape[:2]
        if len(frame.shape) == 3:
            world_frame: np.ndarray = np.zeros((H + 2*self.H_ext, W + 2*self.W_ext, 3), dtype=frame.dtype)
        else:
            world_frame: np.ndarray = np.zeros((H + 2*self.H_ext, W + 2*self.W_ext), dtype=frame.dtype)
        return world_frame

    def get_world_frame_placement_transform(self, return_homogeneous: bool = False) -> np.ndarray:
        out: np.ndarray = np.hstack((np.eye(2), np.zeros((2,1))))
        out[0, 2] = self.W_ext
        out[1, 2] = 2*self.H_ext
        if not return_homogeneous:
            return out
        else:
            return np.vstack((out, np.array([.0, .0, 1.0])))

    def detect_holds(self, frame_id:int, im_path: str):
        # im_bgr: np.ndarray = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
        yolo_res: Results = self.yolo_holds(im_path, conf=.3)[0]
        out: List[HoldBBox] = []
        # detected_holds = yolo_res.boxes
        for id, box in enumerate(yolo_res.boxes):
            box: HoldBBox = HoldBBox.from_ultralytics_bbox(ultralytics_bbox=box, id=f'{frame_id}.{id}')
            out.append(box)
        return out

    def visualize(self, video_segments, frame_names):
        world_frame: np.ndarray = self.get_world_frame(frame_names)
        for out_frame_idx in range(0, len(frame_names)):
            frame: np.ndarray = cv2.imread(os.path.join(self.config.images_dir, frame_names[out_frame_idx]))
            raw_frame = np.array(frame)
            blended_frame = np.array(raw_frame)
            for hold in self.frames_data[out_frame_idx].holds:
                if not hold.is_near_climber:
                    continue
                blended_frame = hold.draw_on_image(im=blended_frame, label=f'{hold.id}')

            mask_polygon = None
            for out_obj_id, mask in video_segments[out_frame_idx].items():
                mask = mask.squeeze()
                new_polygon = mask_to_polygon(mask)
                if mask_polygon is None and new_polygon is not None:
                      mask_polygon = new_polygon

                if out_obj_id == self.HOLDS_OBJECT_ID:
                    pass
                    # mask_3d = np.stack((0 * out_mask, out_mask, 0 * out_mask), axis=2).astype(np.uint8)
                    # blended_frame = cv2.blendLinear(blended_frame, 255 * mask_3d, 0.5 * np.ones_like(out_mask, dtype=np.float32),0.5 * np.ones_like(out_mask, dtype=np.float32))
                elif out_obj_id == self.CLIMBER_OBJECT_ID:
                    bbox = get_bounding_box(mask=mask, pad=50)
                    world_frame_with_climber = np.array(world_frame)
                    end_row = world_frame_with_climber.shape[0]
                    if self.H_ext != 0:
                        end_row = -1#self.H_ext
                    end_col = world_frame_with_climber.shape[1]
                    if self.W_ext != 0:
                        end_col = -self.W_ext
                    if self.static_video:
                        world_frame_with_climber[2 * self.H_ext:end_row, self.W_ext:end_col] = raw_frame
                    else:
                        world_frame_with_climber[2*self.H_ext-1:end_row,self.W_ext:end_col] = raw_frame
                    T = self.transform_to_world[out_frame_idx][:2]
                    bbox.apply_transform(T=self.get_world_frame_placement_transform())
                    bbox.apply_transform(T=T)

                    mask_polygon_world = transform_polygon(mask_polygon, self.get_world_frame_placement_transform())
                    mask_polygon_world = transform_polygon(mask_polygon_world, T)

                    if False:
                        world_frame_final = cv2.warpAffine(world_frame_with_climber, T, (world_frame_with_climber.shape[1], world_frame_with_climber.shape[0]), cv2.INTER_LINEAR)
                        climber_crop = world_frame_final[bbox.top_left[1]:bbox.bottom_right[1], bbox.top_left[0]:bbox.bottom_right[0]]
                        world_frame_final = draw_holds_on_image(world_frame_final, [h for h in self.holds_world if distance_bbox_from_mask(mask_polygon_world, h) < self.config.hold_to_traj_association_distance_px])
                        cv2.imshow("world frame", world_frame_final)

                    if mask_polygon_world is not None:
                        skeleton_world_frame = 0*world_frame_with_climber
                        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # Draw contours on an empty image (optional visualization)
                        contour_image = np.zeros_like(mask, dtype=np.uint8)
                        contour_image = cv2.drawContours(contour_image, contours, -1, 255, 1)
                        if hasattr(self.frames_data[out_frame_idx], "pose"):
                            contour_image = self.frames_data[out_frame_idx].pose[0].plot(img=cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR),conf=False,labels=False, boxes=False)
                        if len(contour_image.shape) == 2:
                            contour_rgb = np.tile(np.expand_dims(contour_image, 2), [1, 1, 3])
                        else:
                            contour_rgb = contour_image
                        if self.static_video:
                            skeleton_world_frame[2 * self.H_ext:end_row, self.W_ext:end_col] = contour_rgb
                        else:
                            skeleton_world_frame[2*self.H_ext-1:end_row,self.W_ext:end_col] = contour_rgb
                        skeleton_world_frame = cv2.warpAffine(skeleton_world_frame, T, (skeleton_world_frame.shape[1], skeleton_world_frame.shape[0]), cv2.INTER_LINEAR)
                        # mask_world = cv2.warpAffine(np.tile(np.expand_dims(mask,2),[1,1,3]).astype(np.uint8)*255, T, (world_frame_with_climber.shape[1], world_frame_with_climber.shape[0]), cv2.INTER_LINEAR)
                        # Filter to only show climbed route holds (pose-based detection) and not occluded by climber
                        route_holds = [h for h in self.holds_world
                                      if not is_bbox_occluded_by_mask(mask=mask_polygon_world, bbox=h)
                                       and h.color_name == self.route_color]
                        skeleton_world_frame = draw_holds_on_image(skeleton_world_frame, route_holds)
                        skeleton_world_frame = add_clock_to_image(skeleton_world_frame, out_frame_idx, self.config.video_fps)
                        joint_image = join_images(world_image=skeleton_world_frame,
                                                  raw_image=raw_frame)
                        self.video_writer.add_frame(joint_image)
                        cv2.imshow("joint", joint_image)

                    # cv2.imshow("original",raw_frame)

                    # Note: Pose visualization code removed during pose estimation migration
                    # Can be re-added using YOLO Pose results.plot() if needed
            if False:
                cv2.imshow("blended frame", blended_frame)

            while True:
                 key = cv2.waitKey(1) & 0xFF
                 if key == 32:  # 32 is the ASCII code for spacebar
                    break

    def get_lightlue_input(self, detector_output: Dict, image: torch.Tensor):
        """
        https://kornia.readthedocs.io/en/latest/feature.html
        keypoints: [B x M x 2]
        descriptors: [B x M x D]
        image: [B x C x H x W] or image_size: [B x 2]
        """
        im_size: torch.Tensor = torch.unsqueeze(torch.Tensor([image.shape[2], image.shape[3]]), dim=0)
        return {"keypoints": torch.unsqueeze(detector_output['keypoints'][0], dim=0),
                "descriptors": torch.unsqueeze(detector_output['descriptors'][0].T, dim=0),
                "image_size": im_size}

    def image_to_tensor(self, im: np.ndarray) -> torch.Tensor:
        # return BxCxHxW tensor
        assert len(im.shape) == 3, "image_to_tensor must be provided with BGR images"
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(im / 255.).float()[None, None].to(self.config.device)

    def get_masked_image(self, video_segments, frame_names: List[str], frame_idx: int) -> np.ndarray:
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            if out_obj_id != self.CLIMBER_OBJECT_ID:
                continue
            out_mask = out_mask.squeeze()
        im = cv2.imread(os.path.join(self.config.images_dir, frame_names[frame_idx]), cv2.IMREAD_GRAYSCALE)
        if self.config.mask_for_matching:
            return (~out_mask).astype(np.uint8) * im
        else:
            return im

    def get_world_frame_idx(self):
        return 0

    def estimate_relative_motion(self, video_segments, frame_names):
        self.relative_motion = {}
        curr_kpts: Optional[KeypointData] = None
        for i in tqdm.tqdm(range(len(frame_names)-1), total=len(frame_names)):
            tqdm.tqdm.write(f'Processing frame {i}')  # Display the current frame index
            if curr_kpts is None:
                curr_img = self.get_masked_image(video_segments=video_segments, frame_names=frame_names, frame_idx=i)
                curr_kpts: KeypointData = self.detector.detectAndCompute(curr_img, self.config.device)
            next_img = self.get_masked_image(video_segments=video_segments, frame_names=frame_names, frame_idx=i+1)
            next_kpts: KeypointData = self.detector.detectAndCompute(next_img, self.config.device)
            kp_matching_res: KeypointMatchingResults = self.matcher.match(next_kpts, curr_kpts)
            self.relative_motion[(i, i+1)] = np.vstack((estimate_relative_pose(kp_matching_res), np.array([.0, .0, 1.0])))
            curr_kpts = next_kpts

        self.transform_to_world = {}
        accumulated_transform = np.eye(3)
        self.transform_to_world[0] = accumulated_transform
        for i in range(1, len(frame_names)):
            accumulated_transform = accumulated_transform @ self.relative_motion[(i-1, i)]
            self.transform_to_world[i] = accumulated_transform

    def match_holds(self):
        holds_world_positions = np.array([h.get_center() for h in self.holds_world])
        T_init_world = self.get_world_frame_placement_transform(return_homogeneous=True)
        for curr_frame_id, frame_data in self.frames_data.items():
            frame_image = frame_data.get_image()
            holds_in_frame = get_holds_centers(frame_data.holds, return_homogeneous=True)
            holds_in_frame_world = self.transform_to_world[curr_frame_id] @ T_init_world @ holds_in_frame
            kdtree = cKDTree(holds_in_frame_world[:2].T)
            distances, indices = kdtree.query(holds_world_positions, k=1)
            for world_idx, (dist, hold_idx) in enumerate(zip(distances, indices)):
                frame_data.holds[hold_idx].id = self.holds_world[world_idx].id
                frame_data.holds[hold_idx].is_labeled = True
                if self.holds_world[world_idx].image is None and dist<self.config.hold_to_traj_association_distance_px:
                    self.holds_world[world_idx].image = frame_data.holds[hold_idx].get_image_crop(frame_image)

    def get_holds_color(self):
        """
        Classify hold colors using CLIP zero-shot classification.
        Assigns human-readable color names and BGR color tuples to each hold.
        Uses GPU and batch processing for maximum speed.
        """
        print("Loading CLIP model for hold color classification...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        clip_model = clip_model.to(self.config.device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print(f"✓ Model loaded on {self.config.device}")

        # Define common climbing hold colors
        color_labels = [
            "red climbing hold",
            "blue climbing hold",
            "green climbing hold",
            "yellow climbing hold",
            "orange climbing hold",
            "purple climbing hold",
            "black climbing hold",
            "white climbing hold",
            "pink climbing hold",
            "gray climbing hold"
        ]

        # Map color names to BGR tuples for visualization
        color_to_bgr = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "orange": (0, 165, 255),
            "purple": (255, 0, 255),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "pink": (203, 192, 255),
            "gray": (128, 128, 128)
        }

        # Collect all holds with images
        hold_images_pil = []
        valid_holds = []
        for hold in self.holds_world:
            if hold.image is not None:
                # Convert BGR (OpenCV) to RGB (PIL)
                hold_image_rgb = cv2.cvtColor(hold.image, cv2.COLOR_BGR2RGB)
                hold_image_pil = Image.fromarray(hold_image_rgb)
                hold_images_pil.append(hold_image_pil)
                valid_holds.append(hold)

        if len(hold_images_pil) == 0:
            print("No holds with images to classify.")
            return

        print(f"Classifying {len(hold_images_pil)} holds in batch...")

        # Batch process all holds at once
        inputs = clip_processor(
            text=color_labels,
            images=hold_images_pil,  # Process all images together
            return_tensors="pt",
            padding=True
        )

        # Move inputs to GPU
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Run inference on all holds at once
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # Shape: (num_holds, num_colors)
            color_indices = logits_per_image.argmax(dim=1)  # Shape: (num_holds,)

        # Assign predicted colors to holds
        for hold, color_idx in zip(valid_holds, color_indices):
            predicted_color = color_labels[color_idx.item()].replace(" climbing hold", "")
            hold.color_name = predicted_color
            hold.bgr_color = color_to_bgr[predicted_color]

        print(f"✓ Hold color classification complete! Classified {len(valid_holds)} holds.")

    def identify_route_color(self):
        """
        Identify the colors of holds touched by the climber using pose-based detection.
        These colors represent the route being climbed.
        Marks world holds as is_climbed_route=True if they were touched.
        """
        touched_hold_ids = set()

        # Collect IDs of all holds marked as climbed route (pose-based detection)
        for frame_id, frame_data in self.frames_data.items():
            for hold in frame_data.holds:
                if hold.is_climbed_route and hold.id is not None:
                    touched_hold_ids.add(hold.id)

        # Mark corresponding world holds as part of climbed route and collect colors
        route_colors = {}
        for hold in self.holds_world:
            if hold.id in touched_hold_ids:
                hold.is_climbed_route = True  # Mark world hold as part of route
                if hold.color_name is not None:
                    if route_colors.get(hold.color_name) is None:
                        route_colors[hold.color_name] = 0
                    route_colors[hold.color_name] += 1

        colors = list(route_colors.keys())
        nb_appearances = np.array(list(route_colors.values()))
        self.route_color = colors[np.argmax(nb_appearances)]
        print(f"✓ Identified route color (pose-based): {self.route_color}")


# TODO - holds occlusion should be handled using climber polygon

if __name__ == "__main__":
    config: ChalkGptConfig = ChalkGptConfig(save_to_disk=True,
                                            try_to_load_from_disk=False,
                                            images_dir='anna',
                                            # video_file='romi.mp4',
                                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                            matcher_threshold=.9,
                                            detector_threshold=.1,
                                            superglue_model='outdoor',
                                            mask_for_matching=False,
                                            max_frames=None,
                                            save_video_path='output',
                                            video_fps=30)
    chalk_gpt: ChalkGpt = ChalkGpt(config)
    chalk_gpt.main()