import dataclasses
import enum
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import super_gradients
import torch
import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from super_gradients.training.utils.predict import ImagePoseEstimationPrediction

from lightglue import LightGlue
from mouse_click import select_pixel
# from mouse_click import get_mouse_click_coords
from sam2.build_sam import build_sam2_video_predictor
from dataclasses import dataclass

from src.base import KeypointData, KeypointMatchingResults
from src.superglue import SuperGlue
from src.superpoint import SuperPoint
from video_extractor import extract_video_frames


@dataclass
class BoundingBox:
    top_left: tuple
    bottom_right: tuple

    def apply_transform(self, T: np.ndarray):
        tl_h: np.ndarray = np.array([self.top_left[0], self.top_left[1], 1.0]).reshape((3, 1))
        br_h: np.ndarray = np.array([self.bottom_right[0], self.bottom_right[1], 1.0]).reshape((3, 1))
        T_h = np.vstack((T, np.array([.0, .0, 1.0]).reshape((1,3))))
        tl = T_h @ tl_h
        br = T_h @ br_h
        self.top_left = tuple(tl[:2].squeeze().astype(int))
        self.bottom_right = tuple(br[:2].squeeze().astype(int))

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

class ChalkGpt:
    config: ChalkGptConfig
    predictor: object
    pose_estimator: object

    CLIMBER_OBJECT_ID = 1
    WALL_OBJECT_ID = 2
    HOLDS_OBJECT_ID = 3
    transform_from_first_frame: Dict[str, np.ndarray]
    H_ext: int
    W_ext: int
    def   __init__(self, config: ChalkGptConfig):
        self.config = config

        sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

        self.yolo_nas = super_gradients.training.models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
        self.yolo_nas.eval()

        superpoint_config = {'keypoint_threshold':self.config.detector_threshold}
        superglue_config = {'weights':self.config.superglue_model,
                            'match_threshold':self.config.matcher_threshold}
        self.detector = SuperPoint(config=superpoint_config).to(self.config.device).eval()
        self.matcher = SuperGlue(superglue_config).to(self.config.device).eval()

        if config.images_dir is None:
            output_dir: str = Path(config.video_file).stem
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                extract_video_frames(config.video_file, output_dir)
            config.images_dir = output_dir

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
                    process_frames = False

            if process_frames:
                frame_names = self.init_frames()
                video_segments = self.process_frames(frame_names=frame_names)
                save_data: Dict = {"frame_names":frame_names,
                             "video_segments":video_segments}
                if self.config.save_to_disk:
                    with open(output_path, 'wb') as f:
                        pickle.dump(save_data, f)

            self.transform_from_first_frame = {}
            self.estimate_relative_motion(video_segments=video_segments, frame_names=frame_names)
            self.visualize(video_segments=video_segments, frame_names=frame_names)

    def init_frames(self):
        frame_names = [
            p for p in os.listdir(self.config.images_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        if self.config.max_frames is not None:
            frame_names = frame_names[:self.config.max_frames]
        return frame_names

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

    def process_frames(self, frame_names):
        inference_state = self.predictor.init_state(video_path=self.config.images_dir, offload_video_to_cpu=True,
                                                    offload_state_to_cpu=True)
        self.manually_select_object(inference_state, tracked_obj=TrackedObjectType.Climber, frame_idx=0,
                                    frame_names=frame_names)
        self.manually_select_object(inference_state, tracked_obj=TrackedObjectType.Holds, frame_idx=0,
                                    frame_names=frame_names)
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            if self.config.max_frames is not None and out_frame_idx > self.config.max_frames:
                break
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        return video_segments

    def get_world_frame(self, frame_names) -> np.ndarray:
        """
        Define extended frame to contain camera motion
        """
        delta_x_per_frame = np.array([t[0, 2] for t in self.transform_from_first_frame.values()])
        delta_y_per_frame = np.array([t[1, 2] for t in self.transform_from_first_frame.values()])
        max_delta_x = delta_x_per_frame[np.argmax(np.abs(delta_x_per_frame))]
        max_delta_y = delta_y_per_frame[np.argmax(np.abs(delta_y_per_frame))]
        frame: np.ndarray = cv2.imread(os.path.join(self.config.images_dir, frame_names[0]))
        H, W = frame.shape[:2]
        self.H_ext = int(abs(max_delta_y))
        self.W_ext = int(abs(max_delta_x))
        if len(frame.shape) == 3:
            world_frame: np.ndarray = np.zeros((H + 2*self.H_ext, W + 2*self.W_ext, 3), dtype=frame.dtype)
        else:
            world_frame: np.ndarray = np.zeros((H + 2*self.H_ext, W + 2*self.W_ext), dtype=frame.dtype)
        return world_frame

    def get_world_frame_placement_transform(self) -> np.ndarray:
        out: np.ndarray = np.hstack((np.eye(2), np.zeros((2,1))))
        out[0, 2] = self.W_ext
        out[1, 2] = self.H_ext
        return out

    def visualize(self, video_segments, frame_names):
        world_frame: np.ndarray = self.get_world_frame(frame_names)
        for out_frame_idx in range(0, len(frame_names)):
            frame: np.ndarray = cv2.imread(os.path.join(self.config.images_dir, frame_names[out_frame_idx]))
            raw_frame = np.array(frame)
            blended_frame = np.array(frame)
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                out_mask = out_mask.squeeze()
                if out_obj_id == self.HOLDS_OBJECT_ID:
                    mask_3d = np.stack((0 * out_mask, out_mask, 0 * out_mask), axis=2).astype(np.uint8)
                    blended_frame = cv2.blendLinear(blended_frame, 255 * mask_3d, 0.5 * np.ones_like(out_mask, dtype=np.float32),0.5 * np.ones_like(out_mask, dtype=np.float32))
                elif out_obj_id == self.CLIMBER_OBJECT_ID:
                    bbox = get_bounding_box(mask=out_mask, pad=50)
                    world_frame_with_climber = np.array(world_frame)
                    end_row = world_frame_with_climber.shape[0]
                    if self.H_ext != 0:
                        end_row = -self.H_ext
                    end_col = world_frame_with_climber.shape[1]
                    if self.W_ext != 0:
                        end_col = -self.W_ext
                    world_frame_with_climber[self.H_ext:end_row,self.W_ext:end_col] = raw_frame
                    T = self.transform_from_first_frame[frame_names[out_frame_idx]]
                    bbox.apply_transform(T=self.get_world_frame_placement_transform())
                    bbox.apply_transform(T=T)
                    world_frame_final = cv2.warpAffine(world_frame_with_climber, T, (world_frame_with_climber.shape[1], world_frame_with_climber.shape[0]), cv2.INTER_LINEAR)
                    climber_crop = world_frame_final[bbox.top_left[1]:bbox.bottom_right[1], bbox.top_left[0]:bbox.bottom_right[0]]
                    cv2.imshow("climber crop", world_frame_final)

                    pose: ImagePoseEstimationPrediction = self.yolo_nas.predict(climber_crop, conf=0.3, fuse_model=False)
                    pose_draw = pose.draw()
                    cv2.imshow("climber pose", pose_draw)
                    mask_3d = np.stack((0 * out_mask, 0 * out_mask, out_mask), axis=2).astype(np.uint8)
                    blended_frame = cv2.blendLinear(blended_frame, 255 * mask_3d, 0.5 * np.ones_like(out_mask, dtype=np.float32),
                                            0.5 * np.ones_like(out_mask, dtype=np.float32))

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

    def estimate_relative_motion(self, video_segments, frame_names):
        ref_img = self.get_masked_image(video_segments=video_segments, frame_names=frame_names, frame_idx=0)
        ref_img_kpts: KeypointData = self.detector.detectAndCompute(ref_img, self.config.device)
        for i in tqdm.tqdm(range(0, len(frame_names)-1), total=len(frame_names)):
            tqdm.tqdm.write(f'Processing frame {i}')  # Display the current frame index
            img_1 = self.get_masked_image(video_segments=video_segments, frame_names=frame_names, frame_idx=i)
            img_1_kp_data: KeypointData = self.detector.detectAndCompute(img_1, self.config.device)
            kp_matching_res: KeypointMatchingResults = self.matcher.match(img_1_kp_data, ref_img_kpts)
            self.transform_from_first_frame[frame_names[i]] = estimate_relative_pose(kp_matching_res)
            # cv2.imshow("match",cv2.drawMatches(img_0, matches.img_0_kpts, img_1, matches.img_1_kpts, matches.matches, None))
            # cv2.waitKey(100)

if __name__ == "__main__":
    config: ChalkGptConfig = ChalkGptConfig(save_to_disk=True,
                                            try_to_load_from_disk=True,
                                            images_dir='downloaded_frames_tag',
                                            # video_file='romi.mp4',
                                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                            matcher_threshold=.9,
                                            detector_threshold=.1,
                                            superglue_model='outdoor',
                                            mask_for_matching=False,
                                            max_frames=100)
    chalk_gpt: ChalkGpt = ChalkGpt(config)
    chalk_gpt.main()
