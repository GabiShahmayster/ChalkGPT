import dataclasses
import enum
import os
import pickle
import time
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
from sam2.build_sam import build_sam2_video_predictor
from dataclasses import dataclass

from superpoint import SuperPoint


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
    video_dir: str
    lightglue_match_threshold: float
    device: torch.device


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

def estimate_relative_pose(matching_results: LightGlueResult) -> np.ndarray:
    INLIER_THRESHOLD: float = 1.0
    maxIters: int = 10000
    confidence = 0.9999
    refineIters: int = 10000
    kpts0: np.ndarray = np.array([np.array(k.pt) for k in matching_results.img_0_kpts])
    kpts1: np.ndarray = np.array([np.array(k.pt) for k in matching_results.img_1_kpts])
    mat, inliers = cv2.estimateAffinePartial2D(from_=kpts1, to=kpts0, method=cv2.RANSAC,
                                        ransacReprojThreshold=INLIER_THRESHOLD, maxIters=maxIters, confidence=confidence,
                                        refineIters=refineIters)
    s = np.linalg.norm(mat[0,:2])
    mat[:2,:2] /= s
    return mat

class ChalkGpt:
    config: ChalkGptConfig
    predictor: object
    pose_estimator: object

    CLIMBER_OBJECT_ID = 1
    WALL_OBJECT_ID = 2
    transform_from_first_frame: Dict[str, np.ndarray]
    H_ext: int
    W_ext: int
    def __init__(self, config: ChalkGptConfig):
        self.config = config

        sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

        self.yolo_nas = super_gradients.training.models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
        self.yolo_nas.eval()

        superpoint_config = {}
        lightglue_config = {"filter_threshold": self.config.lightglue_match_threshold}
        self.detector = SuperPoint(config=superpoint_config).to(self.config.device)
        self.matcher = LightGlue(features='superpoint', **lightglue_config).to(self.config.device)

    def main(self):
        process_frames: bool = True
        os.makedirs('cache', exist_ok=True)
        output_path: str = os.path.join('cache',self.config.video_dir+'.pkl')
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
        self.estimate_relative_motion(frame_names=frame_names)
        self.visualize(video_segments=video_segments, frame_names=frame_names)

    def init_frames(self):
        frame_names = [
            p for p in os.listdir(self.config.video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        return frame_names

    def process_frames(self, frame_names):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.predictor.init_state(video_path=self.config.video_dir, offload_video_to_cpu=True,
                                                   offload_state_to_cpu=True)

            ann_frame_idx = 0
            ann_obj_id = self.CLIMBER_OBJECT_ID

            points = np.array([[495, 320]], dtype=np.float32)
            labels = np.array([self.CLIMBER_OBJECT_ID], np.int32)
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
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
        frame: np.ndarray = cv2.imread(os.path.join(self.config.video_dir, frame_names[0]))
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
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                if out_obj_id != self.CLIMBER_OBJECT_ID:
                    continue
                out_mask = out_mask.squeeze()
                bbox = get_bounding_box(mask=out_mask, pad=50)
                frame: np.ndarray = cv2.imread(os.path.join(self.config.video_dir, frame_names[out_frame_idx]))
                world_frame_with_climber = np.array(world_frame)
                world_frame_with_climber[self.H_ext:-self.H_ext,self.W_ext:-self.W_ext] = frame
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
                blend = cv2.blendLinear(frame, 255 * mask_3d, 0.5 * np.ones_like(out_mask, dtype=np.float32),
                                        0.5 * np.ones_like(out_mask, dtype=np.float32))

                cv2.imshow("blended frame", blend)

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

    def estimate_relative_motion(self, frame_names):
        for i in tqdm.tqdm(range(0, len(frame_names)-1), total=len(frame_names)):
            tqdm.tqdm.write(f'Processing frame {i}')  # Display the current frame index
            img_0 = cv2.imread(os.path.join(self.config.video_dir,frame_names[0]))
            img_1 = cv2.imread(os.path.join(self.config.video_dir,frame_names[i+1]))
            img_0_tensor: torch.Tensor = self.image_to_tensor(img_0)
            img_1_tensor: torch.Tensor = self.image_to_tensor(img_1)
            img0_det = self.detector({"image": img_0_tensor})
            img1_det = self.detector({"image": img_1_tensor})
            lightglue_input_dict = {"image0": self.get_lightlue_input(img0_det, img_0_tensor),
                                    "image1": self.get_lightlue_input(img1_det, img_1_tensor)}
            lightlue_output = self.matcher(lightglue_input_dict)
            matches = lightglue_to_opencv_matches(input_dict=lightglue_input_dict, lightlue_output=lightlue_output)
            self.transform_from_first_frame[frame_names[i]] = estimate_relative_pose(matches)
            # cv2.imshow("match",cv2.drawMatches(img_0, matches.img_0_kpts, img_1, matches.img_1_kpts, matches.matches, None))
            # cv2.waitKey(100)

if __name__ == "__main__":
    config: ChalkGptConfig = ChalkGptConfig(save_to_disk=True,
                                            try_to_load_from_disk=True,
                                            video_dir='downloaded_frames_tag',
                                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                            lightglue_match_threshold=.995)
    chalk_gpt: ChalkGpt = ChalkGpt(config)
    chalk_gpt.main()
