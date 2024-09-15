import dataclasses
import os
import pickle
import time
from typing import Dict, List

import cv2
import numpy as np
import super_gradients
import torch
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
    drone_kpts: List[cv2.KeyPoint]
    sat_kpts: List[cv2.KeyPoint]
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
    return LightGlueResult(drone_kpts=query_kpts,
                           sat_kpts=train_kpts,
                           matches=matches)

class ChalkGpt:
    config: ChalkGptConfig
    predictor: object
    pose_estimator: object

    CLIMBER_OBJECT_ID = 1
    WALL_OBJECT_ID = 2

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

    def visualize(self, video_segments, frame_names):
        for out_frame_idx in range(0, len(frame_names)):
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                if out_obj_id != self.CLIMBER_OBJECT_ID:
                    continue
                out_mask = out_mask.squeeze()
                bbox = get_bounding_box(mask=out_mask, pad=50)
                frame: np.ndarray = cv2.imread(os.path.join(self.config.video_dir, frame_names[out_frame_idx]))
                climber_crop = frame[bbox.top_left[1]:bbox.bottom_right[1], bbox.top_left[0]:bbox.bottom_right[0]]
                cv2.imshow("climber crop", climber_crop)

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
        for i in range(0, len(frame_names)-1):
            img_0 = self.image_to_tensor(cv2.imread(os.path.join(self.config.video_dir,frame_names[i])))
            img_1 = self.image_to_tensor(cv2.imread(os.path.join(self.config.video_dir,frame_names[i+1])))

            img0_det = self.detector({"image": img_0})
            img1_det = self.detector({"image": img_1})
            lightglue_input_dict = {"image0": self.get_lightlue_input(img0_det, img_0),
                                    "image1": self.get_lightlue_input(img1_det, img_1)}
            res = self.matcher(lightglue_input_dict)
            a=3

if __name__ == "__main__":
    config: ChalkGptConfig = ChalkGptConfig(save_to_disk=True,
                                            try_to_load_from_disk=True,
                                            video_dir='downloaded_frames_tag',
                                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                            lightglue_match_threshold=.1)
    chalk_gpt: ChalkGpt = ChalkGpt(config)
    chalk_gpt.main()
