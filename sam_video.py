import os
import time

import cv2
import numpy as np
import super_gradients
import torch
from matplotlib import pyplot as plt
from PIL import Image
from super_gradients.training.utils.predict import ImagePoseEstimationPrediction

from sam2.build_sam import build_sam2_video_predictor
from dataclasses import dataclass

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

class ChalkGpt:
    video_dir: str
    predictor: object
    pose_estimator: object

    CLIMBER_OBJECT_ID = 1

    def __init__(self, video_dir: str):
        self.video_dir = video_dir

        sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

        self.yolo_nas = super_gradients.training.models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
        self.yolo_nas.eval()


    def run(self):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.predictor.init_state(video_path=self.video_dir, offload_video_to_cpu=True,
                                                   offload_state_to_cpu=True)

            frame_names = [
                p for p in os.listdir(self.video_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

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

            for out_frame_idx in range(0, len(frame_names)):
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    if out_obj_id != self.CLIMBER_OBJECT_ID:
                        continue
                    out_mask = out_mask.squeeze()
                    bbox = get_bounding_box(mask=out_mask, pad=50)
                    frame: np.ndarray = cv2.imread(os.path.join(self.video_dir, frame_names[out_frame_idx]))
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

if __name__ == "__main__":
    chalk_gpt: ChalkGpt = ChalkGpt(video_dir='downloaded_frames_tag')
    chalk_gpt.run()

