import glob
import os.path

import cv2
import numpy as np
import super_gradients
import torch
from super_gradients.training.utils.predict import ImagePoseEstimationPrediction

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

if __name__ == "__main__":
    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))
    SEGMENT_CLIMBER: bool = True
    REFINE_POSE: bool = False
    CROP: bool = False
    yolo_nas = super_gradients.training.models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
    yolo_nas.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    #     for f in yolo_nas.predict(r"/home/gabi/GitHub/Experiments/segment-anything-2/temp_video.webm", fuse_model=False).draw():
    #         cv2.imshow("frame",cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    #         cv2.waitKey(30)
        images_folder: str = r"/home/gabi/GitHub/Experiments/segment-anything-2/downloaded_frames_new"
        frame_paths = glob.glob(os.path.join(images_folder, '*.jpg'))
        frame_paths.sort()
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            pose: ImagePoseEstimationPrediction = yolo_nas.predict(frame, conf=0.3, fuse_model=False)

            if REFINE_POSE or SEGMENT_CLIMBER:
                HEAD_JOINT_INDEX: int = 4
                EDGE_PADDING: int = 20
                if len(pose.prediction) == 0:
                    continue
                elif len(pose.prediction) > 1:
                    pose_idx = np.argmax(pose.prediction.scores)
                else:
                    pose_idx = 0

                bb = pose.prediction.bboxes_xyxy[pose_idx].squeeze()
                frame_for_seg = frame

                if CROP:
                    frame_for_seg = frame[int(bb[1]-EDGE_PADDING):int(bb[3]+EDGE_PADDING),int(bb[0]-EDGE_PADDING):int(bb[2]+EDGE_PADDING),:]
                points = pose.prediction.poses[0, :, :2][HEAD_JOINT_INDEX].reshape((1,2))
                labels = np.array([1])

                if REFINE_POSE:
                    predictor.set_image(image=frame_for_seg)
                    inference_state = predictor.predict(point_coords=points, point_labels=labels)
                    mask = np.max(inference_state[0].transpose([1, 2, 0]), axis=2)
                    refined_crop = frame_for_seg * np.tile(np.expand_dims(mask.astype(bool),2),[1,1,3])
                    pose: ImagePoseEstimationPrediction = yolo_nas.predict(refined_crop, conf=0.3, fuse_model=False)

            pose_draw = pose.draw()
            # pose_draw_rgb = cv2.cvtColor(pose_draw, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"frame",pose_draw)
            cv2.setWindowTitle(f"frame",f"frame {frame_idx}")
            cv2.waitKey(1)

