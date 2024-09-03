import glob
import os.path

import cv2
import super_gradients
import torch
from super_gradients.training.utils.predict import ImagePoseEstimationPrediction

if __name__ == "__main__":

    yolo_nas = super_gradients.training.models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
    yolo_nas.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    #     for f in yolo_nas.predict(r"/home/gabi/GitHub/Experiments/segment-anything-2/temp_video.webm", fuse_model=False).draw():
    #         cv2.imshow("frame",cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    #         cv2.waitKey(30)
        images_folder: str = r"/home/gabi/GitHub/Experiments/segment-anything-2/downloaded_frames"
        frame_paths = glob.glob(os.path.join(images_folder, '*.jpg'))
        frame_paths.sort()
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            model_predictions: ImagePoseEstimationPrediction = yolo_nas.predict(frame, conf=0.7, fuse_model=False)
            pose_draw = model_predictions.draw()
            pose_draw_rgb = cv2.cvtColor(pose_draw, cv2.COLOR_RGB2BGR)
            # pose_draw_rgb_resize = cv2.resize(pose_draw_rgb, [pose_draw_rgb.shape[1]*1,pose_draw_rgb.shape[0]*1])
            cv2.imshow(f"frame",pose_draw_rgb)
            cv2.setWindowTitle(f"frame",f"frame {frame_idx}")
            cv2.waitKey(100)
            # bboxes = prediction.bboxes_xyxy  # [Num Instances, 4] List of predicted bounding boxes for each object
            # poses = prediction.poses  # [Num Instances, Num Joints, 3] list of predicted joints for each detected object (x,y, confidence)
            # scores = prediction.scores  # [Num Instances] - Confidence value for each predicted instance

