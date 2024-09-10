import dataclasses
import glob
import os.path
import cv2
import numpy as np
import super_gradients
from attr import dataclass
from numpy.core.numeric import ones_like
from super_gradients.training.utils.predict import ImagePoseEstimationPrediction

from yt_dl import download_frames
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

@dataclasses.dataclass
class Config:
    segment_climber: bool = True
    refine_pose: bool = False
    crop: bool = False


class ChalkGpt:
    def __init__(self, config: Config, sam2_checkpoint, model_cfg):
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))
        self.config = config
        self.yolo_nas = super_gradients.training.models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
        self.yolo_nas.eval()

    def process_frames(self, images_folder):
        frame_paths = glob.glob(os.path.join(images_folder, '*.jpg'))
        frame_paths.sort()
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            pose: ImagePoseEstimationPrediction = self.yolo_nas.predict(frame, conf=0.3, fuse_model=False)

            if self.config.refine_pose or self.config.segment_climber:
                HEAD_JOINT_INDEX: int = 4
                EDGE_PADDING: int = 20
                if len(pose.prediction) == 0:
                    pose_idx = None
                elif len(pose.prediction) > 1:
                    pose_idx = np.argmax(pose.prediction.scores)
                else:
                    pose_idx = 0

                if pose_idx is None:
                    cv2.imshow(f"frame", frame)
                    cv2.setWindowTitle(f"frame", f"frame {frame_idx}")
                    cv2.waitKey(1)
                    continue

                bb = pose.prediction.bboxes_xyxy[pose_idx].squeeze()
                frame_for_seg = frame

                if self.config.segment_climber:
                    frame_for_seg = frame[int(bb[1]-EDGE_PADDING):int(bb[3]+EDGE_PADDING),int(bb[0]-EDGE_PADDING):int(bb[2]+EDGE_PADDING),:]
                points = pose.prediction.poses[0, :, :2][HEAD_JOINT_INDEX].reshape((1,2))
                labels = np.array([1])

                if self.config.segment_climber:
                    segmented_frame = np.ones_like(frame)
                    self.predictor.set_image(image=frame_for_seg)
                    inference_state = self.predictor.predict(point_coords=points, point_labels=labels)
                    mask = np.max(inference_state[0].transpose([1, 2, 0]), axis=2)
                    mask_3d = np.stack((0*mask, 0*mask, mask), axis=2).astype(np.uint8)
                    blend = cv2.blendLinear(frame_for_seg, 255*mask_3d, 0.5*np.ones_like(mask, dtype=np.float32), 0.5*np.ones_like(mask, dtype=np.float32))
                    frame[int(bb[1] - EDGE_PADDING):int(bb[3] + EDGE_PADDING), int(bb[0] - EDGE_PADDING):int(bb[2] + EDGE_PADDING), :] = blend
                    pose.image = frame
                    # refined_crop = frame_for_seg * np.tile(np.expand_dims(mask.astype(bool),2),[1,1,3])
                    # frame_for_seg *= np.tile(np.expand_dims(mask.astype(bool),2),[1,1,3])
                    # a=3
                    # pose: ImagePoseEstimationPrediction = self.yolo_nas.predict(refined_crop, conf=0.3, fuse_model=False)

            pose_draw = pose.draw()
            # pose_draw_rgb = cv2.cvtColor(pose_draw, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"frame",pose_draw)
            cv2.setWindowTitle(f"frame",f"frame {frame_idx}")
            cv2.waitKey(1)

    def download_video(self, url, start_time, num_frames, output_dir, resize_factor=5.0, force_download: bool = False, display_frames: bool = True):
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=720][abr<250]+bestaudio/best[height<=720]',
            'outtmpl': 'temp_video.%(ext)s'
        }
        """
        youtube-dl --get-title -f 'bestvideo[ext=mp4][height<=640][abr<250]+bestaudio/best[height<=640]' https://www.youtube.com/watch?v=VIDEO_ID --get-title 00:00:10-00:00:20
        """
        # Download the video
        if not os.path.exists('temp_video.webm') or force_download:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        # Open the video file
        video = cv2.VideoCapture('temp_video.webm')

        # Set the starting point
        fps = video.get(cv2.CAP_PROP_FPS)
        video.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract frames and display
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(num_frames):
            ret, frame = video.read()
            if ret:
                # Resize frame
                new_width = int(frame.shape[1] / resize_factor)
                new_height = int(frame.shape[0] / resize_factor)
                resized_frame = cv2.resize(frame, (new_width, new_height))

                # Save frame
                cv2.imwrite(os.path.join(output_dir, f'{i:03d}.jpg'), resized_frame)

                # Display frame
                cv2.imshow('Video Frame', resized_frame)

                # Wait for 30 ms and check if user wants to quit
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            # else:
            #     print(f"Could only extract {i} frames.")
            #     break

        # Clean up
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    config: Config = Config(segment_climber=True,
                            refine_pose=False,
                            crop=False)
    chalk_gpt = ChalkGpt(config=config, sam2_checkpoint=sam2_checkpoint, model_cfg=model_cfg)
    
    # url = 'https://www.youtube.com/watch?v=b2v4brHpdxY&t=130s'
    # start_time = 2*60+12  # 22:27 in seconds
    # num_frames = 250
    # output_dir = 'downloaded_frames_new'
    #
    # download_frames(url, start_time, num_frames, output_dir, force_download=True, resize_factor=1)
    
    images_folder: str = r"/home/gabi/GitHub/Experiments/segment-anything-2/downloaded_frames_new"
    chalk_gpt.process_frames(images_folder)