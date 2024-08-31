import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor
from yolo import YOLODetector

matplotlib.use('TkAgg')
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

class VideoWriterAux:
    def __init__(self, filename, fps, frame_size):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
    
    def write_frame(self, frame):
        self.writer.write(frame)
    
    def release(self):
        self.writer.release()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    neg_points = coords[labels==0]
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def refine_mask_with_coordinates(coordinates, ann_frame_idx, ann_obj_id, show_result=True):
    points = np.array(coordinates, dtype=np.float32)
    labels = np.ones(len(coordinates), dtype=np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    if show_result:
        plt.figure(figsize=(12, 8))
        plt.title(f"Frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        plt.show()

if __name__ == "__main__":
    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    yolo_detector = YOLODetector()
    
    MASK = True
    DISPLAY_FRAMES = False
    DISPLAY_ALL_YOLO_OBJECTS = False  # New flag to display all YOLO detected objects
    # video_dir = "assets/video_short"
    video_dir = "downloaded_frames"
    inference_state = predictor.init_state(video_path=video_dir)
    
    frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    plt.figure(figsize=(12, 8))
    for ann_frame_idx in tqdm(range(len(frame_names)), desc="Processing frames"):
        ann_obj_id = 1
        image_path = os.path.join(video_dir, frame_names[ann_frame_idx])
        yolo_results = yolo_detector.detect(image_path)
        
        person_center = None
        largest_person_area = 0
        if len(yolo_results) > 0:
            for result in yolo_results[0].boxes:
                if result.cls[0].item() == 0:
                    x1, y1, x2, y2 = result.xyxy[0]
                    area = (x2 - x1) * (y2 - y1)
                    if area > largest_person_area:
                        largest_person_area = area
                        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
                        person_center = [x_center.item(), y_center.item()]

        points = [person_center] if person_center else [[950, 600], [500, 300]]
        labels = np.ones(len(points), dtype=np.int32)
        points = np.array(points, dtype=np.float32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        if MASK:
            inverted_mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy()
            frame = np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
            masked_frame = np.array(Image.open('/home/gabi/Desktop/eiffel.jpg'))
            masked_frame = cv2.resize(masked_frame, (frame.shape[1], frame.shape[0]))
            masked_frame[inverted_mask] = frame[inverted_mask]
        else:
            masked_frame = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))

        # masked_frame = masked_frame[:760, :]

        if DISPLAY_FRAMES:
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(masked_frame)
            if not MASK:
                show_points(points, labels, plt.gca())
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
            
            if DISPLAY_ALL_YOLO_OBJECTS and len(yolo_results) > 0:
                ax = plt.gca()
                for result in yolo_results[0].boxes:
                    x1, y1, x2, y2 = result.xyxy[0].cpu()
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
                    class_name = yolo_detector.model.names[int(result.cls)]
                    ax.text(x1, y1, class_name, bbox=dict(facecolor='red', alpha=0.5), fontsize=8, color='white')
            
            plt.draw()
            plt.pause(0.001)
            if len(frame_names) == 1:
                plt.show(block=True)
            plt.clf()
        else:
            if 'video_writer' not in locals():
                video_writer = VideoWriterAux('output_video.mp4', 30.0, (masked_frame.shape[1], masked_frame.shape[0]))

            if len(masked_frame.shape) == 2 or masked_frame.shape[2] == 1:
                frame_to_write = cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2BGR)
            elif masked_frame.shape[2] == 3:
                frame_to_write = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
            else:
                frame_to_write = masked_frame
            
            video_writer.write_frame(frame_to_write)

    if not DISPLAY_FRAMES:
        video_writer.release()
        print("Video writing completed. Output saved as 'output_video.mp4'.")