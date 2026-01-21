# ChalkGPT: Climbing Route Detection and Tracking

**Computer Vision Pipeline for Bouldering Analysis**

A computer vision system that automatically detects climbing routes, tracks climbers, and identifies holds using deep learning models including SAM 2, YOLO, and SuperPoint/SuperGlue.

## Overview

ChalkGPT is a comprehensive computer vision pipeline designed to analyze bouldering videos. The system:

- **Tracks climbers** in real-time using SAM 2 video segmentation with optional pose estimation
- **Detects climbing holds** using custom-trained YOLO models
- **Matches holds across frames** using keypoint matching (SuperPoint/SuperGlue)
- **Identifies climbing routes** by analyzing which holds are touched based on pose keypoints (hands/feet)
- **Classifies hold colors** using CLIP zero-shot classification
- **Generates stabilized output videos** showing the complete climbing wall with tracked movements

## Key Features

- **Multi-model pipeline**: Combines SAM 2, YOLOv8 Pose, custom YOLO hold detector, SuperPoint/SuperGlue, and CLIP
- **Pose-based route detection**: Uses hand and foot keypoints to accurately identify touched holds
- **Camera motion compensation**: Estimates frame-to-frame transformations to create a stabilized "world frame"
- **Hold color classification**: Automatically identifies hold colors using CLIP vision-language model
- **Route identification**: Determines the climbing route color by analyzing touched holds
- **Video output**: Creates side-by-side visualization with world frame and raw footage

## Installation

### Prerequisites

The system requires `python>=3.10`, `torch>=2.3.1`, and `torchvision>=0.18.1`.

### Setup

```bash
git clone https://github.com/yourusername/chalkgpt.git
cd chalkgpt

# Install dependencies
pip install -e .

# Install additional requirements
pip install ultralytics shapely transformers pillow scikit-learn scipy
```

### Model Weights

The system automatically downloads SAM 2 and YOLO Pose models. You'll need to provide:

1. **Custom YOLO hold detector**: Place trained weights at `weights/holds/v1/weights/best.pt`
2. **SAM 2 checkpoints**: Automatically downloaded from Hugging Face on first run

## Quick Start

### Basic Usage

```python
import torch
from chalk_gpt import ChalkGpt, ChalkGptConfig

config = ChalkGptConfig(
    save_to_disk=True,
    try_to_load_from_disk=True,
    video_file='climbing_video.mp4',  # Or use images_dir for frame directory
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    matcher_threshold=0.9,
    detector_threshold=0.1,
    max_frames=None,  # Process all frames
    save_video_path='output',
    video_fps=30
)

chalk_gpt = ChalkGpt(config)
chalk_gpt.main()
```

### Configuration Options

```python
@dataclasses.dataclass
class ChalkGptConfig:
    save_to_disk: bool                    # Cache intermediate results
    try_to_load_from_disk: bool           # Load cached data if available
    matcher_threshold: float              # SuperGlue matching threshold (0.0-1.0)
    detector_threshold: float             # SuperPoint keypoint threshold
    device: torch.device                  # Computation device
    images_dir: str = None                # Input frame directory
    video_file: str = None                # Input video file
    mask_for_matching: bool = False       # Apply climber mask to keypoint matching
    max_frames: int = None                # Limit number of frames processed
    hold_to_traj_association_distance_px: float = 10  # Hold-contact distance threshold
    static_video_shift_thr_px: float = 10.0           # Camera motion threshold
    save_video_path: str = None           # Output video directory
    video_fps: int = 30                   # Output video framerate
```

## Pipeline Architecture

### 1. Frame Processing & Segmentation
- Extracts frames from video or loads from directory
- Uses **SAM 2** to segment and track the climber across frames
- Detects climbing holds using custom **YOLO** model in every frame

### 2. Pose Estimation
- Applies **YOLOv8 Pose** to detect keypoints (17 COCO keypoints)
- Focuses on contact points: hands (wrists) and feet (ankles)
- Identifies touched holds based on keypoint proximity

### 3. Camera Motion Estimation
- Extracts **SuperPoint** keypoints from consecutive frames
- Matches keypoints using **SuperGlue**
- Estimates affine transformations using RANSAC
- Builds accumulated transformation matrix for world frame alignment

### 4. Hold Matching & Clustering
- Clusters hold detections across frames using DBSCAN
- Matches frame-level holds to world-frame holds using k-d tree
- Assigns unique IDs to tracked holds throughout the video

### 5. Color Classification
- Extracts hold crops from frames
- Uses **CLIP** (openai/clip-vit-base-patch32) for zero-shot color classification
- Supports 10 common climbing hold colors (red, blue, green, yellow, orange, purple, black, white, pink, gray)
- Batch processes all holds for GPU efficiency

### 6. Route Identification
- Analyzes touched holds (based on pose keypoints)
- Determines the most common hold color among touched holds
- Identifies the climbing route color

### 7. Visualization
- Creates extended "world frame" to accommodate camera motion
- Warps frames into world coordinate system
- Overlays climber skeleton, hold labels, and route information
- Generates side-by-side output: world view + raw footage

## Model Details

| **Component** | **Model** | **Purpose** |
|---------------|-----------|-------------|
| Segmentation & Tracking | SAM 2 (Hiera-Small) | Real-time climber segmentation across video frames |
| Pose Estimation | YOLOv8m-Pose | 17-point COCO keypoint detection for contact point identification |
| Hold Detection | Custom YOLOv8 | Bounding box detection of climbing holds |
| Keypoint Detection | SuperPoint | Sparse keypoint extraction for camera motion estimation |
| Keypoint Matching | SuperGlue (Outdoor) | Robust keypoint matching across frames |
| Color Classification | CLIP (ViT-B/32) | Zero-shot hold color classification |

## Key Classes

### `ChalkGpt`
Main orchestrator class that coordinates the entire pipeline.

### `HoldBBox`
Represents a detected climbing hold with:
- Bounding box coordinates
- Unique ID for tracking
- Color information (name and BGR tuple)
- Flags: `is_near_climber`, `is_climbed_route`, `is_labeled`

### `FrameData`
Stores per-frame information:
- Frame ID and image path
- List of detected holds
- Transformation to world coordinates
- Pose estimation results

### `BoundingBox`
Base class for bounding boxes with:
- Geometric operations (center, crop, transform)
- Shapely polygon representation
- Visualization methods

## Output

The system generates:

1. **Video output**: Side-by-side visualization showing:
   - Left: Stabilized world frame with climber trajectory and route holds
   - Right: Original raw footage
   - Timestamp overlay

2. **Cached data** (optional): Serialized pickle file containing:
   - Frame metadata
   - Video segments (masks)
   - Relative motion transformations
   - Hold detections

## Performance Considerations

- **GPU required**: Pipeline is optimized for CUDA execution
- **Memory usage**: SAM 2 video predictor can be offloaded to CPU if needed
- **Processing speed**: ~5-10 FPS depending on video resolution and GPU
- **Caching**: Enable `save_to_disk=True` to avoid reprocessing

## Limitations

- Requires clear visibility of climber throughout video
- Hold detector trained on specific dataset - may need retraining for different wall styles
- Camera motion estimation works best with textured backgrounds
- Pose estimation requires unoccluded view of climber's hands and feet

## Future Improvements

- [ ] Support for multiple climbers
- [ ] Real-time processing mode
- [ ] 3D reconstruction of climbing wall
- [ ] Difficulty grading based on hold usage
- [ ] Movement quality analysis
- [ ] Integration with climbing gym databases

## Citation

This project builds upon several excellent open-source models:

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

## License

Apache 2.0 License (inherited from SAM 2)

## Acknowledgments

- Meta AI for SAM 2
- Ultralytics for YOLO
- OpenAI for CLIP
- Magic Leap for SuperPoint/SuperGlue
