# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Installation

```bash
pip install -e .
pip install ultralytics shapely transformers pillow scikit-learn scipy
```

**Requirements:** Python >= 3.10, torch >= 2.3.1, torchvision >= 0.18.1. GPU (CUDA) is strongly recommended.

**Model weights:** SAM 2 and YOLO Pose download automatically on first run. The custom hold detector must be placed at `weights/holds/v1/weights/best.pt`.

## Running

```bash
# Run the main pipeline (edit config at bottom of file)
python chalk_gpt.py

# Run VLM move analysis script
python test_route_color_detection_agent.py --dir <frames_dir> --frames 10,25,40
python test_route_color_detection_agent.py --backend ollama --model llava --frames 10,25,40

# Run unit tests
python -m unittest run_tests.py
python -m unittest run_tests.TestVideoWriterAux.test_video_writer_aux
```

## Architecture

ChalkGPT is a multi-model CV pipeline for analyzing bouldering videos. The main entry point is `chalk_gpt.py`, which contains the `ChalkGpt` class and all core logic.

### Pipeline flow (`ChalkGpt.main()`)

1. **Frame extraction** – `video_extractor.py:extract_video_frames` extracts frames from video into a directory (named after the video stem), which becomes the input for SAM 2.

2. **Climber segmentation** – SAM 2 (`facebook/sam2-hiera-small` from HuggingFace) segments and tracks the climber. The user manually clicks to select the climber on frame 0 (`src/mouse_click.py:select_pixel`).

3. **Hold detection** – Custom YOLOv8 model detects climbing holds in every frame and stores them as `HoldBBox` objects in `FrameData.holds`.

4. **Pose estimation** – YOLOv8m-Pose runs on the climber-masked image to get 17 COCO keypoints. Wrists (9,10) and ankles (15,16) are used to determine which holds the climber is touching (`is_climbed_route=True`).

5. **Camera motion estimation** – SuperPoint (`src/superpoint.py`) detects keypoints; SuperGlue (`src/superglue.py`) matches them across consecutive frames. RANSAC fits an affine transform, which is accumulated into `transform_to_world[frame_id]`.

6. **World frame construction** – An extended canvas is created to accommodate all camera motion. `H_ext` and `W_ext` define the padding added around the original frame dimensions.

7. **Hold clustering & labeling** – All hold centers (from frames where holds are near the climber) are projected into world coordinates and clustered with DBSCAN (`eps=10px, min_samples=2`). Cluster centroids become `holds_world` with unique IDs.

8. **Hold matching** – Per-frame holds are matched to world holds via k-d tree lookup. World holds acquire a representative image crop for color classification.

9. **Color classification** – CLIP (`openai/clip-vit-base-patch32`) classifies each hold crop against 10 color labels in batch.

10. **Route identification** – The most common color among `is_climbed_route` holds is the route color.

11. **Visualization** – Side-by-side output video: left = stabilized world frame with climber skeleton and route holds, right = original footage.

### Caching

Intermediate results (SAM 2 masks, motion transforms, hold detections) are pickled to `cache/<frames_dir_stem>.pkl`. Controlled by `ChalkGptConfig.save_to_disk` and `try_to_load_from_disk`.

### Key data structures

- **`ChalkGptConfig`** – All pipeline parameters; either `images_dir` or `video_file` must be set.
- **`FrameData`** – Per-frame container: `id`, `im_path`, `holds: List[HoldBBox]`, `transform_to_world`, `pose`.
- **`HoldBBox(BoundingBox)`** – A detected hold; adds `id`, `color_name`, `bgr_color`, `is_near_climber`, `is_climbed_route`, `is_labeled`, and `image` (crop).
- **`KeypointData` / `KeypointMatchingResults`** (`src/base.py`) – Dual-representation (OpenCV + PyTorch tensors) wrappers for keypoints and matches; implement `__getstate__`/`__setstate__` for pickling.

### Supporting modules

| File | Purpose |
|------|---------|
| `src/superpoint.py` | SuperPoint keypoint detector wrapper |
| `src/superglue.py` | SuperGlue matcher wrapper |
| `src/video_writer.py` | `VideoWriterChalkGpt` (lazy-init on first frame) + `add_clock_to_image` |
| `src/holds_clustering.py` | MeanShift color-histogram clustering (experimental, not used in main pipeline) |
| `src/mouse_click.py` | Interactive pixel selection for seeding SAM 2 |
| `video_extractor.py` | Frame extraction from video files |
| `test_route_color_detection_agent.py` | Standalone VLM script using SmolVLM2/Gemma/Qwen/Ollama to describe climbing moves from frame sequences |
| `sam2/` | Vendored SAM 2 source (Meta AI) |

### Input conventions

- Input frames must be named as zero-padded integers (`0000.jpg`, `0001.jpg`, …) and sorted numerically.
- All images are read as BGR (OpenCV). CLIP and PIL utilities convert to RGB internally.
- Transformation matrices are 2×3 affine (returned by `cv2.estimateAffinePartial2D`); homogeneous 3×3 versions are used for point transforms.
