from typing import Optional
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

def auto_segment_climber_vlm_bbox(frame_path: str, vlm_model_key: str, device: torch.device,
                                    frames_dir: Optional[str] = None, num_frames: int = 5) -> Optional[np.ndarray]:
    """
    Use a local VLM to detect the climber's bounding box from a single frame.
    The model is loaded, used, and immediately deleted to free VRAM.
    Returns np.array([x1, y1, x2, y2], dtype=np.float32) on success, None on failure.
    """
    MODELS = {
        "qwen": "Qwen/Qwen2-VL-2B-Instruct",
        "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
        "256M": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "500M": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "2.2B": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "moondream": "vikhyatk/moondream2",
        "deepseek": "deepseek-ai/deepseek-vl-7b-base",
        "gemma4": "google/gemma-4-E4B-it"
    }
    model_id = MODELS.get(vlm_model_key)
    if model_id is None:
        print(f"[VLM] Unknown vlm_model_key: '{vlm_model_key}'. "
              f"Valid keys: {list(MODELS.keys())}")
        return None

    # Load primary frame to get dimensions
    pil_image = Image.open(frame_path).convert('RGB')
    img_w, img_h = pil_image.size

    # Load multiple frames for temporal context if available
    pil_images = [pil_image]
    if frames_dir:
        frames = sorted(Path(frames_dir).glob("*.jpg")) + sorted(Path(frames_dir).glob("*.png"))
        if len(frames) > 1:
            # Sample evenly across the video
            step = max(1, len(frames) // num_frames)
            for frame_file in frames[::step][1:num_frames]:
                try:
                    pil_images.append(Image.open(frame_file).convert('RGB'))
                except Exception:
                    pass

    prompt_text = (
        f"Bouldering video with {len(pil_images)} frames. "
        "Find the climber - the person consistently visible in all frames touching the wall with their back to camera. "
        "Ignore wall texture, holds, and background changes due to camera motion. "
        "Return the bounding box of the climber's body as: (left,top),(right,bottom). "
        "Output ONLY the coordinates."
    )

    print(f"[VLM] Loading {model_id} for automatic climber detection...")
    processor = AutoProcessor.from_pretrained(model_id)
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    # Format depends on model - Qwen2-VL uses chat template, SmolVLM2 uses <image> token
    if "qwen" in model_id.lower():
        # Use Qwen-style chat template with multiple images
        content = []
        for _ in pil_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        text_inputs = processor.apply_chat_template(messages, tokenize=False)
        inputs = processor(text=text_inputs, images=pil_images, return_tensors="pt")
    else:
        # Use SmolVLM2-style <image> token
        text = "".join(["<image>" for _ in pil_images]) + prompt_text
        inputs = processor(images=pil_images, text=text, return_tensors="pt")

    # Move all inputs to model's device (CUDA if available)
    model_device = next(vlm_model.parameters()).device
    inputs = {k: v.to(model_device) if hasattr(v, 'to') else v
              for k, v in inputs.items()}

    generated_ids = vlm_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Strip echoed user turn if the model includes it
    if "Assistant:" in generated_text:
        generated_text = generated_text.split("Assistant:")[-1].strip()

    # Free VRAM before returning
    del vlm_model, processor, inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[VLM] Raw output: {generated_text!r}")

    # Parse bbox with multiple fallback strategies
    def _valid(x1, y1, x2, y2):
        return x2 > x1 and y2 > y1 and 0 <= x1 and x2 <= img_w and 0 <= y1 and y2 <= img_h

    result = None

    # Strategy 0: Qwen2-VL format (x1,y1),(x2,y2)
    match = re.search(r'\((\d+),(\d+)\)\s*,\s*\((\d+),(\d+)\)', generated_text)
    if match:
        try:
            x1, y1, x2, y2 = [float(match.group(i)) for i in range(1, 5)]
            print(f"[VLM] Parsed (x1,y1),(x2,y2) format: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            if _valid(x1, y1, x2, y2):
                result = (x1, y1, x2, y2)
                print(f"[VLM] Valid coordinates, no scaling needed")
        except ValueError:
            pass

    # Strategy 1: JSON object {x1, y1, x2, y2}
    if result is None:
        for m in re.finditer(r'\{[^}]+\}', generated_text):
            try:
                obj = json.loads(m.group())
                x1 = obj.get('x1') or obj.get('xmin')
                y1 = obj.get('y1') or obj.get('ymin')
                x2 = obj.get('x2') or obj.get('xmax')
                y2 = obj.get('y2') or obj.get('ymax')
                if all(v is not None for v in [x1, y1, x2, y2]):
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    if _valid(x1, y1, x2, y2):
                        result = (x1, y1, x2, y2)
                        break
            except (json.JSONDecodeError, ValueError):
                pass

    # Strategy 2: JSON array [x1, y1, x2, y2]
    if result is None:
        for m in re.finditer(r'\[[\d\s,.\-]+\]', generated_text):
            try:
                arr = json.loads(m.group())
                if len(arr) >= 4:
                    x1, y1, x2, y2 = [float(arr[i]) for i in range(4)]
                    if _valid(x1, y1, x2, y2):
                        result = (x1, y1, x2, y2)
                        break
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    # Strategy 3: Comma-separated integers (including with spaces)
    if result is None:
        # Try to find comma-separated numbers
        match = re.search(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', generated_text)
        if match:
            try:
                x1, y1, x2, y2 = [float(match.group(i)) for i in range(1, 5)]
                if _valid(x1, y1, x2, y2):
                    result = (x1, y1, x2, y2)
            except ValueError:
                pass

    # Strategy 4: Any four integers in order
    if result is None:
        nums = re.findall(r'\b\d+\b', generated_text)
        if len(nums) >= 4:
            try:
                x1, y1, x2, y2 = [float(nums[i]) for i in range(4)]
                if _valid(x1, y1, x2, y2):
                    result = (x1, y1, x2, y2)
            except (ValueError, IndexError):
                pass

    # If not already found, check if first strategy found coords but they were out of bounds
    # In that case, scale them down proportionally
    if result is None:
        # Try to find ANY four numbers and handle them
        nums = re.findall(r'\d+', generated_text)
        if len(nums) >= 4:
            try:
                x1, y1, x2, y2 = [float(nums[i]) for i in range(4)]

                # Smart scaling: if coordinates exceed bounds, scale proportionally
                if x2 > img_w or y2 > img_h:
                    # Scale both dimensions proportionally to fit in image
                    scale_x = img_w / x2 if x2 > 0 else 1.0
                    scale_y = img_h / y2 if y2 > 0 else 1.0
                    scale = min(scale_x, scale_y)  # Use smaller scale to fit both dimensions

                    x1 = x1 * scale
                    y1 = y1 * scale
                    x2 = x2 * scale
                    y2 = y2 * scale
                    print(f"[VLM] Scaled down by {scale:.2f}: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

                # Final clamp to ensure within bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                # Ensure minimum box size
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    result = (x1, y1, x2, y2)
            except (ValueError, IndexError):
                pass

    if result is None:
        print("[VLM] Failed to parse bounding box from VLM output.")
        return None

    x1, y1, x2, y2 = result
    print(f"[VLM] Detected climber bbox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    return np.array([x1, y1, x2, y2], dtype=np.float32)