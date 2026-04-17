from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

def auto_segment_climber_vlm_bbox(frame_path: str, vlm_model_key: str, device: torch.device) -> Optional[np.ndarray]:
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

    pil_image = Image.open(frame_path).convert('RGB')
    img_w, img_h = pil_image.size

    prompt_text = (
        f"There is a person rock climbing on a bouldering wall in this image. "
        f"Return ONLY the bounding box of the climber as: x1,y1,x2,y2 (pixel coordinates). "
        f"Do not explain. Output only the four numbers separated by commas. "
        f"Example: 120,80,350,650"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": frame_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Deferred import — zero overhead when VLM is disabled

    print(f"[VLM] Loading {model_id} for automatic climber detection...")
    processor = AutoProcessor.from_pretrained(model_id)
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, 'to') else v
              for k, v in inputs.items()}

    generated_ids = vlm_model.generate(**inputs, max_new_tokens=60, do_sample=False)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Strip echoed user turn if the model includes it
    if "Assistant:" in generated_text:
        generated_text = generated_text.split("Assistant:")[-1].strip()

    # Free VRAM before returning
    del vlm_model, processor, inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[VLM] Raw output: {generated_text!r}")

    # result = self._parse_vlm_bbox(generated_text, img_w, img_h)
    if generated_text is None:
        print("[VLM] Failed to parse bounding box from VLM output.")
        return None

    x1, y1, x2, y2 = [int(f) for f in generated_text.split(',')]
    print(f"[VLM] Detected climber bbox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    return np.array([x1, y1, x2, y2], dtype=np.float32)