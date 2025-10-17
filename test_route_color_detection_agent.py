#!/usr/bin/env python3
"""
Test file demonstrating SmolVLM2 for climbing move analysis.

This standalone script shows how to use SmolVLM2 to analyze climbing moves
from frame sequences extracted by the ChalkGPT pipeline.

Supports two backends:
- transformers: HuggingFace transformers (requires GPU for best performance)
- ollama: Local Ollama server (requires ollama to be running)

Usage:
    # Using HuggingFace transformers (default)
    python test_smolvlm2_moves.py --dir anna --frames 10,25,40
    python test_smolvlm2_moves.py --dir v5_green --start 50 --end 120 --num-frames 5
    python test_smolvlm2_moves.py --model 2.2B --frames 15,30,45 --show

    # Using Ollama (requires: ollama pull llava)
    python test_smolvlm2_moves.py --backend ollama --model llava --frames 10,25,40
    python test_smolvlm2_moves.py --backend ollama --model llava --start 50 --end 120 --num-frames 5
"""

import argparse
import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Try to import ollama for local inference
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama not available. Install with: pip install ollama")

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization disabled.")


class SmolVLM2MoveAnalyzer:
    """
    Analyzes climbing moves using SmolVLM2 vision-language model.
    """

    # Available model variants
    MODELS = {
        "256M": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "500M": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "2.2B": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "google": "google/gemma-3-4b-it",
        "qwen": "Qwen/Qwen2-VL-2B-Instruct",
    }
    OLLAMA_MODELS = {
        "llava":"llava:latest"
    }
    def __init__(self, model_size: str = "500M", device: str = None, backend: str = "transformers"):
        """
        Initialize SmolVLM2 model for climbing move analysis.

        Args:
            model_size: Model size variant ("256M", "500M", "2.2B", etc.)
            device: Device to use ("cuda" or "cpu"). Auto-detects if None.
            backend: Backend to use ("transformers" or "ollama")
        """
        # if model_size not in self.MODELS:
        #     raise ValueError(f"Invalid model size. Choose from: {list(self.MODELS.keys())}")

        self.model_size = model_size
        if backend == "ollama":
            self.model_path = self.OLLAMA_MODELS.get(model_size)
        else:
            self.model_path = self.MODELS[model_size]
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*60}")
        print(f"SmolVLM2 Climbing Move Analysis")
        print(f"{'='*60}")
        print(f"\nBackend: {self.backend}")
        print(f"Model: {self.model_path}")

        if backend == "ollama":
            if not OLLAMA_AVAILABLE:
                raise RuntimeError("Ollama backend requested but ollama package not installed. Run: pip install ollama")
            print("Using Ollama for local inference")
            print("✓ Ollama client initialized")
            self.processor = None
            self.model = None
        else:
            print(f"Device: {self.device}")

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_path)

            load_kwargs = {
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
            }

            # Try to use flash attention if available
            try:
                load_kwargs["_attn_implementation"] = "flash_attention_2"
                print("Using flash_attention_2 for faster inference")
            except Exception:
                print("flash_attention_2 not available, using default attention")

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                **load_kwargs
            ).to(self.device)

            # Calculate approximate VRAM usage
            param_count = sum(p.numel() for p in self.model.parameters())
            vram_gb = (param_count * 2) / (1024**3)  # 2 bytes per param for bfloat16
            print(f"✓ Model loaded (~{vram_gb:.1f}GB VRAM)")

    def load_frames(self, directory: str, frame_indices: List[int]) -> Tuple[List[str], List[Image.Image]]:
        """
        Load frames from directory.

        Args:
            directory: Directory containing extracted frames
            frame_indices: List of frame indices to load

        Returns:
            Tuple of (frame_paths, frame_images)
        """
        # Get sorted frame list
        frame_files = sorted(
            [f for f in os.listdir(directory)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        frame_paths = []
        frame_images = []

        for idx in frame_indices:
            if idx >= len(frame_files):
                print(f"Warning: Frame {idx} out of range (max: {len(frame_files)-1})")
                continue

            frame_path = os.path.join(directory, frame_files[idx])
            frame_paths.append(frame_path)
            frame_images.append(Image.open(frame_path).convert('RGB'))

        return frame_paths, frame_images

    def create_climbing_prompt(self,
                               frame_count: int,
                               route_color: str = None,
                               hold_info: str = None,
                               move_type_hint: str = None) -> str:
        """
        Create a climbing-specific prompt for move analysis.

        Args:
            frame_count: Number of frames in sequence
            route_color: Color of the climbing route (optional)
            hold_info: Information about holds being used (optional)
            move_type_hint: Hint about expected move type (optional)

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are an expert climbing coach analyzing a climbing move sequence.",
            f"\nYou are viewing {frame_count} frames showing a climbing movement.",
        ]

        if route_color:
            prompt_parts.append(f"\nRoute color: {route_color}")

        if hold_info:
            prompt_parts.append(f"\nHold information: {hold_info}")

        if move_type_hint:
            prompt_parts.append(f"\nExpected move type: {move_type_hint}")

        prompt_parts.append("""

What is the color of the holds which the climber's hands are touching?""")

        return "".join(prompt_parts)

    def _encode_image_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string for Ollama.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _analyze_move_ollama(self,
                            frame_paths: List[str],
                            route_color: str = None,
                            hold_info: str = None,
                            move_type_hint: str = None,
                            max_new_tokens: int = 150) -> Dict:
        """
        Analyze move using Ollama backend.

        Args:
            frame_paths: List of paths to frame images
            route_color: Optional route color information
            hold_info: Optional hold information
            move_type_hint: Optional hint about move type
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()

        # Create prompt
        prompt = self.create_climbing_prompt(
            frame_count=len(frame_paths),
            route_color=route_color,
            hold_info=hold_info,
            move_type_hint=move_type_hint
        )

        # Encode images to base64
        images = [self._encode_image_base64(path) for path in frame_paths]

        # Call Ollama API
        response = ollama.chat(
            model=self.model_path,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': images
            }],
            options={
                'num_predict': max_new_tokens,
            }
        )

        inference_time = time.time() - start_time
        description = response['message']['content'].strip()

        # Estimate tokens (Ollama doesn't always provide exact counts)
        input_tokens = response.get('prompt_eval_count', 0)
        output_tokens = response.get('eval_count', len(description.split()))
        tokens_per_sec = output_tokens / inference_time if inference_time > 0 else 0

        return {
            "description": description,
            "inference_time": inference_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_sec": tokens_per_sec,
            "model_size": self.model_size,
            "frame_count": len(frame_paths)
        }

    def analyze_move(self,
                    frame_paths: List[str],
                    route_color: str = None,
                    hold_info: str = None,
                    move_type_hint: str = None,
                    max_new_tokens: int = 150) -> Dict:
        """
        Analyze a climbing move from frame sequence.
        Routes to appropriate backend (Ollama or Transformers).

        Args:
            frame_paths: List of paths to frame images
            route_color: Optional route color information
            hold_info: Optional hold information
            move_type_hint: Optional hint about move type
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with analysis results
        """
        # Route to appropriate backend
        if self.backend == "ollama":
            return self._analyze_move_ollama(
                frame_paths=frame_paths,
                route_color=route_color,
                hold_info=hold_info,
                move_type_hint=move_type_hint,
                max_new_tokens=max_new_tokens
            )

        # Transformers backend
        start_time = time.time()

        # Build message content with images and text
        content = []

        # Add all frame images
        for frame_path in frame_paths:
            content.append({"type": "image", "url": frame_path})

        # Add text prompt
        prompt = self.create_climbing_prompt(
            frame_count=len(frame_paths),
            route_color=route_color,
            hold_info=hold_info,
            move_type_hint=move_type_hint
        )
        content.append({"type": "text", "text": prompt})

        # Create message
        messages = [{"role": "user", "content": content}]

        # Process input
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move to device with correct dtype
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        inputs = {k: v.to(self.device, dtype=dtype) if torch.is_floating_point(v)
                 else v.to(self.device) for k, v in inputs.items()}

        # Generate description
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens
            )

        # Decode output
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        inference_time = time.time() - start_time

        # Extract just the assistant's response
        full_text = generated_texts[0]
        # Try to extract after "Assistant:" or similar markers
        if "Assistant:" in full_text:
            description = full_text.split("Assistant:")[-1].strip()
        else:
            description = full_text.strip()

        # Calculate tokens generated
        input_tokens = inputs['input_ids'].shape[1]
        output_tokens = generated_ids.shape[1] - input_tokens
        tokens_per_sec = output_tokens / inference_time if inference_time > 0 else 0

        return {
            "description": description,
            "inference_time": inference_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_sec": tokens_per_sec,
            "model_size": self.model_size,
            "frame_count": len(frame_paths)
        }

    def visualize_analysis(self,
                          frame_images: List[Image.Image],
                          frame_indices: List[int],
                          analysis: Dict,
                          save_path: str = None):
        """
        Visualize frames and analysis results.

        Args:
            frame_images: List of frame images
            frame_indices: Frame indices
            analysis: Analysis results dictionary
            save_path: Optional path to save visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Skipping visualization.")
            return

        n_frames = len(frame_images)
        fig = plt.figure(figsize=(15, 8))

        # Create grid: frames on top, description below
        gs = fig.add_gridspec(2, n_frames, height_ratios=[3, 1], hspace=0.3)

        # Display frames
        for i, (img, idx) in enumerate(zip(frame_images, frame_indices)):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(img)
            ax.set_title(f"Frame {idx}", fontsize=10, fontweight='bold')
            ax.axis('off')

        # Display analysis text
        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis('off')

        text_content = f"Move Analysis (SmolVLM2-{analysis['model_size']})\n\n"
        text_content += f"{analysis['description']}\n\n"
        text_content += f"Performance: {analysis['inference_time']:.2f}s | "
        text_content += f"{analysis['tokens_per_sec']:.1f} tokens/s | "
        text_content += f"{analysis['output_tokens']} tokens"

        ax_text.text(0.5, 0.5, text_content,
                    ha='center', va='center',
                    fontsize=11,
                    wrap=True,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle("SmolVLM2 Climbing Move Analysis", fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.tight_layout()
        plt.show()


def print_analysis_results(analysis: Dict, frame_indices: List[int]):
    """Pretty print analysis results."""
    print(f"\n{'━'*60}")
    print(f"Move Analysis Results")
    print(f"{'━'*60}")
    print(f"\nFrame Sequence: {' → '.join(map(str, frame_indices))}")
    print(f"Duration: ~{(frame_indices[-1] - frame_indices[0]) / 30:.1f}s (assuming 30 FPS)")
    print(f"\nDescription:")
    print(f"{'-'*60}")
    print(f"{analysis['description']}")
    print(f"{'-'*60}")
    print(f"\nPerformance Metrics:")
    print(f"  • Inference time: {analysis['inference_time']:.2f}s")
    print(f"  • Tokens generated: {analysis['output_tokens']}")
    print(f"  • Generation speed: {analysis['tokens_per_sec']:.1f} tokens/s")
    print(f"  • Model size: SmolVLM2-{analysis['model_size']}")
    print(f"{'━'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test SmolVLM2 for climbing move analysis"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="downloaded_frames_tag2",
        help="Directory containing extracted frames"
    )
    parser.add_argument(
        "--frames",
        type=str,
        help="Comma-separated frame indices (e.g., '10,25,40')"
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Start frame for sequence"
    )
    parser.add_argument(
        "--end",
        type=int,
        help="End frame for sequence"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Number of frames to sample from sequence (default: 4)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["256M", "500M", "2.2B", "google"],
        help="Model size to use (default: 500M)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "ollama"],
        help="Backend to use for inference (default: transformers)"
    )
    parser.add_argument(
        "--route-color",
        type=str,
        help="Route color information"
    )
    parser.add_argument(
        "--hold-info",
        type=str,
        help="Information about holds being used"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visualization"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Path to save visualization"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Path to save analysis as JSON"
    )

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.dir):
        print(f"Error: Directory '{args.dir}' not found")
        return 1

    # Determine frame indices
    if args.frames:
        frame_indices = [int(x.strip()) for x in args.frames.split(',')]
    elif args.start is not None and args.end is not None:
        frame_indices = np.linspace(args.start, args.end, args.num_frames, dtype=int).tolist()
    else:
        # Default: sample some frames
        print("No frames specified. Using default: 10, 30, 50, 70")
        frame_indices = [i for i in range(0, 200, 20)]

    # Initialize analyzer
    analyzer = SmolVLM2MoveAnalyzer(model_size=args.model, backend=args.backend)

    # Load frames
    print(f"\nLoading frames {frame_indices} from '{args.dir}/'...")
    frame_paths, frame_images = analyzer.load_frames(args.dir, frame_indices)

    if not frame_paths:
        print("Error: No frames loaded")
        return 1

    print(f"✓ Loaded {len(frame_paths)} frames")

    # Analyze move
    print(f"\nAnalyzing climbing move...")
    analysis = analyzer.analyze_move(
        frame_paths=frame_paths,
        route_color=args.route_color,
        hold_info=args.hold_info
    )

    # Print results
    print_analysis_results(analysis, frame_indices)

    # Save JSON if requested
    if args.output_json:
        output_data = {
            "frame_indices": frame_indices,
            "frame_paths": frame_paths,
            "analysis": analysis
        }
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Analysis saved to: {args.output_json}")

    # Visualize if requested
    if args.show or args.save:
        analyzer.visualize_analysis(
            frame_images=frame_images,
            frame_indices=frame_indices,
            analysis=analysis,
            save_path=args.save
        )

    return 0


if __name__ == "__main__":
    exit(main())
