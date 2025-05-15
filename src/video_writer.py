import os

import cv2
import numpy as np


class VideoWriterChalkGpt:
    def __init__(self, output_path, fps=30, fourcc='mp4v'):
        """
        Initialize the video writer.

        Args:
            output_path (str): Path where the output video will be saved
            fps (int): Frames per second for the output video
            fourcc (str): Four character code for the video codec (default: mp4v)
        """
        self.output_path = output_path
        self.fps = fps
        self.fourcc = fourcc
        self.writer = None
        self.frame_size = None
        self.is_initialized = False

    def add_frame(self, frame):
        """
        Add a BGR frame to the video.

        Args:
            frame (numpy.ndarray): BGR image as numpy array
        """
        if not self.is_initialized:
            self._initialize_writer(frame)

        self.writer.write(frame)

    def _initialize_writer(self, frame):
        """
        Initialize the OpenCV VideoWriter object based on the first frame.

        Args:
            frame (numpy.ndarray): First BGR frame to determine dimensions
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a valid numpy array")

        height, width = frame.shape[:2]
        self.frame_size = (width, height)

        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            self.frame_size,
            isColor=True
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer at {self.output_path}")

        self.is_initialized = True

    def release(self):
        """
        Release the video writer and finalize the video file.
        """
        if self.writer is not None:
            self.writer.release()
            self.is_initialized = False
            print(f"Video saved to {self.output_path}")


def test_video_writer():
    """
    Test the VideoWriter class by creating a video with simple generated frames.
    Creates a 5-second video with moving colored rectangles.
    """
    # Setup test parameters
    output_path = "test_output.mp4"
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds

    # Create the video writer
    writer = VideoWriterChalkGpt(output_path, fps=fps)

    # Generate and write frames
    total_frames = fps * duration
    print(f"Generating {total_frames} frames for a {duration}-second video...")

    for i in range(total_frames):
        # Create a black canvas
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate position for moving rectangle
        pos_x = int((i / total_frames) * (width - 100))

        # Draw rectangles with different colors
        # Red rectangle (moving left to right)
        cv2.rectangle(frame, (pos_x, 100), (pos_x + 100, 200), (0, 0, 255), -1)

        # Green rectangle (fixed)
        cv2.rectangle(frame, (width // 4, height // 2), (width // 4 + 80, height // 2 + 80), (0, 255, 0), -1)

        # Blue rectangle (fixed)
        cv2.rectangle(frame, (width // 2, height // 4), (width // 2 + 120, height // 4 + 60), (255, 0, 0), -1)

        # Add frame number text
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add the frame to the video
        writer.add_frame(frame)

        # Print progress
        if i % 30 == 0:  # Report every second
            print(f"Progress: {i}/{total_frames} frames ({i / total_frames * 100:.1f}%)")

    # Finalize the video
    writer.release()

    # Verify the video was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        print(f"Test successful! Video created at {output_path} ({file_size:.2f} MB)")
        return True
    else:
        print("Test failed: Video file was not created")
        return False


def add_clock_to_image(image, frame_index, fps,
                       position='top-right', font_scale=0.7, thickness=2,
                       text_color=(255, 255, 255), bg_color=(0, 0, 0, 0.5)):
    """
    Add a clock overlay to an image using OpenCV.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image (OpenCV format - BGR color space)
    frame_index : int
        The current frame index (0-based)
    fps : float
        Frames per second of the video
    start_time : float or None, optional
        Unix timestamp for the start time. If None, uses current time - elapsed time
    format : str, optional
        Clock format: 'timestamp' (HH:MM:SS), 'elapsed' (MM:SS.ms), or 'frame' (frame number)
    position : str, optional
        Position of the clock: 'top-right', 'top-left', 'bottom-right', or 'bottom-left'
    font_scale : float, optional
        Font scale factor
    thickness : int, optional
        Thickness of the font
    text_color : tuple, optional
        Color of the text (BGR)
    bg_color : tuple, optional
        Background color (BGR, alpha) with alpha value from 0 to 1

    Returns:
    --------
    numpy.ndarray
        Image with clock overlay
    """
    # Create a copy of the image to avoid modifying the original
    result = image.copy()

    # Calculate time based on frame index and fps
    elapsed_seconds = frame_index / fps
    total_milliseconds = int(frame_index * 1000 / fps)

    # Calculate minutes, seconds, and milliseconds directly
    total_milliseconds = int(frame_index * 1000 / fps)
    minutes = total_milliseconds // 60000
    seconds = (total_milliseconds % 60000) // 1000
    milliseconds = total_milliseconds % 1000
    # Format as MM:SS.ms
    time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    # Get the size of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(time_str, font, font_scale, thickness)

    # Determine position coordinates
    h, w = result.shape[:2]
    padding = 50  # Padding from the edges

    if position == 'top-right':
        x = w - text_width - padding
        y = text_height + padding + baseline
    elif position == 'top-left':
        x = padding
        y = text_height + padding + baseline
    elif position == 'bottom-right':
        x = w - text_width - padding
        y = h - padding - baseline
    elif position == 'bottom-left':
        x = padding
        y = h - padding - baseline
    else:
        raise ValueError("Invalid position. Use 'top-right', 'top-left', 'bottom-right', or 'bottom-left'")

    # Create a background rectangle for better readability
    bg_opacity = bg_color[3] if len(bg_color) > 3 else 0.5
    bg_rect_p1 = (x - padding // 2, y - text_height - padding // 2)
    bg_rect_p2 = (x + text_width + padding // 2, y + padding // 2)

    # Create a copy for the background overlay
    overlay = result.copy()
    cv2.rectangle(overlay, bg_rect_p1, bg_rect_p2, bg_color[:3], -1)

    # Apply the overlay with transparency
    cv2.addWeighted(overlay, bg_opacity, result, 1 - bg_opacity, 0, result)

    # Add the text
    cv2.putText(result, time_str, (x, y), font, font_scale, text_color, thickness)

    return result



if __name__ == "__main__":
    # Run the test
    test_video_writer()