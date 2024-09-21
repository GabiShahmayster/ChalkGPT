import cv2
import os

def extract_video_frames(video_path, output_dir):
    """
    Extract all frames from a video file and save them to an output directory.

    Args:
        video_path (str): Path to the input MP4 video file.
        output_dir (str): Path to the output directory where frames will be saved.

    Returns:
        None
    """
    # Check if the output directory exists, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through each frame and save it to the output directory
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to a PNG file and save it to the output directory
        png_path = os.path.join(output_dir, f"{i:04d}.jpg")
        cv2.imwrite(png_path, frame)

    # Release the video capture object
    cap.release()
