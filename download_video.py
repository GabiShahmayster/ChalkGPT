from pathlib import Path

import yt_dlp
import cv2
import os
import numpy as np

def save_video(video_path: str, start_time, num_frames, output_path, resize_factor, fps=30):
    # Open the video file
    video = cv2.VideoCapture(video_path)

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


def download_frames(url, start_time, num_frames, output_dir, resize_factor=5.0, force_download: bool = False):
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=720][abr<250]+bestaudio/best[height<=720]',
        'outtmpl': 'temp_video.%(ext)s'
    }
    """
    youtube-dl --get-title -f 'bestvideo[ext=mp4][height<=640][abr<250]+bestaudio/best[height<=640]' https://www.youtube.com/watch?v=VIDEO_ID --get-title 00:00:10-00:00:20
    """
    # Download the video
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    video_path = 'temp_video.mp4'
    save_video(video_path=video_path,
               output_path=output_dir,
               resize_factor=resize_factor,
               start_time=start_time,
                num_frames=num_frames,
               fps=30)
    if os.path.exists('temp_video.mp4'):
        os.remove('temp_video.mp4')
    elif os.path.exists('temp_video.webm'):
        os.remove('temp_video.webm')
    elif os.path.exists('temp_video.mkv'):
        os.remove('temp_video.mkv')
    else:
        pass

if __name__ == "__main__":
    # Usage
    url = 'https://youtube.com/shorts/RCNDnQ6kIPc?si=b9YouNhN_vKIrJyC'
    start_time = 0  # 22:27 in seconds
    num_frames =1000
    output_dir = 'v5_green'

    download_frames(url, start_time, num_frames, output_dir, force_download=True, resize_factor=1)

