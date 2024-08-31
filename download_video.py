import yt_dlp
import cv2
import os
import numpy as np

def download_frames(url, start_time, num_frames, output_dir, resize_factor=5.0):
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]',
        'outtmpl': 'temp_video.%(ext)s'
    }

    # Download the video
    if not os.path.exists('temp_video.mp4'):
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    # Open the video file
    video = cv2.VideoCapture('temp_video.mp4')

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

# Usage
    url = 'https://www.youtube.com/watch?v=7L8vDnSDrXc&list=PLwLYTnsgqjj8St3u1Xy8OLMnej7tq5JWy&index=2'
    start_time = 17 * 60 + 4  # 22:27 in seconds
    num_frames = 100
    output_dir = 'downloaded_frames'

    download_frames(url, start_time, num_frames, output_dir)

