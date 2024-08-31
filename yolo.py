import dataclasses

from ultralytics import YOLO
import os

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load a pretrained YOLOv8n model
        self.model = YOLO(model_path)

    def detect(self, image_path):
        # Run inference on the image
        results = self.model(image_path)
        return results

if __name__ == "__main__":

    # Example usage:
    image_dir = '/home/gabi/GitHub/Experiments/segment-anything-2/assets/video'
    detector = YOLODetector()  # Create an instance of YOLODetector
    import cv2

    # Get all image files and sort them
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sorted_image_files = sorted(image_files)

    for image_file in sorted_image_files:
            image_path = os.path.join(image_dir, image_file)
            yolo_results = detector.detect(image_path)  # Use the detect method

            for result in yolo_results:
                img = result.plot()  # Get the plotted image
                cv2.imshow("YOLO Detection", img)
                
                # Wait for spacebar press
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:  # 32 is the ASCII code for spacebar
                        break
                    elif key == 27:  # 27 is the ASCII code for ESC
                        print("ESC pressed. Exiting...")
                        cv2.destroyAllWindows()
                        exit()

    cv2.destroyAllWindows()  # Close all windows after the loop
