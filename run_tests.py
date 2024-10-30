import glob
import unittest
import tempfile
import os
import cv2
import numpy as np
from wakepy.modes import keep
from ultralytics import YOLO


class TestVideoWriterAux(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, 'test_output.mp4')
        self.fps = 30
        self.frame_size = (640, 480)

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        os.rmdir(self.temp_dir)

    def test_predict_yolo_v11(self):
        model = YOLO("weights/v0/weights/best.pt")  # load a pretrained model (recommended for training)
        for im in glob.glob("/home/gabi/Datasets/ClimbingHolds/test/images/*.jpg"):
            results = model(im)  # return a list of Results objects

            # Process results list
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs
                result.show()  # display to screen

    def test_train_yolo_v11_holds(self):
        import comet_ml

        comet_ml.init(api_key="nF5qlywvVU26jsRXDepkJ48hM", project_name="climbing_holds")
        # Load a model
        model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)
        # Train the model
        with keep.running():
            results = model.train(data="/home/gabi/Datasets/ClimbingHolds/data.yaml", epochs=300, imgsz=640, dropout=0.5)

    def test_train_yolo_v11_shoes(self):
        import comet_ml

        comet_ml.init(api_key="nF5qlywvVU26jsRXDepkJ48hM", project_name="climbing_shoes")
        # Load a model
        model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)
        # Train the model
        with keep.running():
            results = model.train(data="/home/gabi/Datasets/ClimbingShoes/data.yaml", epochs=300, imgsz=640, dropout=0.5)

    def test_video_writer_aux(self):
        # Create a VideoWriterAux instance
        with VideoWriterAux(self.output_path, self.fps, self.frame_size) as writer:
            # Create a sample frame
            frame = np.random.randint(0, 256, (*self.frame_size, 3), dtype=np.uint8)
            
            # Write the frame
            writer.write_frame(frame)

        # Verify that the video file was created
        self.assertTrue(os.path.exists(self.output_path))

        # Read the video file and check its properties
        cap = cv2.VideoCapture(self.output_path)
        self.assertTrue(cap.isOpened())

        # Check video properties
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FPS)), self.fps)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), self.frame_size[0])
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), self.frame_size[1])

        # Read the frame and compare with the original
        ret, read_frame = cap.read()
        self.assertTrue(ret)
        self.assertTrue(np.array_equal(frame, read_frame))

        cap.release()

if __name__ == '__main__':
    unittest.main()