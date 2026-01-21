import cv2
import numpy as np

def select_pixel(image, label):
    def draw_crosshair(image, x, y, color=(0, 0, 255), thickness=1):
        cv2.line(image, (x, 0), (x, image.shape[0]), color, thickness)
        cv2.line(image, (0, y), (image.shape[1], y), color, thickness)

    # Load the image
    if image is None:
        print("Error: Could not load image.")
        return None

    # Create a window
    cv2.namedWindow("Image Pixel Selector")

    selected_pixel = []
    mouse_x, mouse_y = 0, 0

    def add_text(im: np.ndarray, label: str):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Specify the position and size of the text
        org = (50, 50)
        fontScale = 2
        color = (255, 255, 255)  # Blue color in BGR
        thickness = 5
        # Add the text to the image
        cv2.putText(im, label, org, font, fontScale, color, thickness)

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_pixel, mouse_x, mouse_y
        mouse_x, mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_pixel.append((x, y))

    cv2.setMouseCallback("Image Pixel Selector", mouse_callback)

    while len(selected_pixel) < 5:
        display_image = image.copy()
        #
        # # Draw crosshair at current mouse position
        # draw_crosshair(display_image, mouse_x, mouse_y, color=(0, 255, 0), thickness=1)

        add_text(display_image, label)
        cv2.imshow("Image Pixel Selector", display_image)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return selected_pixel

    cv2.destroyAllWindows()
    return selected_pixel

if __name__ == "__main__":
    # reading the image
    img = cv2.imread('/home/gabi/GitHub/Experiments/segment-anything-2/downloaded_frames_tag2/000.jpg', 1)
    xy = select_pixel(img,"select")
    print(xy)
    # close the window
