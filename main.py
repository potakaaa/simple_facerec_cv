import cv2
from simple_facerec import SimpleFacerec
import screeninfo
import numpy as np

# Initialize face recognition and load images
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Open the webcam
cam = cv2.VideoCapture(1)

# Get screen dimensions
screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

def resize_frame_aspect_ratio(frame, target_width, target_height):
    (h, w) = frame.shape[:2]
    aspect_ratio = w / h

    # Determine new dimensions while preserving aspect ratio
    if target_width / target_height > aspect_ratio:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # Resize the frame to the new dimensions
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Create a black background with target dimensions
    background = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Center the resized frame on the background
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

    return background

while True:
    ret, frame = cam.read()
    
    if not ret:
        break
    
    # Detect faces
    face_loc, face_name = sfr.detect_known_faces(frame)
    for loc, name in zip(face_loc, face_name):
        y1, x1, y2, x2 = loc[0], loc[3], loc[1], loc[2]
        cv2.putText(frame, name, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # Resize the frame while preserving aspect ratio
    frame_resized = resize_frame_aspect_ratio(frame, screen_width, screen_height)

    # Display the resized frame
    cv2.namedWindow("CS3 Lakip", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("CS3 Lakip", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("CS3 Lakip", frame_resized)

    key = cv2.waitKey(1)
    if key == ord('s'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
