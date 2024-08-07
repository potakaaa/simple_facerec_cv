import os
import face_recognition
import cv2
import numpy as np
import glob

class FaceRecognition:
    def __init__(self):
         pass

    def load_known_faces(self, known_faces_dir):
        known_face_encodings = []
        known_face_names = []
        
        i = 0
        for filename in os.listdir(known_faces_dir):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                known_face_encodings.append(face_encodings[0])

                known_face_names.append(os.path.splitext(filename)[0])
                i += 1
        
        print(f"{i} faces found")
        
        return known_face_encodings, known_face_names

    def draw_rectangle(self, frame, x, y, w, h, name="Stranger"):
       
        color = (0, 0, 255)
        thickness = 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

    def maintain_aspect_ratio(self, frame, screen_width, screen_height):
        frame_height, frame_width = frame.shape[:2]
  
        aspect_ratio = frame_width / frame_height

   
        if screen_width / screen_height > aspect_ratio:
            new_height = screen_height
            new_width = int(aspect_ratio * new_height)
        else:
            new_width = screen_width
            new_height = int(new_width / aspect_ratio)

        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Create a new frame with black borders
        new_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        x_offset = (screen_width - new_width) // 2
        y_offset = (screen_height - new_height) // 2
        new_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

        return new_frame