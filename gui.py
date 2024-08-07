import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime
from tkinter import simpledialog, messagebox, Button
from facerec_functions import FaceRecognition
import face_recognition


frs = FaceRecognition()
known_face_encodings, known_face_names = frs.load_known_faces("images/")

class CVApp:
    def __init__(self, window):
        self.window = window
        self.window.title("CS3 Lakip")

        self.video_capture = cv2.VideoCapture(0)
        self.current_image = None
        
        self.screen_width = 933
        self.screen_height = 700

        # Create a canvas with the fullscreen size
        self.canvas = tk.Canvas(window, width=self.screen_width, height=self.screen_height)
        self.canvas.pack()

        self.capture_button = tk.Button(window, text="Capture", command=self.save_current_frame)
        self.capture_button.pack()
        self.raw_frame = None 
        self.updateWebcam()       

        Button(root, text="Quit", command=root.destroy).pack()  

    def updateWebcam(self):
    
        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error loading face cascade.")
            return

        ret, frame = self.video_capture.read()
        if ret:
            self.raw_frame = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])

                if face_encoding:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])
                    name = "Stranger"

                    if True in matches:
                        match_index = matches.index(True)
                        name = known_face_names[match_index]

                frs.draw_rectangle(frame, x, y, w, h, name)

            frame = cv2.resize(frame, (self.screen_width, self.screen_height))

            self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            self.photo = ImageTk.PhotoImage(image=self.current_image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            key = cv2.waitKey(1)
            self.window.after(15, self.updateWebcam)

    def save_current_frame(self):
        if self.raw_frame is not None:
            key = cv2.waitKey(-1)
            cv2.imshow('Captured', self.raw_frame)
            filename = simpledialog.askstring("Input", "Enter a filename for the image:")
            if not filename:
                messagebox.showwarning("Input Required", "Filename cannot be empty.")
                cv2.destroyWindow("Captured")
                return

            if not filename.lower().endswith('.png'):
                filename += '.png'

            save_folder = "images"
            os.makedirs(save_folder, exist_ok=True)

            file_path = os.path.join(save_folder, filename)

            raw_image = Image.fromarray(cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2RGB))
            raw_image.save(file_path)
            cv2.destroyWindow("Captured")
            print(f"Image saved to {file_path}")

root = tk.Tk()
root.attributes("-fullscreen", True)


app = CVApp(root)

root.mainloop()