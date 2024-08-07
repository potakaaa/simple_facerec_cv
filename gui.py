import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from simple_facerec import SimpleFacerec

# Initialize face recognition and load images
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

class CVApp:
    def __init__(self, window):
        self.window = window
        self.window.title("CS3 Lakip")

        self.video_capture = cv2.VideoCapture(0)
        self.current_image = None
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.updateWebcam()

    def updateWebcam(self):
    
        ret, frame = self.video_capture.read()
        if ret:
            face_loc, face_name = sfr.detect_known_faces(frame)
            for loc, name in zip(face_loc, face_name):
                y1, x1, y2, x2 = loc[0], loc[3], loc[1], loc[2]
                cv2.putText(frame, name, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

            self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            self.photo = ImageTk.PhotoImage(image=self.current_image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            

            key = cv2.waitKey(1)
            self.window.after(1, self.updateWebcam)

root = tk.Tk()

app = CVApp(root)

root.mainloop()