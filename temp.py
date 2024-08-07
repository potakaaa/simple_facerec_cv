import cv2
import face_recognition
from facerec_functions import FaceRecognition

def main():
    
    frs = FaceRecognition()

    known_faces_dir = "images/"  
    known_face_encodings, known_face_names = frs.load_known_faces(known_faces_dir)

    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    cv2.namedWindow('Face Recognition', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    screen_width, screen_height = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if screen_width == 0 and screen_height == 0:
            screen_width = cv2.getWindowImageRect('Face Recognition')[2]
            screen_height = cv2.getWindowImageRect('Face Recognition')[3]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])

            if face_encoding:
                # Compare with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])
                name = "Stranger"

                if True in matches:
                    match_index = matches.index(True)
                    name = known_face_names[match_index]

            frs.draw_rectangle(frame, x, y, w, h, name)

        frame = frs.maintain_aspect_ratio(frame, screen_width, screen_height)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
