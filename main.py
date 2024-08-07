import cv2
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    face_loc, face_name = sfr.detect_known_faces(frame)
    for loc, name in zip(face_loc, face_name):
        y1, x1, y2, x2 = loc[0], loc[3], loc[1], loc[2]

        cv2.putText(frame, name, (x1 - 5, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        break

cam.release()
cv2.destroyAllWindows()