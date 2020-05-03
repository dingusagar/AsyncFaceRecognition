import cv2
import imutils


cap = cv2.VideoCapture('http://192.168.43.1:8080/video')

while(True):
    ret, frame = cap.read()
    frame = imutils.resize(frame,width=1080)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break