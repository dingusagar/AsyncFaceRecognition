

'''

This program is a demo to show that you can use the android smartphone camera and stream it via http the python program running 
on the computer for processing. 

Step 1 : Install IP Camera App in android.  Play Store Link : https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en_IN
Step 2 : Make sure both the android device and pc are on the same network. (Connect to same wifi hotspot)
Step 3 : Find the Ip of android device in the network. ( you can use nmap tool for this or use Network Utilities App on playstore)
Step 4 : modify the IP on VideoSteam object in the code..
Step 5 : Start streaming from IP Camera
Step 6 : Start executing this script
'''


import time
import cv2
import imutils

from imutils.video import VideoStream

# update the ip address here to the ip address of the device streaming the camera input.
vs = VideoStream(src='http://192.168.43.1:8080/video').start()
frame_num = 0
avg_fps = 0

while True:
    # grab the frame from the threaded video stream
    start_time = time.time()
    frame = vs.read()
    frame = imutils.resize(frame,width=1080)
    cv2.imshow('frame',frame)

    key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    end_time = time.time()
    current_fps = 1 / (end_time - start_time)
    avg_fps = (avg_fps * frame_num + current_fps) / (frame_num + 1)
    frame_num += 1

    print('FPS = {:.2f}'.format(avg_fps))

cv2.destroyAllWindows()
vs.stop()