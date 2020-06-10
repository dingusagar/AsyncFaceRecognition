# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
from enhanced_recognition import ImageEnhancement
import threading
from multiprocessing import Queue
import numpy as np
import datetime
import config
from attendance import AttendanceMarker

# constants 
FRAME_WINDOW = 'Frame'
RECOGNITION_WINDOW = 'Recognition'
TEXT_WINDOW = 'Help'
BG_THREAD_NAME = 'bg_thread'

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)


que =Queue()


# initialise attendance_marker
attendance_marker = AttendanceMarker()




# naming and placing windows
cv2.namedWindow(TEXT_WINDOW)
cv2.moveWindow(TEXT_WINDOW,320,700)
cv2.namedWindow(FRAME_WINDOW)
cv2.moveWindow(FRAME_WINDOW,320,135)
cv2.namedWindow(RECOGNITION_WINDOW)
cv2.moveWindow(RECOGNITION_WINDOW, 1000, 160) 


# enhancement 
enhancement = ImageEnhancement()

image_to_process = None
processed_frame = np.zeros((224,224,3))
recognized_faces = []

def show_text_window(log = None):
	frame = np.zeros((250,1000,3),dtype=np.uint8)
	text = "Press r to recognise faces \nPress s to save the attendance \nPress q to quit"
	y0, dy = 20, 20
	for i, line in enumerate(text.split('\n')):
		y = y0 + i*dy
		cv2.putText(frame, line.strip(), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (137, 243, 111), 1)

	if log:
		log = '[ logs ]\n \n' + log
		y0 = y + dy*3
		for i, line in enumerate(log.split('\n')):
			y = y0 + i*dy
			cv2.putText(frame, line.strip(), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (101, 243, 224), 1)
	cv2.imshow(TEXT_WINDOW,frame)

show_text_window()


def log(text):
	print('[INFO] ' + text)
	show_text_window(log=text)

def post_process_frame(frame):

    # Uncomment if you want to improve the resolution of the frame based on super resolution models. Will increase the latency
	# frame = enhancement.improve_quality(frame,type='gans')
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])


	encodings = face_recognition.face_encodings(rgb, boxes)

	names = []
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"


		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# rescale the face coordinates
		# top = int(top * r)
		# right = int(right * r)
		# bottom = int(bottom * r)
		# left = int(left * r)

		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	

	print('job completed!')
	return frame , names

bg_thread = None


# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	original_frame = frame.copy()
	image_to_process = original_frame
	
	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=480)
	r = frame.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	
	# loop over the recognized faces
	for (top, right, bottom, left) in boxes:
		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, 'face', (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)


	if bg_thread and  not bg_thread.isAlive():
		if que:
			processed_frame, recognized_faces =que.get()
		bg_thread = None
	
	
	
	# if the video writer is None *AND* we are supposed to write
	# the output video to disk initialize the writer
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)

	# if the writer is not None, write the frame with recognized
	# faces t odisk
	if writer is not None:
		writer.write(frame)

	# check to see if we are supposed to display the output frame to
	# the scree

    # check to see if we are supposed to display the output frame to
    # the screen
	if args["display"] > 0:
		cv2.imshow(FRAME_WINDOW, frame)
		cv2.imshow(RECOGNITION_WINDOW,processed_frame)


		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
		elif key == ord("r"):
			if bg_thread == None :
				bg_thread = threading.Thread(target=lambda q, arg1: q.put(post_process_frame(arg1)),name=BG_THREAD_NAME, args=(que, rgb))
				bg_thread.start()
				log('Recognising faces...')

			else:
				log('Recognition process already active..please wait.')

		elif key == ord('s'):
			attendance_marker.mark_attendance(recognized_faces)
			log('Marking attendance for {}'.format(recognized_faces))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()