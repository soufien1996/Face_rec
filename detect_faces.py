# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import calendar
from datetime import datetime, timedelta
import helpers
import face_recognition
import dlib
import cluster_faces

#faced
from faced import FaceDetector
from faced.utils import annotate_image

font = cv2.FONT_HERSHEY_SIMPLEX
totalInferenceDuration = 0
success = True


def run(mode,localPath):
	global font
	global success
	global totalInferenceDuration
	print("CUDE usage status : "+str(dlib.DLIB_USE_CUDA))
	#faced
	face_detector = FaceDetector()
	startTS = time.time()

	""" Load models """
	predictor_path = "assets/shape_predictor_5_face_landmarks.dat"
	face_rec_model_path = "assets/dlib_face_recognition_resnet_model_v1.dat"
	facerec = dlib.face_recognition_model_v1(face_rec_model_path)
	sp = dlib.shape_predictor(predictor_path)

	""" Check local/stream availability """
	if (mode == "stream"):
		# initialize the video stream and allow the cammera sensor to warmup
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		w = int(vs.get(3))
		h = int(vs.get(4))
		time.sleep(2.0)
	elif (mode == "local"):
		vidcap = cv2.VideoCapture(localPath)
		success,frame = vidcap.read()
		fps = vidcap.get(cv2.CAP_PROP_FPS)
		frameCtr = 0
		w = int(vidcap.get(3))
		h = int(vidcap.get(4))


		
	
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h)) 
    
	while success:
		processStartTs = time.time()

		""" Acquire the next frame """
		if (mode == "stream"):
			frame = vs.read()
			
		elif (mode == "local"):
			success,frame = vidcap.read()
			
			frameCtr += 1

		""" grab the frame from the threaded video stream and resize it
		 to have a maximum width of 400 pixels """
		try:
			frame = imutils.resize(frame, width=400)
		except AttributeError:
			continue
		try:
			rgb_img = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
		except:
			break
		inferenceStartTs = time.time()
		#faced (thresh argument can be added, 0.85 by default-)
		bboxes = face_detector.predict(rgb_img)
		inferenceEndTs = time.time()
		totalInferenceDuration += inferenceEndTs - inferenceStartTs
		
		helpers.min_clusters = len(bboxes)
		if (mode == "stream"):
			timestamp = calendar.timegm(time.gmtime())
		elif (mode == "local"):
			timestamp = float(frameCtr/fps)

		for x, y, w, h, p in bboxes:
			top = int(y + h/2)
			left = int(x - w/2)
			bottom = int(y - h/2)
			right = int(x + w/2)
			cv2.rectangle(frame, (left, bottom), (right, top), (99, 44, 255), 1)
			cv2.putText(frame,str(p),(left,top+5), font, 0.2,(255,255,255),1,cv2.LINE_AA)
			shape = sp(frame,dlib.rectangle(left, bottom, right, top))
			# Compute the 128D vector that describes the face in img identified by
			face_descriptor = facerec.compute_face_descriptor(frame, shape)
			bestIndex = cluster_faces.match(face_descriptor) 
			if (bestIndex >= 0):
				cv2.putText(frame,str(helpers.unique_persons[bestIndex]["uuid"]),(left,top+10), font, 0.2,(0,255,255),1,cv2.LINE_AA)
				data = [{"uuid": helpers.unique_persons[bestIndex]["uuid"],"timestamp": timestamp}]
				helpers.individual_stats.extend(data)
			else :
				cv2.putText(frame,"Learning...",(left,top+10), font, 0.2,(0,255,255),1,cv2.LINE_AA)
				data = [{"label": 0,"timestamp": timestamp,"encoding": face_descriptor}]
				helpers.candidate_persons.extend(data)

		try:
			frame = imutils.resize(frame, width=720)
		except AttributeError:
			continue

		cv2.putText(frame,"FPS : "+str(int(1/(time.time()-processStartTs))),(20,30), font, 1,(0,255,0),3,cv2.LINE_AA,False)
		out.write(frame)
        
		#cv2.imshow("Frame", frame)
		if(len(helpers.candidate_persons) >= (helpers.MIN_FACES_PER_CLUSTER* helpers.min_clusters)):
			cluster_faces.cluster()
		key = cv2.waitKey(1) & 0xFF
	 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	# do a bit of cleanup
	if (mode == "stream"):
		vs.stop()
	endTS = time.time()
	out.release()
	print("Total number of unique faces = ",len(helpers.unique_persons))
	print("Total duration")
	print(endTS - startTS)
	print("Total inference duration")
	print(totalInferenceDuration)
