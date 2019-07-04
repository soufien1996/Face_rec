# USAGE
# python cluster_faces.py --encodings encodings.pickle

# import the necessary packages
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2
import helpers
import dlib
import time
from math import sqrt
import uuid


def cluster():
	labelIdx = 0
	encodings = [d["encoding"] for d in helpers.candidate_persons]
	labels = dlib.chinese_whispers_clustering(encodings, 0.5)
	num_classes = len(set(labels))
	for label in labels:
		helpers.candidate_persons[labelIdx]["label"] = int(label)
		labelIdx += 1 
	
	for label in range(num_classes):
		face_encs = [fe for fe in helpers.candidate_persons if fe["label"] == label]
		if (len(face_encs) >= helpers.MIN_FACES_PER_CLUSTER):
			mean_enc = np.zeros(128)
			for fe in face_encs:
				mean_enc += fe["encoding"]
			mean_enc = mean_enc / len(face_encs)
			helpers.unique_persons.append({"uuid": uuid.uuid1(), "Mean": mean_enc})
	uuids = [d["uuid"] for d in helpers.unique_persons]
	helpers.candidate_persons = []		

	#print ("Number of unique faces : {} {} {}".format(num_classes-len(blacklist),faceInstances,blacklist), end="\r", flush=True)
def match(candidate):

	bestThresh = 9999
	bestIndex = -1
	if(len(helpers.unique_persons) > 0):
		for index,person in enumerate(helpers.unique_persons) :
			currThresh = helpers.euclidean_dist(candidate,dlib.vector(person["Mean"])) 
			if (currThresh < helpers.MAX_MATCHING_THRESH):
				if (currThresh < bestThresh):
					bestIndex = index
					bestThresh = currThresh
		
	return bestIndex
