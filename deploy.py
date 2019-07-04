import detect_faces
import cluster_faces
import helpers
import argparse

helpers.init()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode",type=str, required=True,
	help="mode (stream or local)")
ap.add_argument("-l", "--local_path",type=str,default="",
	help="path to local video")
args = vars(ap.parse_args())

mode = args["mode"]
localPath = args["local_path"]

detect_faces.run(mode,localPath)
#cluster_faces.run()