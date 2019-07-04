from math import sqrt

def init():
	global mode
	global localPath
	mode = ""
	localPath = ""


	global groupStat
	global individual_stats
	global candidate_persons
	global unique_persons
	unique_persons = []
	candidate_persons = []
	groupStat = {
		"maxNbSimVisitors":0,
		"minNbSimVisitors":1000,
		"maxVisitDuration":0,
		"minVisitDuration":0,
		"meanVisitDuration":0,
		"returnRatio":0,
		"peakHour":0,
		"ageAverage":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
		"genderAverage":[0.0,0.0],
		"nbVisitors":0
	}
	individual_stats = []

	global min_clusters
	global MIN_FACES_PER_CLUSTER
	global MAX_MATCHING_THRESH
	min_clusters = 0
	MIN_FACES_PER_CLUSTER = 30
	MAX_MATCHING_THRESH = 0.35

def euclidean_dist(vector_x, vector_y):
    if len(vector_x) != len(vector_y):
        raise Exception('Vectors must be same dimensions')
    return sqrt(sum((vector_x[dim] - vector_y[dim]) ** 2 for dim in range(len(vector_x))))