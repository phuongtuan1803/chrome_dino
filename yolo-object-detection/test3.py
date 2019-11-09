from PIL import ImageGrab
import numpy as np
import argparse
import time
import cv2
import mss
import mss.tools
from PIL import Image
from ScreenViewer import ScreenViewer

def _get_box(detection):
	# scale the bounding box coordinates back relative to
	# the size of the image, keeping in mind that YOLO
	# actually returns the center (x, y)-coordinates of
	# the bounding box followed by the boxes' width and
	# height
	box = detection[0:4] * np.array([W, H, W, H])
	(centerX, centerY, width, height) = box.astype("int")

	# use the center (x, y)-coordinates to derive the top
	# and and left corner of the bounding box
	x = int(centerX - (width / 2))
	y = int(centerY - (height / 2))

	# update our list of bounding box coordinates,
	# confidences, and class IDs
	return [x, y, int(width), int(height)]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default=r"videos/dino_2.webm",
	help="path to input video")
ap.add_argument("-o", "--output", default=r"output/dino_2.avi",
	help="path to output video")
ap.add_argument("-cf", "--config", default=r"F:\Git\chrome_dino\chrome_dino\cfg\yolov3-tiny_obj_4c.cfg",
	help="base path to *.cfg")
ap.add_argument("-w", "--weights", default=r"F:\Git\chrome_dino\chrome_dino\yolov3-tiny_obj_4c_10000.weights",
	help="base path to *.weights")
ap.add_argument("-n", "--names", default=r"F:\Git\chrome_dino\chrome_dino\data\obj.names",
	help="base path to *.names")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

np.random.seed(42)

labelsPath = args["names"]

LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(args["config"], args["weights"])
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None
(W, H) = (None, None)
min_x_prev = 0
sv = ScreenViewer()
sv.GetHWND('chrome://dino/ - Google Chrome')
sv.Start()

while True:
	start = time.time()
	# monitor = {"top": 150, "left": 100, "width": 1820, "height": 400}
	# sct_img = mss.mss().grab(monitor)
	# img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

	img = sv.GetScreen()

	frame = np.array(img)

	if W is None or H is None:
		(H, W) = frame.shape[:2]
	stop_0 = time.time()

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=True)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	dinosaur = None
	object = list()
	is_gameover = False
	min_x = 0
	stop_0 = time.time()
	detections = [detection for output in layerOutputs for detection in output]
	scores = [detection[5:] for detection in detections]
	classIDlist = [np.argmax(score) for score in scores]
	confidencelist = [score[classID] for (score, classID) in zip(scores,classIDlist)]

	select_items = [(detection, score, classID, confidence)
						for (detection, score, classID, confidence) in zip(detections, scores, classIDlist, confidencelist)
						if confidence > args["confidence"]]
	stop_1 = time.time()

	for (detection, score, classID, confidence) in select_items:
		if classID == 0:
			dinosaur = _get_box(detection)
		elif classID == 1 or classID == 2:
			object.append((classID, _get_box(detection)))
		elif classID == 3:
			is_gameover = True
		boxes.append(_get_box(detection))
		confidences.append(float(confidence))
		classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# Cal Weight
	speed = 0.0
	distance = 0
	obj_type = -1
	obj_height = 0
	obj_width = 0

	# get neanest object
	if not(len(object) == 0 or dinosaur is None):
		front_obj = [obj[1][0] for obj in object if obj[1][0] > dinosaur[0]]
		if len(front_obj) > 0:
			min_x = min(front_obj)
			speed = max(0,min_x_prev - min_x)
			min_x_prev = min_x

			for obj in object:
				if obj[1][0] == min_x:
					obj_type = obj[0]
					obj_height, obj_width = obj[1][2], obj[1][3]
					distance = max(min_x - dinosaur[0],0)
			cv2.arrowedLine(frame, (dinosaur[0], dinosaur[1]), (dinosaur[0] + distance, dinosaur[1]), (0, 0, 255), 1)
		else:
			speed = 0

	if obj_type == -1:
		type_str = "None"
	else:
		type_str = LABELS[obj_type]
	text = f'speed: {speed: 10.2f}.  type: {type_str}.  distance: {distance:10d}.  obj_height: {obj_height:10d}.  obj_width: {obj_width:10d}'
	cv2.putText(frame, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0) , 2)

	end = time.time()
	fps_value = 1 / (end -start)
	per1 = (stop_0- start)/ (end -start) *100
	per2 = (stop_1- start)/ (end -start) *100
	cv2.putText(frame, f'fps: {fps_value: 3.2f}. Per1: {per1:.0f}%. Per2: {per2:.0f}%', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

	cv2.imshow("frame", frame)
	key = cv2.waitKey(1)
	if key == 27:
		break
	if writer is None:
		writer = cv2.VideoWriter(args["output"], fourcc, 30,(frame.shape[1], frame.shape[0]), True)
	writer.write(frame)

sv.Stop()
writer.release()

