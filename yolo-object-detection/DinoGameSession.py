from PIL import ImageGrab
import numpy as np
import webbrowser
import time
import cv2
import mss
import mss.tools
from PIL import Image
import pyautogui
import trex_nn
from Config import Config
from selenium.webdriver import Chrome


class DinoGameSession:

	def __init__(self):
		self.score = 0

	def open_url_on_chrome(url):
		webbrowser.get(Config.CHROME_PATH).open(Config.GAME_URL,new=2)

	@staticmethod
	def _start_game_():
		pyautogui.press('space')
		time.sleep(0.2)

	@staticmethod
	def press_up():
		pyautogui.press("up")

	@staticmethod
	def press_down():
		pyautogui.press("down")

	@staticmethod
	def _get_box_(detection , H, W):
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

	def play(self, parameters_set, infor_str = ''):
		# self.open_url_on_chrome()
		self.score = 0

		np.random.seed(42)
		net = cv2.dnn.readNetFromDarknet(Config.YOLO_CONFIG_FILE, Config.WEIGHTS_FILE)
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		(W, H) = (None, None)
		min_x_prev = 0
		speed_prev = 0
		is_game_over = False
		is_game_started = False

		while True:

			start = time.time()
			with mss.mss() as sct:
				monitor = {"top": 150, "left": 100, "width": 1820, "height": 400}
				sct_img = sct.grab(monitor)

			img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
			frame = np.array(img)

			if W is None or H is None:
				(H, W) = frame.shape[:2]
			stop_0 = time.time()

			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)
			stop_1 = time.time()

			# initialize our lists of detected bounding boxes, confidences,
			# and class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []

			dinosaur = None
			game_over_sign = None
			object = list()
			detections = [detection for output in layerOutputs for detection in output]
			scores = [detection[5:] for detection in detections]
			classIDlist = [np.argmax(score) for score in scores]
			confidencelist = [score[classID] for (score, classID) in zip(scores,classIDlist)]

			select_items = [(detection, score, classID, confidence)
								for (detection, score, classID, confidence) in zip(detections, scores, classIDlist, confidencelist)
								if confidence > Config.CONFIDENCE_VALUE]

			stop_2 = time.time()

			for (detection, score, classID, confidence) in select_items:
				if classID == 0:
					dinosaur = DinoGameSession._get_box_(detection, H, W)
				elif classID == 1 or classID == 2:
					object.append((classID, DinoGameSession._get_box_(detection, H, W)))
				elif classID == 3:
					game_over_sign = DinoGameSession._get_box_(detection, H, W)
				boxes.append(DinoGameSession._get_box_(detection, H, W))
				confidences.append(float(confidence))
				classIDs.append(classID)

			# apply non-maxima suppression to suppress weak, overlapping
			# bounding boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, Config.CONFIDENCE_VALUE, Config.THRESHOLD_VALUE)

			# ensure at least one detection exists
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					# draw a bounding box rectangle and label on the frame
					color = [int(c) for c in Config.COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(Config.LABELS[classIDs[i]], confidences[i])
					cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			# =======================================================================================
			# CHECK GAME IS IN SCREEN AND IS GAME STARTED
			if dinosaur is None and game_over_sign is None:
				# print("Game screen is not loaded yet")
				cv2.imshow("frame", frame)
				key = cv2.waitKey(1)
				if key == 27:
					break
				continue

			if not is_game_started:
				DinoGameSession._start_game_()
				if game_over_sign is None:
					# print("Game Session is started")
					is_game_started = True
					start_time = time.time()
				else:
					# print("Game Session is not started yet")
					cv2.imshow("frame", frame)
					key = cv2.waitKey(1)
					if key == 27 or is_game_over:
						break
					continue
			if is_game_started and game_over_sign is not None:
				is_game_over = True

			# Cal Weight
			speed = 0.0
			distance = 0
			obj_type = -1
			obj_height = 0
			obj_width = 0

			# get nearest object
			if not(len(object) == 0 or dinosaur is None):
				front_obj = [obj[1][0] for obj in object if obj[1][0] > dinosaur[0]]
				if len(front_obj) > 0:
					min_x = min(front_obj)
					speed = (max(0,min_x_prev - min_x) + speed_prev*2)/3
					if min_x > min_x_prev:
						self.score+=1
					min_x_prev = min_x
					speed_prev = speed

					obj_nearest = [obj for obj in object if obj[1][0] == min_x][0]
					obj_type = obj_nearest[0]
					obj_height, obj_width = obj_nearest[1][2], obj_nearest[1][3]
					distance = max(min_x - (dinosaur[0] + dinosaur[2]),0)

					cv2.arrowedLine(frame, (dinosaur[0] + dinosaur[2], obj_nearest[1][1]), (dinosaur[0] + dinosaur[2] + distance, obj_nearest[1][1]), (0, 0, 255), 6,cv2.LINE_AA,0,0.01)
				else:
					speed = 0

			# input_set = [speed, obj_type, distance, obj_height, obj_width]
			input_set = [distance, speed, obj_width]
			trex_nn.wrap_model(input_set, parameters_set, Config.N_X)

			text = f'speed: {speed: 10.2f}.  type: {obj_type}.  distance: {distance:10d}.  obj_height: {obj_height:10d}.  obj_width: {obj_width:10d}'
			cv2.putText(frame, text, (50, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0) , 2)

			end = time.time()
			fps_value = 1 / (end -start)
			cv2.putText(frame, f'fps: {fps_value: 3.2f}. Score: {self.score}. {infor_str}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

			time_capture = (stop_0- start)/ (end -start) *100
			time_yolo = (stop_1- stop_0)/ (end -start) *100
			time_check_image = (stop_2 - stop_1) / (end - start) * 100
			cv2.putText(frame, f'Time Screen Capture: {time_capture:.0f}%. Time YOLO: {time_yolo:.0f}%. Time check image: {time_check_image:.0f}%', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			cv2.imshow("frame", frame)
			key = cv2.waitKey(1)
			if key == 27 or is_game_over:
				break


if __name__ == '__main__':
	parameters_set = None
	obj = DinoGameSession()
	obj.play(parameters_set)