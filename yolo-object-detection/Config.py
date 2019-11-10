import logging
import numpy as np
import time


class Config:
	# FOR DINO GAME SESSION
	CHROME_PATH = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s"
	GAME_URL = 'chrome://dino/'

	LABELS_FILE = r"..\chrome_dino\data\obj.names"

	YOLO_CONFIG_FILE = r"..\chrome_dino\cfg\yolov3-tiny_obj_4c.cfg"
	WEIGHTS_FILE = r"..\chrome_dino\backup\yolov3-tiny_obj_4c_15500.weights"

	CONFIDENCE_VALUE = 0.8
	THRESHOLD_VALUE = 0.3
	LABELS = open(LABELS_FILE).read().strip().split("\n")
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

	# FOR GENE
	N_X = 5
	N_H = 5
	N_Y = 1
	POP_SIZE = 12
	MUTATION_PROB = 0.1
	N_SIZE = 4
	RANDOM_SET = [6, 5, 4, 3]
	BODY_KEYS = ["W1", "W2", "b1", "b2"]
	MUTATION_RANGE = [0.005, 0.5, 0.2, 0.05]

	log_brief = None
	log_survival = None

	@staticmethod
	def init():
		Config.log_brief = Config.setup_logger('log1', time.strftime("./logs/%Y-%m-%d_%H_%M_%S_brief.log"))
		Config.log_survival = Config.setup_logger('log2', time.strftime("./logs/%Y-%m-%d_%H_%M_%S_survival.log"))

	@staticmethod
	def setup_logger(logger_name, log_file, level=logging.INFO):
		l = logging.getLogger(logger_name)
		formatter = logging.Formatter('%(asctime)s : %(message)s')
		file_handler = logging.FileHandler(log_file, mode='w')
		file_handler.setFormatter(formatter)
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(formatter)

		l.setLevel(level)
		l.addHandler(file_handler)
		l.addHandler(stream_handler)

		logging.basicConfig(filename=log_file, level=level,format="%(asctime)s %(levelname)s %(message)s")
		return l
