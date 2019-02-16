import cv2, os
import numpy as np
import sys
from utils import moving_average, plot
import queue
from sklearn import linear_model


class Solver():
	def __init__(self, config):
		self.vid = cv2.VideoCapture(config.vidpath)
		self.txtfile = config.txtfile
		self.vis = config.vis
		self.len_gt = config.len_gt
		self.test_vid = cv2.VideoCapture(config.test_vidpath)
		# Separate function to allow for different methods to be inculcated into the same class
		self.setupParams()

	def setupParams(self):
		""" intialize parameters for tracking and extracting features
		Load ground truth parameters from txt file"""
		# Lucas Kanade parameters
		self.lk_params = dict(winSize = (21, 21),
							  maxLevel = 2,
							  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

		self.frame_idx = 0
		self.prev_pts = None
		self.detect_interval = 1
		self.temp_preds = np.zeros(int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)))

		# Construct data table for history of images
		with open(self.txtfile, 'r') as file_:
			gt = file_.readlines()
			gt = [float(x.strip()) for x in gt]
		
		self.gt = np.array(gt[:self.len_gt])

		self.median = []
		self.window = 80 # for moving average
		self.prev_gray = None

	def constructMask(self, mask = None, test=False):
		"""Constructs a mask to only take into consideration the road """
		vid = self.test_vid if test else self.vid
		if mask is None:
			W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
			H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
			mask = np.zeros(shape = (H,W), dtype = np.uint8)
			mask.fill(255)
		else:
			W = mask.shape[1]
			H = mask.shape[0]

		cv2.rectangle(mask, (0, 0), (W, H), (0, 0, 0), -1)

		x_top_offset = 180
		x_btm_offset = 35

		poly_pts = np.array([[[640-x_top_offset, 250], [x_top_offset, 250], [x_btm_offset, 350], [640-x_btm_offset, 350]]], dtype=np.int32)
		cv2.fillPoly(mask, poly_pts, (255, 255, 255))

		return mask


	def processFrame(self, frame):
		""" Gaussian Blur and then apply Lucas Kanade"""
		frame = cv2.GaussianBlur(frame, (3,3), 0)

		curr_pts, _st, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame, self.prev_pts, None, **self.lk_params)
		# Store flow (x, y, dx, dy)
		flow = np.hstack((self.prev_pts.reshape(-1, 2), (curr_pts - self.prev_pts).reshape(-1, 2)))

		preds = []
		for x, y, u, v in flow:
			if v < -0.05:
				continue
			# Translate points to center
			x -= frame.shape[1]/2
			y -= frame.shape[0]/2

			# Append to preds taking care of stability issues
			if y == 0 or (abs(u) - abs(v)) > 11:
				preds.append(0)
				preds.append(0)
			elif x == 0:
				preds.append(0)
				preds.append(v / (y*y))
			else:
				preds.append(u / (x * y))
				preds.append(v / (y*y))

		return [n for n in preds if n>=0]

	def getKeyPts(self, offset_x=0, offset_y=0):
		""" return key points with offset """
		if self.prev_pts is None:
		  return None
		return [cv2.KeyPoint(x=p[0][0] + offset_x, y=p[0][1] + offset_y, _size=10) for p in self.prev_pts]
	
	def getFeatures(self, frame_gray, mask):
		return cv2.goodFeaturesToTrack(frame_gray,30,0.1,10,blockSize=10,
													mask=mask)

	def run(self):

		# Construct mask first
		mask = self.constructMask()
		prev_key_pts = None

		while self.vid.isOpened() and self.frame_idx<len(self.gt):
			ret, frame = self.vid.read()
			if not ret:
				break

			# Convert to B/W
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame_gray = frame_gray[130:350, 35:605]
			# mask_vis = frame.copy() # <- For visualization
			
			# Process each frame
			if self.prev_pts is None:
				self.temp_preds[self.frame_idx] = 0
			else:
				# Get median of predicted V/hf values
				preds = self.processFrame(frame_gray)
				self.temp_preds[self.frame_idx] = np.median(preds) if len(preds) else 0

			# Extract features
			self.prev_pts = self.getFeatures(frame_gray, mask[130:350, 35:605])
			self.prev_gray = frame_gray
			self.frame_idx += 1
			
			# For visualization purposes only
			if self.vis:
				prev_key_pts = self.visualize(frame, mask_vis, prev_key_pts)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break


		self.vid.release()
		cv2.destroyAllWindows()


		# Split predictions into train and validation - 
		# split = self.frame_idx//10
		# train_preds = self.temp_preds[:self.frame_idx-split]
		# val_preds = self.temp_preds[self.frame_idx - split:self.frame_idx]
		# gt_train = self.gt[:len(train_preds)]
		# gt_val = self.gt[len(train_preds):self.frame_idx]

		# Fit to ground truth
		# preds = moving_average(train_preds, self.window)
		preds = moving_average(self.temp_preds, self.window)

		reg = linear_model.LinearRegression(fit_intercept=False)
		reg.fit(preds.reshape(-1, 1), self.gt) 
		hf_factor = reg.coef_[0]
		print("Estimated hf factor = {}".format(hf_factor))


		preds = self.temp_preds * hf_factor
		preds = moving_average(preds, self.window)
		mse = np.mean((preds - self.gt)**2)
		print("MSE for train", mse)
		

		# estimate training error
		# pred_speed_train = train_preds * hf_factor
		# pred_speed_train = moving_average(pred_speed_train, self.window)
		# mse = np.mean((pred_speed_train - gt_train)**2)

		# Estiamte validation error
		# pred_speed_val = val_preds * hf_factor
		# pred_speed_val = moving_average(pred_speed_val, self.window)
		# mse = np.mean((pred_speed_val - gt_val)**2)
		# print("MSE for val", mse)
		
		# plot(pred_speed_val, gt_val)

		return hf_factor


	def visualize(self, frame, mask_vis, prev_key_pts, speed=None):
		self.constructMask(mask_vis)

		
		mask_vis = cv2.bitwise_not(mask_vis)
		frame_vis = cv2.addWeighted(frame, 1, mask_vis, 0.3, 0)
		key_pts = self.getKeyPts(35, 130)
		cv2.drawKeypoints(frame_vis, key_pts, frame_vis, color=(0,0,255))
		cv2.drawKeypoints(frame_vis, prev_key_pts, frame_vis, color=(0,255,0))
		if speed is not None:
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame_vis, "speed {}".format(speed), (10, 35), font, 1.2, (0, 0, 255))
		
		cv2.imshow('test',frame_vis)
		return key_pts

	def test(self, hf_factor):
		mask = self.constructMask(test=True)
		
		self.prev_gray = None
		test_preds = np.zeros(int(self.test_vid.get(cv2.CAP_PROP_FRAME_COUNT)))
		frame_idx = 0
		curr_estimate = 0
		prev_key_pts = None
		q = queue.Queue(maxsize=self.window//2)
		q.put(0)
		while self.test_vid.isOpened() and frame_idx<len(self.gt):
			ret, frame = self.test_vid.read()
			if not ret:
				break

			# Convert to B/W
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			frame_gray = frame_gray[130:350, 35:605]
			mask_vis = frame.copy() # <- For visualization
			
			# Process each frame
			# For the first frame
			pred_speed = 0
			if self.prev_pts is None:
				test_preds[frame_idx] = 0
			else:
				# Get median of predicted V/hf values
				preds = self.processFrame(frame_gray)
				pred_speed = np.median(preds) * hf_factor if len(preds) else 0
				
				# Moving average
				if q.full():
					temp = q.get()
				q.put(pred_speed)
				pred_speed = np.mean(np.array(list(q.queue)))
				test_preds[frame_idx] =  pred_speed # m/s -> mph

			# Extract features
			# print(frame_idx, frame_gray.shape, mask.shape)
			self.prev_pts = self.getFeatures(frame_gray, mask[130:350, 35:605])
			self.prev_gray = frame_gray
			frame_idx += 1
			
			# For visualization purposes only
			if self.vis:
				prev_key_pts = self.visualize(frame, mask_vis, prev_key_pts, speed=pred_speed)
				if cv2.waitKey(25) & 0xFF == ord('q'):
					break
		
		self.test_vid.release()
		cv2.destroyAllWindows()
		
		with open("test.txt", "w") as file_:
			for item in test_preds:
				file_.write("%s \n" % item)





