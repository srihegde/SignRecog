#################################################################################
# 
# Segmenting and Tracking Skin Pixels
# Author: Srinidhi Hegde
# 
# Changes made for openCV 2.4.13
# 1. In func getDenseOptFlow, calcOpticalFlowFarneback changed to
#             calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)
# 2. In func getDenseOptFlow, if-break statement checking end of video removed
#################################################################################

from os import walk
import cv2
import numpy as np

# Color Filter: To segment an image to obtain skin pixels
def segment(img):
	_,mask = cv2.threshold(img,160, 200,cv2.THRESH_BINARY)
	segimg = cv2.bitwise_and(img, mask)
	return segimg

def getFiles():
	for (_, _, filenames) in walk('../resources/videos/'):
		break
	return filenames


def getSparseOptFlow():
	# params for ShiTomasi features
	feature_params = dict( maxCorners = 100,
							qualityLevel = 0.1,
							minDistance = 5,
							blockSize = 5 )

	# parameters for LK optical flow
	lk_params = dict( winSize  = (15,15),
				maxLevel = 2,
				criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	filenames = getFiles()
	for f in filenames:
		print 'Reading file: ' + f
		cap = cv2.VideoCapture('../resources/videos/'+f)
		
		# Take first frame and find corners in it
		ret, old_frame = cap.read()
		pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
		old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
		old_frame = old_frame[0:250, 0:336]
		old_frame = segment(old_frame)
		# print old_frame.shape
		
		p0 = cv2.goodFeaturesToTrack(old_frame, mask = None, **feature_params)

		# Create some random colors
		color = np.random.randint(0,255,(100,3))
		startPts = p0
		

		while cap.isOpened():
			ret, frame = cap.read()

			mask = np.zeros_like(old_frame)

			if ret:
				# To crop relevant portion of image
				frame = frame[0:250, 0:336]
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				segf = segment(frame)

				
				# If there are less feature points in the frame
				if len(p0) < 5:
					p0 = cv2.goodFeaturesToTrack(segf, mask = None, **feature_params)
					startPts = p0

				# calculate optical flow
				p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, segf, p0, None, **lk_params)

				# Select good points
				good_new = p1[st==1]
				good_old = startPts[st == 1]

				cumSlope = 0;

				# draw the tracks
				for i,(new,old) in enumerate(zip(good_new,good_old)):
					a,b = new.ravel()
					c,d = old.ravel()

					mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
					frame = cv2.circle(segf,(a,b),5,color[i].tolist(),-1)

				segf = cv2.add(frame,mask)
				
				if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
					break
				

				cv2.imshow('Video',segf)
				k = cv2.waitKey(30) & 0xff
				if k == 27:
					break
			else:
				break

		cv2.destroyAllWindows()
		cap.release()


def getDenseOptFlow():
	filenames = getFiles()
	ofAllVideos = []
	for f in filenames:
		print 'Reading file: ' + f
		cap = cv2.VideoCapture('../resources/videos/'+f)
		ofFile = []

		ret, frame1 = cap.read()
		frame1 = frame1[0:250, 0:336]
		frame1 = segment(frame1)
		prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
		hsv = np.zeros_like(frame1)
		hsv[...,1] = 255

		while cap.isOpened():
			ret, frame2 = cap.read()
			if ret:
				frame2 = frame2[0:250, 0:336]
				frame2 = segment(frame2)
				
				next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

				flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)
				ofFile.append(flow)

				mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
				hsv[...,0] = ang*180/np.pi/2
				hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
				bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

				#if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
				#		break

				cv2.imshow('frame2',bgr)
				k = cv2.waitKey(30) & 0xff
				if k == 27:
					break

				# If saving of frame is reqd enable the following
				# elif k == ord('s'):
				# 	cv2.imwrite('opticalfb.png',frame2)
				# 	cv2.imwrite('opticalhsv.png',bgr)
				prvs = next
			else:
				break

		cap.release()
		cv2.destroyAllWindows()
		ofAllVideos.append(ofFile)

	return ofAllVideos


ofvectors = getDenseOptFlow()
# print ofvectors[0][0]
