from os import walk
import cv2

# Color Filter: To segment an image to obtain skin pixels
def segment(img):
	_,mask = cv2.threshold(img,160, 200,cv2.THRESH_BINARY)
	segimg = cv2.bitwise_and(img, mask)
	return segimg


for (_, _, filenames) in walk('../resources/videos/'):
	break

for f in filenames:
	print 'Reading file: ' + f
	cap = cv2.VideoCapture('../resources/videos/'+f)
	
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			# To crop relevant portion of image
			frame = frame[0:250, 0:336]
			segf = segment(frame)

			cv2.imshow('Video',segf)
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
		else:
			break

	cv2.destroyAllWindows()
	cap.release()