from sliding_window import pyramid, sliding_window
from sklearn.externals import joblib
import argparse
import time
import cv2
import numpy as np

# HOG parametrization
winSize = (32,32)
blockSize = (16,16)
blockStride = (4,4)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
useSignedGradients = True

# Define HOG descriptor 
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,
	cellSize,nbins,derivAperture,winSigma,histogramNormType
	,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)

# Load pickle model
clf = joblib.load('ship_hog_svm_clf.pkl')

# Define image and Window size
image = cv2.imread('scenes/lb_1.png')
(winW, winH) = (64, 64)

# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
		

		# since we do not have a classifier, we'll just draw the window
		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		
		cropped_img = resized[y:y + winH, x:x + winW]
		cropped_img_resized = cv2.resize(cropped_img, winSize)
		descriptor = np.transpose(hog.compute(cropped_img_resized))
		prob = clf.predict_proba(descriptor)
		y_pred = np.argmax(prob[0])
		
		if y_pred == 1:
			print 'Ship found!'
			time.sleep(3)
		
		cv2.imshow("Window", clone)
		cv2.imshow("Cropped", cropped_img)
		cv2.waitKey(1)
		time.sleep(0.025)
