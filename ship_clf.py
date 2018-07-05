import os
import zipfile
import numpy as np 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.externals import joblib
import cv2

# Decompress dataset and testing images
zip_ref = zipfile.ZipFile("dataset.zip", 'r')
zip_ref.extractall()
zip_ref.close()

# Load dataset images
path = 'ships_dataset'
img_files = [(os.path.join(root, name))
    for root, dirs, files in os.walk(path)
    for name in files if name.endswith((".jpg"))]

# HOG parametrization
winSize = (32,32)
blockSize = (16,16)
blockStride = (8,8)
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

# Retrieve information and image patches from XML
features = np.zeros((1,324),np.float32)
labels = np.zeros(1,np.int64)
for i in img_files:
	# Read images
	img = cv2.imread(i)
	# Resize to winSize
	resized_img = cv2.resize(img, winSize)
	# Compute HOG descriptor ans stack them as features
	descriptor = np.transpose(hog.compute(resized_img))
	#print descriptor.shape
	features = np.vstack((features, descriptor))
	# Stack the labels
	labels = np.vstack((labels, int(i[-5])))

features = np.delete(features, (0), axis=0)
labels = np.delete(labels, (0), axis=0).ravel()

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(features, 
													labels, 
													test_size=0.2, 
													random_state=42)
print 'X_train: ', X_train.shape, 'y_train', y_train.shape
print 'X_test: ', X_test.shape, 'X_test: ', y_test.shape

# Define Support Vectors Machine Classifier
clf = svm.SVC()
# Fit with training data and labels
clf.fit(X_train, y_train)

# Predict with the trained classifier for the testing set
y_pred = clf.predict(X_test)
print 'Accuracy: ', accuracy_score(y_test, y_pred)

print 'Classification report:'
print classification_report(y_test, y_pred)
# Save classifier
joblib.dump(clf, 'ship_hog_svm_clf.pkl') 