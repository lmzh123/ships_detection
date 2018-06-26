# import the necessary packages
import imutils

def pyramid(image, scale=1.5, minSize=(30, 30)):
	'''
	This function returns a set of scaled images known as the image
	pyramid, the initial image is downscaled by a certain scale until 
	the minimum size of the image is reached. 
	Args:
		image(numpy.array): Initial image as a numpy array.
		scale(float): Scale to perform the sequential downsamplings.
		minSize(tuple): (x,y) for the smallest image of the pyramid

	Returns:
		pyramid(list[numpy.array]): List of images as numpy arrays.
	'''
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	'''
	This function returns a set of windows or regions of interest from 
	an image
	Args:
		image(numpy.array): Initial image as a numpy array.
		stepSize(int): Amount of Vertical and horizontal pixels to
		slide the window.
		windowSize(tuple): (x,y) for the size of the window in pixels

	Returns:
		pyramid(list[numpy.array]): List of window images as numpy arrays.
	'''
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])