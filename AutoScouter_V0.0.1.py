#Installed CV2 library from instructions at:
#https://pypi.org/project/opencv-python/
#Used "pip3 install opencv-python" in CMD

#Standard imports
import cv2 #OpenCV for Python library
import numpy as np;
 
# Read image
im = cv2.imread("blob.jpg", cv2.IMREAD_COLOR)
 
# Set up the detector with default parameters
detector = cv2.SimpleBlobDetector_create()
 
# Detect blobs
keypoints = detector.detect(im)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

#this line of code does not work, apparently because of a bug in OpenCV 4.0.0 that has been fixed in 4.0.1
#however it seems that OpenCv-Python hasn't been updated to that version yet.
#im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints = im.copy()
for marker in keypoints:
	im_with_keypoints = cv2.drawMarker(im_with_keypoints, tuple(int(i) for i in marker.pt), color=(0, 255, 0))

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
