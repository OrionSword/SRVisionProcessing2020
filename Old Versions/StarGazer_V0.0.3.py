#Installed CV2 library from instructions at:
#https://pypi.org/project/opencv-python/
#Used "pip3 install opencv-python" in CMD
#created with Python 3.6 on 04-22-2020 by Orion DeYoe

#Standard imports
import cv2 #OpenCV for Python library
import numpy as np

def PointDistance(X1, Y1, X2, Y2):
    return math.sqrt((X2-X1)**2 + (Y2-Y1)**2)

def PointDirection(X1, Y1, X2, Y2): #in radians
    DX = X2 - X1
    DY = Y2 - Y1
    
    if ((DX >= 0) and (DY == 0)):
        return 0

    elif ((DX > 0) and (DY > 0)):
        return math.atan(DY / DX)

    elif ((DX == 0) and (DY > 0)):
        return math.pi * 0.5

    elif ((DX < 0) and (DY > 0)):
        return math.atan(DY / DX) + math.pi

    elif ((DX < 0) and (DY == 0)):
        return math.pi

    elif ((DX < 0) and (DY < 0)):
        return math.atan(DY / DX) + math.pi

    elif ((DX == 0) and (DY < 0)):
        return math.pi * 1.5

    elif ((DX > 0) and (DY < 0)):
        return math.atan(DY / DX) + 2 * math.pi

def Radians(ang):
    return ang * math.pi / 180

def Degrees(ang):
    return ang * 180 / math.pi

def DrawClosedContour(image, contour, color=(0,0,255), thiccness=1):
        last_point = None
        for point in contour:
                if last_point is not None:
                        image = cv2.line(image,(last_point[0],last_point[1]),(point[0],point[1]),color,thiccness)
                last_point = point
        image = cv2.line(image,(last_point[0],last_point[1]),(contour[0][0],contour[0][1]),color,thiccness)
        return image
 
# Read image
im = cv2.imread("example_vision_target_small.jpg", cv2.IMREAD_COLOR)

#Convert BGR (RGB) to HSV
im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# Setup SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
params.minThreshold = 128
params.maxThreshold = 255
params.thresholdStep = 10

        # Merge blobs that are close together
params.minDistBetweenBlobs = 5

        # Filter by Area
params.filterByArea = False
#params.minArea = 1500

        # Filter by Circularity
params.filterByCircularity = False
#params.minCircularity = 0.1

        # Filter by Convexity
params.filterByConvexity = False
#params.minConvexity = 0.87

        # Filter by Inertia
params.filterByInertia = False
#params.minInertiaRatio = 0.01

# Set up the detector with parameters
# OpenCV 3 & 4 requires adding the "_create" to the constructor below
detector = cv2.SimpleBlobDetector_create(params)

#Create window
cv2.namedWindow("Thresholds") #, cv2.WINDOW_AUTOSIZE
cv2.moveWindow("Thresholds", 0, 0)
cv2.resizeWindow("Thresholds", 500, 350)
cv2.createTrackbar("H High","Thresholds",102,255,lambda x:x)
cv2.createTrackbar("S High","Thresholds",255,255,lambda x:x)
cv2.createTrackbar("V High","Thresholds",255,255,lambda x:x)
cv2.createTrackbar("H Low","Thresholds",74,255,lambda x:x)
cv2.createTrackbar("S Low","Thresholds",150,255,lambda x:x)
cv2.createTrackbar("V Low","Thresholds",129,255,lambda x:x)

cv2.namedWindow("Images")
cv2.moveWindow("Images", 700, 0)

#is it loop time brother
cont = True
while cont:
        #Threshold image
        h_upper_threshold = cv2.getTrackbarPos("H High", "Thresholds")
        s_upper_threshold = cv2.getTrackbarPos("S High", "Thresholds")
        v_upper_threshold = cv2.getTrackbarPos("V High", "Thresholds")
        h_lower_threshold = cv2.getTrackbarPos("H Low", "Thresholds")
        s_lower_threshold = cv2.getTrackbarPos("S Low", "Thresholds")
        v_lower_threshold = cv2.getTrackbarPos("V Low", "Thresholds")
        
        upper_thresholds = np.array([h_upper_threshold,s_upper_threshold,v_upper_threshold])
        lower_thresholds = np.array([h_lower_threshold,s_lower_threshold,v_lower_threshold])

        mask_im = cv2.inRange(im_hsv, lower_thresholds, upper_thresholds)
        mask_im_bgr = cv2.cvtColor(mask_im, cv2.COLOR_GRAY2BGR)

        #Detect blobs
        keypoints = detector.detect(mask_im_bgr)
        contours = cv2.findContours(mask_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        simple_contour = cv2.approxPolyDP(contours[0][0], 10, True)
        
        one_contour = [[point[0][0],point[0][1]] for point in contours[0][0]]
        """one_contour = []
        for point in contours[0][0]:
                one_contour.append([point[0][0],point[0][1]])"""
        one_simple_contour = [[point[0][0],point[0][1]] for point in simple_contour]

        #Draw keypoints
        im_with_keypoints = im.copy()
        for marker in keypoints:
                im_with_keypoints = cv2.drawMarker(im_with_keypoints, tuple(int(i) for i in marker.pt), color=(0, 255, 0))

        #Draw lines
        im_with_keypoints = DrawClosedContour(im_with_keypoints, one_contour, (0,0,255), 2)
        im_with_keypoints = DrawClosedContour(im_with_keypoints, one_simple_contour, (255,0,0), 1)

        #Display the images
        combined_im = cv2.vconcat([im_with_keypoints, mask_im_bgr])
        
        cv2.imshow("Images", combined_im)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
                break


#CLEAN UP
cv2.destroyAllWindows()


# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

#this line of code does not work, apparently because of a bug in OpenCV 4.0.0 that has been fixed in 4.0.1
#however it seems that OpenCv-Python hasn't been updated to that version yet.
#    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
