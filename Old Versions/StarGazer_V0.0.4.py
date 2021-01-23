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
im = cv2.imread("example_vision_target_small_complex.jpg", cv2.IMREAD_COLOR)

#Convert BGR (RGB) to HSV
im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

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
min_points = 3
approx_poly_tolerance = 5
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
        contour_data = cv2.findContours(mask_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #RETR_EXTERNAL RETR_LIST

        contours = [  [ [point[0][0],point[0][1]] for point in contour ] for contour in contour_data[0] if len(contour) >= min_points  ] #strips out the "contours" that have less than the specified number of points
        simple_contours = []
        for contour in contour_data[0]:
            if len(contour) >= min_points:
                simple_data = cv2.approxPolyDP(contour, approx_poly_tolerance, True)
                if len(simple_data) >= min_points:
                    simple_contours.append(  [[point[0][0],point[0][1]] for point in simple_data]  )

        #Draw lines
        im_with_keypoints = im.copy()
        for contour in contours:
            im_with_keypoints = DrawClosedContour(im_with_keypoints, contour, (0,0,255), 2)

        for simple_contour in simple_contours:
            im_with_keypoints = DrawClosedContour(im_with_keypoints, simple_contour, (255,0,0), 1)

        #Display the images
        combined_im = cv2.vconcat([im_with_keypoints, mask_im_bgr])
        
        cv2.imshow("Images", combined_im)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
                break


#CLEAN UP
cv2.destroyAllWindows()


