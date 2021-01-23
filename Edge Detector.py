# Edge Detector 0.1.0
# Caleb Keller March 16, 2019

# Use the sliders in the "Edges" window to control the edge detection parameters.
# Once the parameters are set correctly, press "a" to begin running hough line detection.
# A new window will appear with detected lines drawn in green on top of the original input.
# If there are too many edges, the program will slow down significantly. If it gets too slow, let go of the a key
# and the hough line detection algorithm will stop. To quit the program, press "q".

import cv2
import numpy as np
import random
import math

# Edge Detection and drawing function. (will probably be changed to not draw and just return a list of lines in the future)

def DrawEdges(Image):
    #print("drawing ",UpperLimit," , ",LowerLimit)

    # Use Canny edge detection to find all of the edges in the video using parameters input by the user.
    Edges = cv2.Canny(Image,LowerLimit,UpperLimit)

    # Detect lines if the a key is being pressed
    if cv2.waitKey(1) & 0xFF == ord('a'):
        
        Output = Image.copy()

        # Run the hough line transform and get a list of coordinates where each line is
        Lines = cv2.HoughLinesP(Edges, rho=1, theta=np.pi/180, threshold=Threshold, minLineLength=MinLineLength, maxLineGap=MaxLineGap)

        # Draw the lines
        if Lines is not None:
            for Line in Lines:
                for x1,y1,x2,y2 in Line:
                    cv2.line(Output,(x1,y1),(x2,y2),(0,255,0),2)
                    #print((y1-y2),"X + ",(x2-x1),"Y + ",(x1*y2)-(x2*y1)," = 0")
                    
        cv2.imshow("Lines",cv2.resize(Output,(500,400)))
        #cv2.imshow("Lines",Output)
        
            
    cv2.imshow("Edges",cv2.resize(Edges,(500,400)))

# Functions required to make the trackbars work

def UpperTrackbar(val):
    global UpperLimit
    UpperLimit = val

def LowerTrackbar(val):
    global LowerLimit
    LowerLimit = val

def ThresholdTrackbar(val):
    global Threshold
    Threshold = val

def LengthTrackbar(val):
    global MinLineLength
    MinLineLength = val

def GapTrackbar(val):
    global MaxLineGap
    MaxLineGap = val

# Setting default values for variables

UpperLimit = 490
LowerLimit = 330

Threshold = 170
MinLineLength = 845
MaxLineGap = 320

# Creating windows

cv2.namedWindow("Edges")
cv2.moveWindow("Edges", 0, 0)
cv2.createTrackbar("Upper Limit","Edges",UpperLimit,1000,UpperTrackbar)
cv2.createTrackbar("Lower Limit","Edges",LowerLimit,1000,LowerTrackbar)

cv2.namedWindow("Lines")
cv2.moveWindow("Lines", 501, 0)
cv2.createTrackbar("Threshold","Lines",Threshold,1000,ThresholdTrackbar)
cv2.createTrackbar("Minimum Line Length","Lines",MinLineLength,1000,LengthTrackbar)
cv2.createTrackbar("Maximum Line Gap","Lines",MaxLineGap,1000,GapTrackbar)

# Loading video from file

Cap = cv2.VideoCapture("TestVideo.mp4")

while True:
    
    ret, frame = Cap.read()

    #cv2.imshow("Input",cv2.resize(frame,(500,400)))

    DrawEdges(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

Cap.release()
cv2.destroyAllWindows()

