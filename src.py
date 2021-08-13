import cv2
import numpy as np


img_trained = cv2.imread('steve.jpg') # train image
img_trained = cv2.resize(img_trained,(0,0) , fx=0.75 , fy=0.75) # resize the train image

orb = cv2.ORB_create() # Create an object of ORB detector
kp1 , desc1 = orb.detectAndCompute(img_trained,None) # Detect and Compute the key points and descriptors of the trained image
img_trained = cv2.drawKeypoints(img_trained,kp1,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True) # Create an object of BF MATCHER and define a method for it

cap = cv2.VideoCapture(0) # read from webcam

while True:
    ret,frame = cap.read() # read the frame of webcam
    frame = cv2.resize(frame,(0,0),fx=0.75 , fy=0.75) # resize the frame of webcam
    kp2 , desc2 = orb.detectAndCompute(frame,None) # Detect and Compute the key points and descriptors of the input image
    # frame = cv2.drawKeypoints(frame,kp2,None)
    
    matches = bf.match(desc1,desc2) # match the features of trained image and input image(webcam) by BF MATCHER
    if len(matches) > 10 :
        matches = sorted(matches,key= lambda x : x.distance) # sort the DMATCHES by their distance
        frame = cv2.drawMatches(img_trained,kp1,frame,kp2,matches[:10],None) # draw the match points
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2) # calculate and find the Source points of trained image
        des_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2) # calculate and find the Source points of input image
        M , mask = cv2.findHomography(src_pts,des_pts,cv2.RANSAC,5.0) # Do Homography by RANSAC method
    else:
        pass

    

    cv2.imshow('webcam' , frame) # show the output
    cv2.imshow('img trained' , img_trained) # show the trained image

    if cv2.waitKey(1) == 0xFF & ord('q'): # a condition for breaking the loop
        break


cv2.destroyAllWindows()
cap.release()