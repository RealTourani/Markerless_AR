import cv2

img_trained = cv2.imread('steve.jpg')
img_trained = cv2.resize(img_trained,(0,0) , fx=0.75 , fy=0.75)

orb = cv2.ORB_create()
kp1 , desc1 = orb.detectAndCompute(img_trained,None)
# img_trained = cv2.drawKeypoints(img_trained,kp1,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(0,0),fx=0.75 , fy=0.75)
    kp2 , desc2 = orb.detectAndCompute(frame,None)
    # frame = cv2.drawKeypoints(frame,kp2,None)

    matches = bf.match(desc1,desc2)
    matches = sorted(matches,key= lambda x : x.distance)
    frame = cv2.drawMatches(img_trained,kp1,frame,kp2,matches[:10],None)
    
    cv2.imshow('webcam' , frame)
    cv2.imshow('img trained' , img_trained)

    if cv2.waitKey(1) == 0xFF & ord('q'):
        break


cv2.destroyAllWindows()
cap.release()