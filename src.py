# import the necessary libraries
import cv2
import numpy as np

######

frame_num = 0
detected_obj = False

#######

trained_img = cv2.imread('steve.jpg') # read the sample image
hI , wI , cI = trained_img.shape # gettin the Height, width and channel of the trained image

sample_video = cv2.VideoCapture('steve_video.mp4') # read the sample video
# sample_video.set(1,854)
success , video_frame = sample_video.read() # read each frame of sample video
video_frame = cv2.resize(video_frame,(wI,hI)) # resize the sample video


orb = cv2.ORB_create(nfeatures=1000) # create an object of ORB detector
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True) # create BFMatcher object

kp1, desc1 = orb.detectAndCompute(trained_img,None) # Detect and Compute the key points and descriptors of the trained image

cap = cv2.VideoCapture(0) # read from the webcam source

while True:
    ret, frame = cap.read() # read each frame of webcam
    copy_frame = frame.copy() # copy the frame

    if detected_obj == True:
        if frame_num == sample_video.get(cv2.CAP_PROP_FRAME_COUNT): # if the frame of video is 0
            sample_video.set(cv2.CAP_PROP_POS_FRAMES,0) # return at the first frame of video 
            frame_num = 0

        success , video_frame = sample_video.read() # read each frame of video
        video_frame = cv2.resize(video_frame,(wI,hI)) # resize it like sample image
    else:
        sample_video.set(cv2.CAP_PROP_POS_FRAMES,0) # stop show the video and return to the first frame
        frame_num = 0


    kp2, desc2 = orb.detectAndCompute(frame,None) # Detect and Compute the key points and descriptors of the input frame
    matches = bf.match(desc1,desc2) # Match descriptors.

    if len(matches) > 15:
        detected_obj = True
        matches = sorted(matches, key= lambda x : x.distance) # Sort them in the order of their distance.
        drawn_img = cv2.drawMatches(trained_img,kp1,frame,kp2,matches[:10],None) # Draw first 10 matches.
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2) # calculate and find the Source points of trained image
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2) # calculate and find the Source points of input image
        M, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0) # Do Homography by RANSAC method

        
        src_coor_pts = np.float32([ [0,0], [0,hI], [wI,hI], [wI,0]]).reshape(-1,1,2) # calculate the coordinates of the trained image
        dst_coor_pts = cv2.perspectiveTransform(src_coor_pts,M) # calculate the coordinates of input frame (from webcam)
        frame = cv2.polylines(frame,[np.int32(dst_coor_pts)],True,(0,0,255),2) # draw a polyline (polygon) on detected image that is display (from webcam)

        warp_video = cv2.warpPerspective(video_frame,M,(frame.shape[1],frame.shape[0])) # wrapped by M and sample video to make an image with a black background and a video instead of detected image

        mask_win = np.zeros((frame.shape[0],frame.shape[1]),np.uint8) # make a black image by numpy
        mask_win = cv2.fillPoly(mask_win,[np.int32(dst_coor_pts)],(255,255,255)) # make a white space instead of video on balck image
        mask_win_inv = cv2.bitwise_not(mask_win) # inverse the black and white by bitwise operator
        copy_frame = cv2.bitwise_and(copy_frame,copy_frame,mask=mask_win_inv) # show the Photo margins from webcam
        copy_frame = cv2.bitwise_or(warp_video,copy_frame) # show the video instead of sample image

        cv2.imshow('warp_video',copy_frame) # show the output

    else: # it has to work when the feature detection could not find the features (the video must be display)
        detected_obj = False
        copy_frame = frame # put the real frame instead of copy frame
        cv2.imshow('warp_video',copy_frame) # show the result


    

    if cv2.waitKey(35) ==  0XFF & ord('q'): # condition for breaking the loop (if you press q)
        break
    frame_num += 1


cap.release()
cv2.destroyAllWindows()

