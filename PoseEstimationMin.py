import cv2 as cv
import mediapipe as mp
import time

#%% instance for draw landmark nodes
mpDraw = mp.solutions.drawing_utils

#%% pose detection
mpPose = mp.solutions.pose
pose = mpPose.Pose()

#%% import videos from disk
capture = cv.VideoCapture('videos/0.mp4')

# previous time
pTime=0

#%% collecting image from video and showing
while True:
    
    # read frame by frame and load into img
    success, img = capture.read()
    
    # mediapipe running with rgb
    # but opencv uses gbr. lets convert
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # pose detection results
    results = pose.process(imgRGB)
    
    # landmarks shows us the nodes of body that detected pose
    # if it is true thar means detect some poses
    if results.pose_landmarks:
        # draw nodes
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        
        # numerate pose points and learn where exactly it is
        for id, lm in enumerate(results.pose_landmarks.landmark):
            
            # take each landmarks coordinates
            h, w, c = img.shape
            
            # coordinates
            cx, cy = int(lm.x*w), int(lm.y*h)
            # this code allows us know to "Is coordinates true?"
            #cv.circle(img, (cx, cy), 10, (255, 0, 0), cv.FILLED)
        
    # fps configurations for drawing
      # current time
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    # draw fps
    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN,
                3, (20, 53, 200), 3)
    
    # show frames
    cv.imshow('Poses',img)
    if cv.waitKey(1) & 0xFF==ord(" "):
        break