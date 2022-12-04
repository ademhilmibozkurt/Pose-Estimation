# this modules purpose is use estimator in other projects

import cv2 as cv
import mediapipe as mp
import time

class PoseDetector():
    
    def __init__(self, mode=False, upperBody=False, smooth=True,
                 detectionCon=True, trackCon=0.5):
        
        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upperBody,
                                     self.smooth, self.detectionCon, self.trackCon)
        
    def findPose(self, img, draw=True):
    
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
    
        if self.results.pose_landmarks:
            if draw:    
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            
        return img

    def getPosition(self, img, draw=True):
        
        self.lmList = []
        if self.results.pose_landmarks:
            
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])    
                
                if draw:
                    cv.circle(img, (cx, cy), 10, (255, 0, 0), cv.FILLED)
           
        return self.lmList
   
   
def main():
    
    capture = cv.VideoCapture('videos/8.mp4')
    pTime=0

    detector = PoseDetector()    

    while True:
       
        success, img = capture.read()
        img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        img = detector.findPose(img)
    
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
    
        cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN,
                3, (20, 53, 200), 3)
    
        cv.imshow('Poses',img)
        if cv.waitKey(1) & 0xFF==ord(" "):
            break
    

if __name__=="__main__":
    main()