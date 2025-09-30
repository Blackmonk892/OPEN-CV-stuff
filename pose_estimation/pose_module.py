import cv2 as cv
import mediapipe as mp
import time
import math

class posedetector():
    def __init__(self,mode = False, upBody = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
        smooth_landmarks=self.smooth,
        min_detection_confidence=self.detectionCon,
        min_tracking_confidence=self.trackCon
        )

    def findpose(self,img, draw = True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS
                )
        return img
    
    def findposition(self, img, draw = True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm, in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy), 5, (255,0,0), cv.FILLED)
        return self.lmList
    
    def findangle(self,img,p1,p2,p3, draw = True):
        x1, y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        x3,y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2))

        if angle < 0:
            angle += 360
        
        if draw:
            cv.line(img,(x1,y1), (x2,y2), (255,255,255), 3)
            cv.line(img, (x3,y3), (x2,y2), (255,255,255), 3)
            cv.circle(img, (x1,y1), 10, (0,0,255), cv.FILLED)
            cv.circle(img, (x2,y2), 10, (0,0,255), cv.FILLED)
            cv.circle(img, (x3,y3), 10, (0,0,255), cv.FILLED)
            cv.circle(img, (x1,y1), 15, (0,0,255), 2)
            cv.circle(img, (x2,y2), 15, (0,0,255), 2)
            cv.circle(img, (x3,y3), 15, (0,0,255), 2)
            cv.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        return angle
    
url = "http://192.168.1.190:4747/video"

def main():
        cap = cv.VideoCapture(url)
        ptime = 0
        detector = posedetector()
        while True:
            success, img = cap.read()
            img =detector.findpose(img)
            lmList = detector.findposition(img, draw = False)
            if len(lmList) != 0:
                print(lmList[14])
                cv.circle(img, (lmList[14][1], lmList[14][2]), 15, (0,0,255), cv.FILLED)
            ctime = time.time()
            fps = 1/ (ctime - ptime)
            ptime = ctime
            cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_COMPLEX,3,(255,0,0), 3)
            cv.imshow("video", img)
            cv.waitKey(1)
        

if __name__ == "__main__":
    main()




