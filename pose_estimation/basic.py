import cv2 as cv
import mediapipe as mp
import time

url = "http://192.168.1.190:4747/video"
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv.VideoCapture(url)
ptime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            print(id,lm)
            cx,cy = int(lm.x * w), int(lm.y * h)
            cv.circle(img, (cx,cy), 5, (255,0,0), cv.FILLED)
    
    ctime = time.time()
    fps = 1/ (ctime - ptime)
    ptime = ctime
    cv.putText(img,str(int(fps)), (70,50), cv.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
    cv.imshow("image", img)
    cv.waitKey(1)