import cv2 as cv
import mediapipe as mp
import time

url = "http://192.168.1.190:4747/video"

cap = cv.VideoCapture(url)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

ptime = 0
ctime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)


    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id,cx,cy)
                cv.circle(img,(cx,cy), 5, (255,0,255), cv.FILLED)
            
            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)

    
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
           

    cv.imshow("handtracking", img)
    cv.waitKey(1)

