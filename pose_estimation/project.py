import cv2 as cv
import time
from pose_estimation import pose_module as pm

cap = cv.VideoCapture(r"C:\Users\xoxo3\Desktop\video-for-pose-detection-1-1080x1920-30fps.mp4")
ptime = 0
detector = pm.posedetector()

while True:
    success, img = cap.read()
    img = cv.resize(img,(360,640))
    img = detector.findpose(img)

    lmList = detector.findposition(img, draw = False)
    if len(lmList) != 0:
        print(lmList[11])
        cv.circle(img, (lmList[11][1], lmList[11][2]), 15, (0,0,255), cv.FILLED)
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_COMPLEX,3, (255,0,0), 3)
    cv.imshow("video", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

