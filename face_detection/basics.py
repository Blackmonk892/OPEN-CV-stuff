import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

ptime = 0

mpfacedetection = mp.solutions.face_detection
mpdraw = mp.solutions.drawing_utils
facedetection = mpfacedetection.FaceDetection(0.75)

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = facedetection.process(imgRGB)
    print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv.rectangle(img, bbox,(255,0,255),2)
            cv.putText(img,f'{int(detection.score[0] * 100)}%', (bbox[0],bbox[1] - 20), cv.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
    
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv.putText(img,f'FPS: {int(fps)}', (20,70), cv.FONT_HERSHEY_COMPLEX,3, (0,255,0),2)
    cv.imshow("image", img)
    cv.waitKey(1)


