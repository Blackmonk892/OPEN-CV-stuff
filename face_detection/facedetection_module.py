import cv2 as cv
import mediapipe as mp
import time

class facedetector():
    def __init__(self,minDetectionCon = 0.5):

        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    
    def findFace(self, img,draw = True):

        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih,iw,ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                bboxs.append([id,bbox,detection.score])

                if draw:
                    img = self.fancyDraw(img,bbox)
                    cv.putText(img,f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        return img, bboxs
    

    def fancyDraw(self, img, bbox, l = 30, t=5, rt =1):
        x,y,w,h = bbox
        x1,y1 = x + w, y + h

        cv.rectangle(img,bbox,(255,0,255), rt)
        cv.line(img,(x,y), (x + 1,y), (255,0,255), t)
        cv.line(img, (x,y), (x, y + 1), (255,0,255), t)
        cv.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        cv.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        cv.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img


def main():
    cap = cv.VideoCapture(0)
    ptime = 0
    detector = facedetector()

    while True:
        success, img = cap.read()
        img, bboxs = detector.findFace(img)
        print(bboxs)

        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv.imshow("image", img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()