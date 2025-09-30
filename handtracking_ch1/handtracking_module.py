import cv2 as cv
import mediapipe as mp
import time


#url = "http://192.168.1.190:4747/video"

class handdetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findposition(self, img, handno = 0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handno]
            for id,lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy), 5, (255,0,255), cv.FILLED)
        return lmList
    
def main():
    ptime = 0
    ctime = 0
    cap = cv.VideoCapture(0)
    detector = handdetector()
    while True:
        success, img = cap.read()
        if not success:
            print("❌ Failed to grab frame.")
            continue
        img = detector.findhands(img)
        lmList = detector.findposition(img)
        if len(lmList) != 0:
            print(lmList[4])
        ctime = time.time()
        fps = 1/ (ctime - ptime)
        ptime = ctime
        cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
        cv.imshow("hand detection module", img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()

    

        