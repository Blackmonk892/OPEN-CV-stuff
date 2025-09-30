import cv2 as cv
import mediapipe as mp
import time

class facemeshdetector():
    def __init__(self, staticMode = False, maxFaces = 2, minDetectionCon = 0.5,minTrackCon = 0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,  
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)

    def findFaceMesh(self, img, draw = True):
        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_TESSELATION,self.drawSpec,self.drawSpec)
                    face = []
                    for id, lm in enumerate(faceLms.landmark):
                        ih,iw,ic = img.shape
                        x,y = int(lm.x * iw), int(lm.y * ih)
                        face.append([x,y])
                        faces.append(face)
        return img, faces
    

def main():
    cap = cv.VideoCapture(0)
    ptime = 0
    detector = facemeshdetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img,faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(faces[0])
            ctime = time.time()
            fps = 1/ (ctime - ptime)
            ptime = ctime
            cv.putText(img,f'FPS: {int(fps)}', (20,70),cv.FONT_HERSHEY_SCRIPT_SIMPLEX,3,(0,255,0),3)
            cv.imshow("image", img)
            if cv.waitKey(1) & 0xFF == ord('q'):
               break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()



        