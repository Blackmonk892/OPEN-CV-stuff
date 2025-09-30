import cv2 as cv
import numpy as np

image = cv.imread(r"C:\Users\xoxo3\OneDrive\Desktop\PYTHON\OPENCV-PYTHON_3.10\contour_detection\contour2.jpg")


output = image.copy()

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (5,5), 0)

edges = cv.Canny(blur, 50, 150)

contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv.contourArea(cnt) < 1000:
        continue
    epsilon = 0.03 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    x,y,w,h =cv.boundingRect(approx)
    aspect_ration = float(w)/h

    if 5 < len(approx) <= 7 and (aspect_ration < 0.6 or aspect_ration > 1.4):
        cv.drawContours(output, [approx], -1, (0,255,0), 4)
        cv.putText(output, "Arrow", (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)


cv.imshow("Edges", edges)
cv.imshow("Detected arrows", output)
cv.waitKey(0)
cv.destroyAllWindows()