import cv2

cap = cv2.VideoCapture("http://192.168.1.190:4747/video")

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("IP Webcam Stream", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
