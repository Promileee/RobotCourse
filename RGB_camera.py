import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头，请检查摄像头是否连接或已被占用。")
    exit()

while True:
    ret, img = cap.read()
    if not ret or img is None:
        print("无法读取视频帧。")
        break

    cv2.imshow("RGB", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
