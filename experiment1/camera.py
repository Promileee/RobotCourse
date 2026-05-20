import cv2
import numpy as np


def make_tile(image, title, size):
    tile = cv2.resize(image, size)
    cv2.putText(tile, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return tile


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = np.flip(frame, axis=1).copy()

        height, width = frame.shape[:2]
        panel_size = (width, height)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        red = frame.copy()
        red[:, :, 0] = 0
        red[:, :, 1] = 0

        green = frame.copy()
        green[:, :, 0] = 0
        green[:, :, 2] = 0

        blue = frame.copy()
        blue[:, :, 1] = 0
        blue[:, :, 2] = 0

        blank = np.zeros_like(frame)

        top_row = cv2.hconcat([
            make_tile(frame, "Original", panel_size),
            make_tile(gray, "Gray", panel_size),
            make_tile(red, "Red", panel_size),
        ])
        bottom_row = cv2.hconcat([
            make_tile(green, "Green", panel_size),
            make_tile(blue, "Blue", panel_size),
            make_tile(blank, "", panel_size),
        ])

        canvas = cv2.vconcat([top_row, bottom_row])
        cv2.imshow("Camera Demo", canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
