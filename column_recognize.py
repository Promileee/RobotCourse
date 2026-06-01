import cv2
import numpy as np
from openni import openni2

# --- HSV ranges for red, blue, green ---
LOWER_RED1 = np.array([0, 120, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 70])
UPPER_RED2 = np.array([180, 255, 255])

LOWER_BLUE = np.array([100, 120, 70])
UPPER_BLUE = np.array([130, 255, 255])

LOWER_GREEN = np.array([40, 70, 70])
UPPER_GREEN = np.array([80, 255, 255])

COLORS = {
    "Red": ((0, 0, 255), (LOWER_RED1, UPPER_RED1), (LOWER_RED2, UPPER_RED2)),
    "Blue": ((255, 0, 0), (LOWER_BLUE, UPPER_BLUE), None),
    "Green": ((0, 255, 0), (LOWER_GREEN, UPPER_GREEN), None),
}

DEPTH_MAX = 1300  # only detect within 1300mm

# --- RGB camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open RGB camera. Check connection.")
    exit()

# --- Depth camera ---
openni2.initialize()
dev = openni2.Device.open_any()
print(dev.get_device_info())
depth_stream = dev.create_depth_stream()
depth_stream.start()

cv2.namedWindow("Result")
cv2.createTrackbar("Min Area", "Result", 500, 5000, lambda _: None)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Cannot read RGB frame.")
        break

    depth_frame = depth_stream.read_frame()
    dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
    dpt1 = np.asarray(dframe_data[:, :, 0], dtype="float32")
    dpt2 = np.asarray(dframe_data[:, :, 1], dtype="float32")
    dpt2 *= 255
    dpt = dpt1 + dpt2
    dpt = dpt[:, ::-1]  # flip to align with RGB

    # Depth mask: within range
    depth_mask = np.where((dpt > 0) & (dpt < DEPTH_MAX), 255, 0).astype(np.uint8)
    if depth_mask.shape[:2] != frame.shape[:2]:
        depth_mask = cv2.resize(depth_mask, (frame.shape[1], frame.shape[0]))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    min_area = cv2.getTrackbarPos("Min Area", "Result")
    result = frame.copy()

    for name, (color_bgr, range1, range2) in COLORS.items():
        mask = cv2.inRange(hsv, range1[0], range1[1])
        if range2 is not None:
            mask2 = cv2.inRange(hsv, range2[0], range2[1])
            mask = cv2.bitwise_or(mask, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Combine color mask with depth mask
        mask = cv2.bitwise_and(mask, depth_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if not (0.2 < aspect_ratio < 5.0):
                continue

            # Center of the rectangle
            cx, cy = x + w // 2, y + h // 2
            if 0 <= cy < dpt.shape[0] and 0 <= cx < dpt.shape[1]:
                d_val = dpt[cy, cx]
            else:
                d_val = -1

            cv2.rectangle(result, (x, y), (x + w, y + h), color_bgr, 2)
            label = f"{name} ({cx},{cy}) d={d_val:.0f}mm"
            cv2.putText(result, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

    cv2.imshow("Result", result)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
depth_stream.stop()
dev.close()
cv2.destroyAllWindows()
