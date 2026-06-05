import cv2
import numpy as np
from openni import openni2

# --- HSV ranges for red, blue, green ---
LOWER_RED1 = np.array([0, 100, 50])
UPPER_RED1 = np.array([15, 255, 255])
LOWER_RED2 = np.array([160, 100, 50])
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
DEPTH_RATIO = 0.2  # at least 20% of rect area must be within depth range

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
        color_mask = cv2.inRange(hsv, range1[0], range1[1])
        if range2 is not None:
            mask2 = cv2.inRange(hsv, range2[0], range2[1])
            color_mask = cv2.bitwise_or(color_mask, mask2)

        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Find contours from color mask only (no depth filtering yet)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if not (0.2 < aspect_ratio < 5.0):
                continue

            # Check depth coverage: what % of this rect's color pixels are within depth range
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, color_mask.shape[1]), min(y + h, color_mask.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue

            roi_color = color_mask[y1:y2, x1:x2]
            roi_depth = depth_mask[y1:y2, x1:x2]
            color_pixels = cv2.countNonZero(roi_color)
            if color_pixels == 0:
                continue
            in_range_pixels = cv2.countNonZero(cv2.bitwise_and(roi_color, roi_depth))
            depth_ratio = in_range_pixels / color_pixels

            if depth_ratio < DEPTH_RATIO:
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

    # Depth colormap for visualization
    depth_vis = cv2.normalize(np.clip(dpt, 0, DEPTH_MAX), None, 0, 255,
                              cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    cv2.imshow("Result", result)
    cv2.imshow("Depth", depth_vis)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
depth_stream.stop()
dev.close()
cv2.destroyAllWindows()
