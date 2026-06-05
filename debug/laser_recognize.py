from openni import openni2
import cv2
import numpy as np

# --- RGB camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open RGB camera")
    exit()

# --- Depth camera ---
openni2.initialize()
dev = openni2.Device.open_any()
print(dev.get_device_info())
depth_stream = dev.create_depth_stream()
depth_stream.start()

DEPTH_MAX = 1300  # 1m = 1000mm, only detect laser within this range

cv2.namedWindow("Result")
cv2.createTrackbar("Threshold", "Result", 235, 255, lambda _: None)
cv2.createTrackbar("Blur", "Result", 3, 20, lambda _: None)
cv2.createTrackbar("DepthMax_mm", "Result", DEPTH_MAX, 5000, lambda _: None)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Cannot read RGB frame")
        break

    # --- Depth frame ---
    depth_frame = depth_stream.read_frame()
    dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
    dpt1 = np.asarray(dframe_data[:, :, 0], dtype="float32")
    dpt2 = np.asarray(dframe_data[:, :, 1], dtype="float32")
    dpt2 *= 255
    dpt = dpt1 + dpt2
    dpt = dpt[:, ::-1]  # flip to align with RGB (adjust if needed)

    depth_max = cv2.getTrackbarPos("DepthMax_mm", "Result")

    # Depth mask: valid (0 < depth < max)
    depth_mask = np.where((dpt > 0) & (dpt < depth_max), 255, 0).astype(np.uint8)

    # Resize depth mask to match RGB frame size if different
    if depth_mask.shape[:2] != frame.shape[:2]:
        depth_mask = cv2.resize(depth_mask, (frame.shape[1], frame.shape[0]))

    # --- Brightness detection ---
    thresh_val = cv2.getTrackbarPos("Threshold", "Result")
    blur_size = cv2.getTrackbarPos("Blur", "Result")
    if blur_size % 2 == 0:
        blur_size += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    _, bright = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    # --- Combine: bright AND within depth range ---
    combined = cv2.bitwise_and(bright, depth_mask)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = frame.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 2:
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Get depth at the detected point
                if 0 <= cy < dpt.shape[0] and 0 <= cx < dpt.shape[1]:
                    d_val = dpt[cy, cx]
                else:
                    d_val = -1
                cv2.circle(result, (cx, cy), 15, (0, 255, 0), 2)
                cv2.circle(result, (cx, cy), 3, (0, 255, 0), -1)
                cv2.putText(result, f"({cx}, {cy}) d={d_val:.0f}mm",
                            (cx + 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Depth colormap for visualization
    depth_vis = cv2.normalize(np.clip(dpt, 0, depth_max), None, 0, 255,
                              cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    cv2.imshow("Result", result)
    cv2.imshow("Depth + Mask", np.hstack([depth_vis,
              cv2.cvtColor(depth_mask, cv2.COLOR_GRAY2BGR) * 255]))
    cv2.imshow("Bright + Combined", np.hstack([bright, combined]))

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
depth_stream.stop()
dev.close()
cv2.destroyAllWindows()