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

COLOR_CONFIG = {
    "Red":   {"ranges": [(LOWER_RED1, UPPER_RED1), (LOWER_RED2, UPPER_RED2)], "bgr": (0, 0, 255)},
    "Green": {"ranges": [(LOWER_GREEN, UPPER_GREEN)],                           "bgr": (0, 255, 0)},
    "Blue":  {"ranges": [(LOWER_BLUE, UPPER_BLUE)],                            "bgr": (255, 0, 0)},
}

DEPTH_MAX = 1300
DEPTH_RATIO = 0.2
MIN_AREA = 500


def _make_color_mask(hsv, ranges):
    mask = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
    for low, high in ranges[1:]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def _find_rects(color_mask, depth_mask, min_area=MIN_AREA):
    """Find valid rectangles from a color mask, filtering by depth coverage."""
    rects = []
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)
        if not (0.2 < aspect < 5.0):
            continue

        # Clip ROI to image bounds
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, color_mask.shape[1]), min(y + h, color_mask.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue

        roi_color = color_mask[y1:y2, x1:x2]
        roi_depth = depth_mask[y1:y2, x1:x2]
        color_pixels = cv2.countNonZero(roi_color)
        if color_pixels == 0:
            continue
        in_range = cv2.countNonZero(cv2.bitwise_and(roi_color, roi_depth))
        if in_range / color_pixels < DEPTH_RATIO:
            continue

        rects.append((x, y, w, h))

    return rects


def detect_color_rects(max_frames=10):
    """
    Capture up to max_frames and return the first frame's RGB image and
    color-rect mappings where all three colors (Red, Green, Blue) are detected.

    Returns:
        img       : the RGB frame (BGR numpy array) where all three were found,
                    or the last captured frame if no complete detection.
        rects     : dict {"Red": (x,y,w,h), "Green": (x,y,w,h), "Blue": (x,y,w,h)}
                    Missing colors map to None.
        depth_data: the depth array (480x640 float32) for the returned frame.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open RGB camera")

    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    best_img = None
    best_rects = None
    best_depth = None

    try:
        for _ in range(max_frames):
            ret, img = cap.read()
            if not ret or img is None:
                continue

            depth_frame = depth_stream.read_frame()
            dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
            dpt1 = np.asarray(dframe_data[:, :, 0], dtype="float32")
            dpt2 = np.asarray(dframe_data[:, :, 1], dtype="float32")
            dpt2 *= 255
            dpt = dpt1 + dpt2
            dpt = dpt[:, ::-1]

            depth_mask = np.where((dpt > 0) & (dpt < DEPTH_MAX), 255, 0).astype(np.uint8)
            if depth_mask.shape[:2] != img.shape[:2]:
                depth_mask = cv2.resize(depth_mask, (img.shape[1], img.shape[0]))

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            frame_rects = {}
            for name, cfg in COLOR_CONFIG.items():
                color_mask = _make_color_mask(hsv, cfg["ranges"])
                found = _find_rects(color_mask, depth_mask)
                frame_rects[name] = found[0] if found else None

            best_img = img
            best_rects = frame_rects
            best_depth = dpt

            # Stop if all three found in this frame
            if all(v is not None for v in frame_rects.values()):
                break

    finally:
        cap.release()
        depth_stream.stop()
        dev.close()

    return best_img, best_rects, best_depth


# --- Debug visualization ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img, rects, _ = detect_color_rects(max_frames=5)

    if img is None:
        print("No frame captured.")
        exit()

    # Fill detected rect regions with pure color
    vis = img.copy()
    fill_colors = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}

    for name, rc in rects.items():
        if rc is not None:
            x, y, w, h = rc
            cv2.rectangle(vis, (x, y), (x + w, y + h), fill_colors[name], -1)
            print(f"{name}: x={x}, y={y}, w={w}, h={h}  ({x}~{x+w}, {y}~{y+h})")
        else:
            print(f"{name}: not detected")

    # BGR -> RGB for matplotlib
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(vis_rgb)
    plt.title("Detected Rectangles (filled)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
