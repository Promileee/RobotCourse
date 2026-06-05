import cv2
import numpy as np
from openni import openni2
import itertools

DEPTH_MAX = 1300
DEPTH_RATIO = 0.2

# Reference BGR colors for matching
REF_COLORS = {
    "Red":   (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue":  (255, 0, 0),
}
COLOR_NAMES = ["Red", "Green", "Blue"]


def _rect_depth_ok(x, y, w, h, depth_mask, ratio=DEPTH_RATIO):
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, depth_mask.shape[1]), min(y + h, depth_mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return False
    roi_depth = depth_mask[y1:y2, x1:x2]
    total = (x2 - x1) * (y2 - y1)
    if total == 0:
        return False
    return cv2.countNonZero(roi_depth) / total >= ratio


def _pick_best_triple(candidates):
    """Pick the 3 rectangles with most similar areas."""
    if len(candidates) < 3:
        return candidates
    candidates.sort(key=lambda r: r[4])
    best, best_var = None, float("inf")
    for i in range(len(candidates) - 2):
        triple = candidates[i:i + 3]
        areas = [r[4] for r in triple]
        var = np.var(areas) / (np.mean(areas) + 1e-6)
        if var < best_var:
            best_var = var
            best = triple
    return best


def _match_colors(selected, frame):
    """Match 3 rectangles to Red/Green/Blue by closest mean BGR."""
    if len(selected) != 3:
        return [(x, y, w, h, "?") for (x, y, w, h, _) in selected]

    rect_means = []
    for (x, y, w, h, _) in selected:
        roi = frame[y:y + h, x:x + w]
        m = cv2.mean(roi)[:3]
        rect_means.append(m)

    best_perm, best_dist = None, float("inf")
    for perm in itertools.permutations(COLOR_NAMES):
        dist = sum(
            np.sqrt((REF_COLORS[p][0] - rect_means[i][0]) ** 2 +
                    (REF_COLORS[p][1] - rect_means[i][1]) ** 2 +
                    (REF_COLORS[p][2] - rect_means[i][2]) ** 2)
            for i, p in enumerate(perm)
        )
        if dist < best_dist:
            best_dist = dist
            best_perm = perm

    return [(x, y, w, h, best_perm[i]) for i, (x, y, w, h, _) in enumerate(selected)]


# --- Cameras ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open RGB camera.")
    exit()

openni2.initialize()
dev = openni2.Device.open_any()
print(dev.get_device_info())
depth_stream = dev.create_depth_stream()
depth_stream.start()

cv2.namedWindow("3. Morphology")
cv2.createTrackbar("CannyLow", "3. Morphology", 50, 255, lambda _: None)
cv2.createTrackbar("CannyHigh", "3. Morphology", 150, 255, lambda _: None)
cv2.createTrackbar("CloseIter", "3. Morphology", 2, 5, lambda _: None)
cv2.createTrackbar("MinArea", "3. Morphology", 500, 5000, lambda _: None)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Cannot read RGB frame.")
        break

    # --- Depth (for later; mask not applied to morphology here) ---
    depth_frame = depth_stream.read_frame()
    dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
    dpt1 = np.asarray(dframe_data[:, :, 0], dtype="float32")
    dpt2 = np.asarray(dframe_data[:, :, 1], dtype="float32")
    dpt2 *= 255
    dpt = dpt1 + dpt2
    dpt = dpt[:, ::-1]

    depth_mask = np.where((dpt > 0) & (dpt < DEPTH_MAX), 255, 0).astype(np.uint8)
    if depth_mask.shape[:2] != frame.shape[:2]:
        depth_mask = cv2.resize(depth_mask, (frame.shape[1], frame.shape[0]))

    # --- Morphology pipeline (RGB only) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    canny_low = cv2.getTrackbarPos("CannyLow", "3. Morphology")
    canny_high = cv2.getTrackbarPos("CannyHigh", "3. Morphology")
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Morphological close to connect edge fragments into solid contours
    close_iter = cv2.getTrackbarPos("CloseIter", "3. Morphology")
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    # Open to remove small noise
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Find rectangular candidates ---
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = cv2.getTrackbarPos("MinArea", "3. Morphology")
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * float(h)
        extent = area / rect_area if rect_area > 0 else 0

        # Polygon approximation
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Rectangular: 4~8 vertices, high extent, reasonable aspect ratio
        if not (4 <= len(approx) <= 10 and extent > 0.55):
            continue
        if not (0.25 < w / float(h) < 4.0):
            continue

        # Depth check
        if not _rect_depth_ok(x, y, w, h, depth_mask):
            continue

        candidates.append((x, y, w, h, area))

    # --- Pick 3 most similar-sized rectangles ---
    selected = _pick_best_triple(candidates)

    # --- Color matching ---
    matched = _match_colors(selected, frame)

    # --- Visualization ---

    # 1) Original RGB
    rgb_vis = frame.copy()

    # 2) Depth colormap
    depth_vis = cv2.normalize(np.clip(dpt, 0, DEPTH_MAX), None, 0, 255,
                              cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # 3) Morphology: edges + morphed side by side
    morph_vis = np.hstack([
        cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR),
    ])
    cv2.putText(morph_vis, f"Canny [{canny_low},{canny_high}]", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(morph_vis, "Morphed", (edges.shape[1] + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    h, w = morph_vis.shape[:2]
    cv2.putText(morph_vis, f"Candidates: {len(candidates)}  Selected: {len(selected)}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 4) Rectangle detection
    rect_det_vis = frame.copy()
    for (x, y, w, h, _) in candidates:
        cv2.rectangle(rect_det_vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
    for (x, y, w, h, _) in selected:
        cv2.rectangle(rect_det_vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # 5) Color matching
    color_match_vis = frame.copy()
    for (x, y, w, h, cname) in matched:
        bgr = REF_COLORS.get(cname, (255, 255, 255))
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), bgr, -1)
        color_match_vis = cv2.addWeighted(color_match_vis, 0.5, overlay, 0.5, 0)
        cv2.rectangle(color_match_vis, (x, y), (x + w, y + h), bgr, 2)
        cx, cy = x + w // 2, y + h // 2
        if 0 <= cy < dpt.shape[0] and 0 <= cx < dpt.shape[1]:
            d_val = dpt[cy, cx]
        else:
            d_val = -1
        label = f"{cname} ({cx},{cy}) d={d_val:.0f}mm"
        cv2.putText(color_match_vis, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

    cv2.imshow("1. RGB", rgb_vis)
    cv2.imshow("2. Depth", depth_vis)
    cv2.imshow("3. Morphology", morph_vis)
    cv2.imshow("4. Rect Detection", rect_det_vis)
    cv2.imshow("5. Color Match", color_match_vis)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
depth_stream.stop()
dev.close()
cv2.destroyAllWindows()