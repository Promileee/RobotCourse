import cv2
import numpy as np
from openni import openni2


# ═══════════════════════════════════════════════════════════════
# RANSAC 圆拟合 —— 对底部杂物等离群点鲁棒
# ═══════════════════════════════════════════════════════════════

def _fit_circle_3pts(p1, p2, p3):
    """三点拟合圆，返回 (cx, cy, r) 或 None（共线时）"""
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    mx1, my1 = (x1 + x2) / 2, (y1 + y2) / 2
    mx2, my2 = (x2 + x3) / 2, (y2 + y3) / 2
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x3 - x2, y3 - y2
    eps = 1e-10

    if abs(dy1) < eps and abs(dy2) < eps:
        return None
    if abs(dy1) < eps:
        cx = mx1
        cy = my2 - (dx2 / dy2) * (cx - mx2)
    elif abs(dy2) < eps:
        cx = mx2
        cy = my1 - (dx1 / dy1) * (cx - mx1)
    else:
        a1, b1, c1 = dx1, dy1, mx1 * dx1 + my1 * dy1
        a2, b2, c2 = dx2, dy2, mx2 * dx2 + my2 * dy2
        det = a1 * b2 - a2 * b1
        if abs(det) < eps:
            return None
        cx = (c1 * b2 - c2 * b1) / det
        cy = (a1 * c2 - a2 * c1) / det

    r = np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
    return (cx, cy, r)


def ransac_fit_circle(points, n_iter=200, inlier_thresh=6, min_inlier_ratio=0.35):
    """
    RANSAC 圆拟合。随机取3点确定候选圆，统计圆周上的内点数。
    底部杂物不在圆盘圆周上 → 被当离群点忽略。
    圆盘上方弧线干净 → 主导拟合结果。
    """
    n_pts = len(points)
    if n_pts < 10:
        return None

    best_inliers = 0
    best_circle = None

    for _ in range(n_iter):
        idx = np.random.choice(n_pts, 3, replace=False)
        circle = _fit_circle_3pts(points[idx[0]], points[idx[1]], points[idx[2]])
        if circle is None:
            continue
        cx, cy, r = circle
        if r <= 0 or r > 500:
            continue

        dists = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        inliers = np.sum(np.abs(dists - r) < inlier_thresh)

        if inliers > best_inliers:
            best_inliers = inliers
            best_circle = (cx, cy, r, inliers)

    if best_circle is None:
        return None

    inlier_ratio = best_circle[3] / n_pts
    if inlier_ratio < min_inlier_ratio:
        return None

    return best_circle


# ═══════════════════════════════════════════════════════════════
# 深度图圆盘检测
# ═══════════════════════════════════════════════════════════════

def detect_disk_from_depth(dpt, depth_min, depth_max, min_radius, max_radius,
                           min_circularity, min_area, kernel_size):
    mask = np.where((dpt > depth_min) & (dpt < depth_max), 255, 0).astype(np.uint8)

    ks = max(1, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best = None
    best_score = 0

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        comp_mask = (labels == i).astype(np.uint8)
        ys, xs = np.where(comp_mask)
        if len(xs) < 10:
            continue

        # 深度平坦度
        depths = dpt[ys, xs]
        depth_std = np.std(depths)
        flatness = max(0.0, 1.0 - depth_std / 50.0)

        # RANSAC 圆拟合（对底部杂物鲁棒）
        points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        ransac_result = ransac_fit_circle(points)
        if ransac_result is None:
            continue

        cx, cy, r, n_inliers = ransac_result
        cx, cy, r = int(cx), int(cy), int(r)

        if r < min_radius or r > max_radius:
            continue

        # 圆形度 = 内点数 / (圆周上应有的像素数)
        circumference = 2 * np.pi * r
        inlier_circularity = n_inliers / circumference if circumference > 0 else 0
        # 截断到 [0, 1]
        inlier_circularity = min(1.0, inlier_circularity)

        if inlier_circularity < min_circularity:
            continue

        # RANSAC 已经过滤了离群点，综合得分
        score = inlier_circularity * flatness * n_inliers
        if score > best_score:
            best_score = score
            best = (cx, cy, r, inlier_circularity, flatness,
                    float(np.mean(depths)), float(depth_std), n_inliers)

    return best


# ═══════════════════════════════════════════════════════════════
# RGB 白色精修
# ═══════════════════════════════════════════════════════════════

def refine_white_disk(rgb, depth_result, margin, v_min, s_max):
    if depth_result is None:
        return None, None

    dcx, dcy, dr = depth_result[0], depth_result[1], depth_result[2]
    h, w = rgb.shape[:2]

    x1 = max(0, dcx - dr - margin)
    y1 = max(0, dcy - dr - margin)
    x2 = min(w, dcx + dr + margin)
    y2 = min(h, dcy + dr + margin)

    roi = rgb[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, v_min])
    upper = np.array([180, s_max, 255])
    white_mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)

    best_label = 0
    best_area = 0
    for i in range(1, n_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        if a > best_area:
            best_area = a
            best_label = i

    if best_label == 0:
        return None, white_mask

    comp_mask = (labels == best_label).astype(np.uint8)
    ys, xs = np.where(comp_mask)
    if len(xs) < 5:
        return None, white_mask

    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    # 白色区域也用 RANSAC 拟合，对杂乱背景鲁棒
    ransac_result = ransac_fit_circle(points, n_iter=100, inlier_thresh=5, min_inlier_ratio=0.25)
    if ransac_result is None:
        return None, white_mask
    fcx, fcy, fr, _ = ransac_result

    refined_cx = x1 + int(fcx)
    refined_cy = y1 + int(fcy)

    offset = np.sqrt((refined_cx - dcx) ** 2 + (refined_cy - dcy) ** 2)
    if offset > dr * 1.5:
        return None, white_mask

    return (refined_cx, refined_cy, int(fr), best_area), white_mask


def nothing(x):
    pass


# ═══════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════

def main():
    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开RGB摄像头")
        exit()

    cv2.namedWindow("Disk Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Disk Detection", 1280, 550)

    cv2.createTrackbar("DepthMin(mm)", "Disk Detection", 200, 2000, nothing)
    cv2.createTrackbar("DepthMax(mm)", "Disk Detection", 1000, 2000, nothing)
    cv2.createTrackbar("Min Radius", "Disk Detection", 20, 300, nothing)
    cv2.createTrackbar("Max Radius", "Disk Detection", 200, 400, nothing)
    cv2.createTrackbar("Circularity%", "Disk Detection", 25, 100, nothing)
    cv2.createTrackbar("Min Area", "Disk Detection", 500, 5000, nothing)
    cv2.createTrackbar("Kernel Size", "Disk Detection", 5, 21, nothing)
    cv2.createTrackbar("White VMin", "Disk Detection", 140, 255, nothing)
    cv2.createTrackbar("White SMax", "Disk Detection", 60, 255, nothing)
    cv2.createTrackbar("RGB Margin", "Disk Detection", 40, 100, nothing)

    print("q=退出 | 白色=深度RANSAC | 绿色=RGB白色精修 | 红色框=RGB搜索ROI")

    while True:
        ret, rgb = cap.read()
        if not ret or rgb is None:
            break

        frame = depth_stream.read_frame()
        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
        dpt2 *= 255
        dpt = dpt1 + dpt2
        dpt = dpt[:, ::-1]

        depth_min = cv2.getTrackbarPos("DepthMin(mm)", "Disk Detection")
        depth_max = cv2.getTrackbarPos("DepthMax(mm)", "Disk Detection")
        min_r = cv2.getTrackbarPos("Min Radius", "Disk Detection")
        max_r = cv2.getTrackbarPos("Max Radius", "Disk Detection")
        circ_pct = cv2.getTrackbarPos("Circularity%", "Disk Detection")
        min_area = cv2.getTrackbarPos("Min Area", "Disk Detection")
        kernel_sz = cv2.getTrackbarPos("Kernel Size", "Disk Detection")
        v_min = cv2.getTrackbarPos("White VMin", "Disk Detection")
        s_max = cv2.getTrackbarPos("White SMax", "Disk Detection")
        rgb_margin = cv2.getTrackbarPos("RGB Margin", "Disk Detection")
        if kernel_sz % 2 == 0:
            kernel_sz += 1

        # ── 深度检测 ──
        depth_result = detect_disk_from_depth(
            dpt, depth_min, depth_max, min_r, max_r,
            circ_pct / 100.0, min_area, kernel_sz)

        # ── RGB白色精修 ──
        refined, white_mask_roi = refine_white_disk(
            rgb, depth_result, rgb_margin, v_min, s_max)

        # ── 深度图彩色化 ──
        dpt_vis = dpt.copy()
        dpt_vis[dpt_vis > depth_max] = depth_max
        dpt_norm = cv2.normalize(dpt_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dpt_color = cv2.applyColorMap(dpt_norm, cv2.COLORMAP_JET)

        display = rgb.copy()

        # ── 绘制深度检测结果（白色） ──
        if depth_result is not None:
            dcx, dcy, dr = depth_result[0], depth_result[1], depth_result[2]
            cv2.circle(dpt_color, (dcx, dcy), dr, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(dpt_color, (dcx, dcy), (255, 255, 255),
                           cv2.MARKER_CROSS, 10, 2)

            cv2.circle(display, (dcx, dcy), dr, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(display, (dcx, dcy), (255, 255, 255),
                           cv2.MARKER_CROSS, 10, 2)

        # ── 绘制RGB精修结果（绿色） ──
        if refined is not None:
            rcx, rcy, rr, white_area = refined
            cv2.circle(display, (rcx, rcy), rr, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(display, (rcx, rcy), (0, 255, 0),
                           cv2.MARKER_CROSS, 15, 2)

            if depth_result is not None:
                # RGB-Depth偏移线
                offset = np.sqrt((rcx - dcx) ** 2 + (rcy - dcy) ** 2)
                cv2.line(display, (dcx, dcy), (rcx, rcy), (0, 0, 255), 1)
                cv2.putText(display, f"RGB white px={white_area} offset={offset:.1f}",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 白色检测区域半透明覆盖
            if white_mask_roi is not None and depth_result is not None:
                x1 = max(0, dcx - dr - rgb_margin)
                y1 = max(0, dcy - dr - rgb_margin)
                x2 = min(display.shape[1], dcx + dr + rgb_margin)
                y2 = min(display.shape[0], dcy + dr + rgb_margin)
                full = np.zeros(rgb.shape[:2], dtype=np.uint8)
                full[y1:y2, x1:x2] = white_mask_roi
                overlay = np.zeros_like(display)
                overlay[full > 0] = (0, 255, 0)
                display = cv2.addWeighted(display, 1.0, overlay, 0.2, 0)

        # ── RGB搜索ROI红色虚线框 ──
        if depth_result is not None:
            x1 = max(0, dcx - dr - rgb_margin)
            y1 = max(0, dcy - dr - rgb_margin)
            x2 = min(display.shape[1], dcx + dr + rgb_margin)
            y2 = min(display.shape[0], dcy + dr + rgb_margin)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

        if depth_result is None:
            cv2.putText(display, "NO DISK", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        combined = np.hstack([dpt_color, display])
        cv2.imshow("Disk Detection", combined)

        if cv2.waitKey(1) == ord('q'):
            break

    depth_stream.stop()
    dev.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
