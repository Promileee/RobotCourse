import cv2
import numpy as np
import math
from openni import openni2

# ==========================================
# 严格的几何约束阈值 (像素)
# ==========================================
TOL_SQUARE = 15       # 靶标长宽差最大允许值
TOL_SIZE = 15         # 三个靶标最大和最小尺寸差值
TOL_TRIANGLE = 20     # 等边三角形三边最大差值
MASK_SCALE = 1.25     # 外接圆放缩比例 (用于包住整个靶标以提取特征)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 形态学去噪核 (加大以去除更多噪点)
MEDIAN_KSIZE = 5      # 自适应阈值前的中值滤波核大小
MAX_FRAME_SHIFT = 10  # 相邻帧靶标点最大位移 (像素)
MAX_FRAME_SHIFT_RELAX = 20  # 上一帧被约束时放宽的阈值
TOL_TRIANGLE_TRACK = 8      # 追踪阶段正三角形边长最大差值 (像素)
MAX_TRI_FAIL = 7            # 连续不满足正三角形约束的最大帧数, 超限则重新初始化

def get_distance(p1, p2):
    """计算两点之间的欧式距离"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def extract_angular_signature(binary_img, center, radius):
    """从圆形ROI中提取极坐标角度签名 (360维向量, 用于旋转估计)"""
    cx, cy = center
    r = radius

    # 遮罩圆形 + 裁剪
    circle_mask = np.zeros(binary_img.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), r, 255, -1)
    masked = cv2.bitwise_and(binary_img, circle_mask)

    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(binary_img.shape[1], cx + r), min(binary_img.shape[0], cy + r)
    crop = masked[y1:y2, x1:x2]

    # 放入正方形画布 (圆心在正中心)
    side = r * 2
    canvas = np.zeros((side, side), dtype=np.uint8)
    ox, oy = cx - r, cy - r  # 画布原点在原始图中的坐标
    canvas[max(0, y1 - oy):min(side, y2 - oy), max(0, x1 - ox):min(side, x2 - ox)] = crop

    # 极坐标变换: 宽度=360(角度), 高度=r(半径)
    polar = cv2.warpPolar(canvas, (360, r), (r, r), r, cv2.WARP_POLAR_LINEAR)

    # 沿径向求和 → 360维角度签名, 归一化
    sig = np.sum(polar, axis=0).astype(np.float32)
    sig = sig / (np.linalg.norm(sig) + 1e-8)
    return sig

def estimate_rotation_angle(sig_ref, sig_cur):
    """互相关估计旋转角度, 返回角度(度)"""
    corr = np.fft.ifft(np.fft.fft(sig_ref) * np.conj(np.fft.fft(sig_cur))).real
    lag = np.argmax(corr)
    if lag > 180:
        lag -= 360
    return float(-lag)

# ==========================================
# 1. 初始化相机
# ==========================================
print("正在初始化相机...")
openni2.initialize()
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开RGB摄像头")
    exit()

print("正在跳过前5帧以稳定传感器...")
for _ in range(5):
    cap.read()
    depth_stream.read_frame()
    cv2.waitKey(100)

# ==========================================
# 外层循环: 追踪失败时自动重新初始化
# ==========================================
quit_requested = False

while not quit_requested:

    print("\n开始执行严苛的初始帧捕获逻辑...")
    confirmed_targets = None
    ref_frame = None
    frame_count = 0

    # ==========================================
    # PHASE 1: 严苛的初始化捕获
    # ==========================================
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        depth_frame = depth_stream.read_frame()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 100 < w * h < 40000:
                bboxes.append((x + w // 2, y + h // 2, x, y, w, h))

        clusters = []
        DIST_THRESH = 15
        for box in bboxes:
            added = False
            for cluster in clusters:
                if abs(box[0] - cluster[0][0]) < DIST_THRESH and abs(box[1] - cluster[0][1]) < DIST_THRESH:
                    cluster.append(box)
                    added = True
                    break
            if not added:
                clusters.append([box])

        target_candidates = []
        for cluster in clusters:
            if len(cluster) >= 4:
                min_x = min(b[2] for b in cluster)
                min_y = min(b[3] for b in cluster)
                max_r = max(b[2] + b[4] for b in cluster)
                max_b = max(b[3] + b[5] for b in cluster)
                tw, th = max_r - min_x, max_b - min_y
                if 0.7 < (float(tw)/th if th>0 else 0) < 1.4 and 2000 < tw * th < 25000:
                    tcx, tcy = min_x + tw // 2, min_y + th // 2
                    target_candidates.append((min_x, min_y, tw, th, tcx, tcy))

        final_targets = []
        target_candidates.sort(key=lambda b: b[2]*b[3], reverse=True)
        for cand in target_candidates:
            tcx, tcy = cand[4], cand[5]
            if not any(ft[0] < tcx < ft[0]+ft[2] and ft[1] < tcy < ft[1]+ft[3] for ft in final_targets):
                final_targets.append(cand)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        identified = {}

        for x, y, w, h, cx, cy in final_targets:
            # 防止越界
            y1, y2 = max(0, y), min(frame.shape[0], y+h)
            x1, x2 = max(0, x), min(frame.shape[1], x+w)
            roi_hsv = hsv[y1:y2, x1:x2]
            if roi_hsv.size == 0: continue

            mask_r = cv2.bitwise_or(
                cv2.inRange(roi_hsv, np.array([0, 50, 50]), np.array([15, 255, 255])),
                cv2.inRange(roi_hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
            )
            mask_g = cv2.inRange(roi_hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
            mask_b = cv2.inRange(roi_hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))

            counts = {"Red": cv2.countNonZero(mask_r), "Green": cv2.countNonZero(mask_g), "Blue": cv2.countNonZero(mask_b)}
            best_color = max(counts, key=counts.get)

            if counts[best_color] > 100:
                identified[best_color] = (x, y, w, h, cx, cy)

        preview_img = frame.copy()
        cv2.putText(preview_img, f"Searching... Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Initialization", preview_img)

        # 展示二值图 (形态学去噪后), 供观察效果
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL)
        cv2.imshow("Init Binary", binary_clean)
        cv2.waitKey(1)

        # --- 严苛校验 ---
        if len(identified) != 3 or not all(c in identified for c in ["Red", "Green", "Blue"]):
            continue

        tr, tg, tb = identified["Red"], identified["Green"], identified["Blue"]

        max_sq_diff = max(abs(tr[2]-tr[3]), abs(tg[2]-tg[3]), abs(tb[2]-tb[3]))
        if max_sq_diff > TOL_SQUARE: continue

        ws, hs = [tr[2], tg[2], tb[2]], [tr[3], tg[3], tb[3]]
        max_size_diff = max(max(ws)-min(ws), max(hs)-min(hs))
        if max_size_diff > TOL_SIZE: continue

        d_rg = get_distance((tr[4], tr[5]), (tg[4], tg[5]))
        d_gb = get_distance((tg[4], tg[5]), (tb[4], tb[5]))
        d_br = get_distance((tb[4], tb[5]), (tr[4], tr[5]))
        max_tri_diff = max(d_rg, d_gb, d_br) - min(d_rg, d_gb, d_br)
        if max_tri_diff > TOL_TRIANGLE: continue

        print("\n✅✅✅ 所有严苛几何条件均已满足！初始化捕获成功！ ✅✅✅")
        confirmed_targets = identified
        ref_frame = frame.copy()
        break  # 跳出初始化循环

    cv2.destroyWindow("Initialization")
    cv2.destroyWindow("Init Binary")

    # ==========================================
    # PHASE 2: 创建 ORB 星座模板 (使用二值化图像)
    # ==========================================
    print("正在构建星座模板与特征库...")
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 提取三个靶标中心点，计算最小外接圆
    pts = np.array([[data[4], data[5]] for _, data in confirmed_targets.items()], dtype=np.float32)
    (center_x, center_y), strict_radius = cv2.minEnclosingCircle(pts)
    ref_mask_center = (int(center_x), int(center_y))
    ref_mask_radius = int(strict_radius * MASK_SCALE)

    # 生成模板遮罩
    mask = np.zeros(ref_frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, ref_mask_center, ref_mask_radius, 255, -1)

    # 提取参考特征 (二值化 + 中值滤波 + 形态学去噪)
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.medianBlur(ref_gray, MEDIAN_KSIZE)
    ref_binary = cv2.adaptiveThreshold(ref_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ref_binary = cv2.morphologyEx(ref_binary, cv2.MORPH_OPEN, KERNEL)
    ref_kp, ref_des = orb.detectAndCompute(ref_binary, mask)
    ref_targets_centers = {color: (data[4], data[5]) for color, data in confirmed_targets.items()}

    # 提取参考角度签名 (用于极坐标旋转估计)
    ref_angular_sig = extract_angular_signature(ref_binary, ref_mask_center, ref_mask_radius)
    print(f"模板构建完毕，共提取 {len(ref_kp)} 个特征点。进入实时追踪！\n")

    # ==========================================
    # PHASE 3: ORB + 极坐标旋转 实时刚体追踪
    # ==========================================
    colors_bgr = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}
    current_targets = ref_targets_centers.copy()
    prev_targets = ref_targets_centers.copy()
    was_constrained = False  # 上一帧是否因位移约束被跳过, 用于自适应放宽
    consecutive_tri_fail = 0  # 连续不满足正三角形约束的帧数

    while True:
        ret, frame = cap.read()
        if not ret: break

        depth_frame = depth_stream.read_frame()
        dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        depth_map = (np.asarray(dframe_data[:, :, 0], dtype="float32") +
                     np.asarray(dframe_data[:, :, 1], dtype="float32") * 255)[:, ::-1]

        result_img = frame.copy()

        # 提取当前帧特征 (二值化 + 中值滤波 + 形态学去噪)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, MEDIAN_KSIZE)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL)

        # --- 极坐标旋转估计 ---
        cur_angular_sig = extract_angular_signature(binary, ref_mask_center, ref_mask_radius)
        rot_angle = estimate_rotation_angle(ref_angular_sig, cur_angular_sig)

        # --- ORB 特征匹配 + 仿射变换 ---
        kp, des = orb.detectAndCompute(binary, mask)

        tracking_success = False
        if des is not None and len(des) > 10:
            matches = matcher.match(ref_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:max(10, int(len(matches) * 0.2))]

            if len(good_matches) >= 10:
                src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC)

                if M is not None:
                    # 用算出的矩阵计算候选新位置
                    new_targets = {}
                    for color, (orig_cx, orig_cy) in ref_targets_centers.items():
                        nx = M[0, 0] * orig_cx + M[0, 1] * orig_cy + M[0, 2]
                        ny = M[1, 0] * orig_cx + M[1, 1] * orig_cy + M[1, 2]
                        new_targets[color] = (int(nx), int(ny))

                    # 约束1: 帧间位移限制, 上一帧若被约束则本帧自动放宽
                    effective_max_shift = MAX_FRAME_SHIFT_RELAX if was_constrained else MAX_FRAME_SHIFT
                    frame_constrained = False
                    for color in new_targets:
                        px, py = prev_targets[color]
                        nx, ny = new_targets[color]
                        if get_distance((px, py), (nx, ny)) > effective_max_shift:
                            new_targets[color] = (px, py)
                            frame_constrained = True
                    was_constrained = frame_constrained

                    # 约束2: 靶标点必须在固定圈内
                    mc_x, mc_y = ref_mask_center
                    for color in new_targets:
                        nx, ny = new_targets[color]
                        if get_distance((mc_x, mc_y), (nx, ny)) > ref_mask_radius * 0.85:
                            # 超界则沿径向拉回到圈内
                            dx, dy = nx - mc_x, ny - mc_y
                            dist = math.hypot(dx, dy) + 1e-8
                            scale = (ref_mask_radius * 0.85) / dist
                            new_targets[color] = (int(mc_x + dx * scale), int(mc_y + dy * scale))

                    tracking_success = True
                    current_targets = new_targets

        if tracking_success:
            # 约束3: 三个靶标应构成近似正三角形
            target_pts = [(cx, cy) for cx, cy in current_targets.values()]
            if len(target_pts) == 3:
                d01 = get_distance(target_pts[0], target_pts[1])
                d12 = get_distance(target_pts[1], target_pts[2])
                d20 = get_distance(target_pts[2], target_pts[0])
                tri_diff = max(d01, d12, d20) - min(d01, d12, d20)
                if tri_diff > TOL_TRIANGLE_TRACK:
                    consecutive_tri_fail += 1
                    if consecutive_tri_fail >= MAX_TRI_FAIL:
                        print(f"\n连续 {MAX_TRI_FAIL} 帧不满足正三角形约束 (边长差 {tri_diff:.1f}px), 重新初始化...\n")
                        break  # 跳出追踪循环, 回到 Phase 1
                else:
                    consecutive_tri_fail = 0

            # 验证: 仿射旋转角 vs 极坐标旋转角
            affine_angle = math.degrees(math.atan2(M[1, 0], M[0, 0])) if M is not None else 0.0
            status = f"TRACKING | Rot: {rot_angle:.1f}deg (affine: {affine_angle:.1f}deg)"
            if consecutive_tri_fail > 0:
                status += f" | TriFail:{consecutive_tri_fail}"
            cv2.putText(result_img, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            pts = []
            for color, (cx, cy) in current_targets.items():
                bgr = colors_bgr[color]
                pts.append((cx, cy))
                cv2.circle(result_img, (cx, cy), 15, bgr, 3)
                cv2.circle(result_img, (cx, cy), 4, (255, 255, 255), -1)

                d_val = depth_map[cy, cx] if (0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]) else 0
                cv2.putText(result_img, f"{color} {d_val:.0f}mm", (cx-30, cy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

            # 绘制固定观察圈
            cv2.circle(result_img, ref_mask_center, ref_mask_radius, (0, 255, 255), 1, cv2.LINE_AA)
            # 绘制圈内半径约束边界
            cv2.circle(result_img, ref_mask_center, int(ref_mask_radius * 0.85), (0, 180, 180), 1, cv2.LINE_AA)

            if len(pts) == 3:
                cv2.polylines(result_img, [np.array(pts)], isClosed=True, color=(255, 255, 255), thickness=1)

            prev_targets = current_targets.copy()
        else:
            cv2.putText(result_img, "TRACKING LOST! (Keep still or return to frame)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 可视化: 仅展示特征计算圈内的二值化内容，其余遮罩为黑色
        binary_masked = np.zeros_like(binary)
        cv2.circle(binary_masked, ref_mask_center, ref_mask_radius, 255, -1)
        binary_masked = cv2.bitwise_and(binary, binary_masked)
        cv2.imshow("Constellation Tracking", result_img)
        cv2.imshow("Tracking Binary (Masked)", binary_masked)
        if cv2.waitKey(1) == ord('q'):
            quit_requested = True
            break

cap.release()
depth_stream.stop()
dev.close()
cv2.destroyAllWindows()
