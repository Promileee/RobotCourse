"""
靶标识别与追踪模块 (加入高鲁棒性时域滤波)
基于ORB特征匹配和极坐标旋转估计的刚体追踪系统。
结合了 EMA 平滑与异常帧剔除，保障电机闭环控制的绝对稳定性。

使用方法:
    from camera_manager import CameraManager
    from target_recognize import TargetRecognizer

    cam = CameraManager()
    cam.start()
    tr = TargetRecognizer()
    tr.setup(cam)
    tr.run_full_pipeline()
"""

import cv2
import numpy as np
import math
from backup_2026_06_05.camera_manager import CameraManager


# ==========================================
# 几何与追踪约束阈值
# ==========================================
TOL_SQUARE = 15
TOL_SIZE = 15
TOL_TRIANGLE = 20
MASK_SCALE = 1.25
MEDIAN_KSIZE = 5
TOL_TRIANGLE_TRACK = 8
MAX_TRI_FAIL = 7


def get_distance(p1, p2):
    """计算两点之间的欧式距离"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def extract_angular_signature(binary_img, center, radius):
    """从圆形ROI中提取极坐标角度签名 (360维向量, 用于旋转估计)"""
    cx, cy = center
    r = radius

    circle_mask = np.zeros(binary_img.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), r, 255, -1)
    masked = cv2.bitwise_and(binary_img, circle_mask)

    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(binary_img.shape[1], cx + r), min(binary_img.shape[0], cy + r)
    crop = masked[y1:y2, x1:x2]

    side = r * 2
    canvas = np.zeros((side, side), dtype=np.uint8)
    ox, oy = cx - r, cy - r
    canvas[max(0, y1 - oy):min(side, y2 - oy), max(0, x1 - ox):min(side, x2 - ox)] = crop

    polar = cv2.warpPolar(canvas, (360, r), (r, r), r, cv2.WARP_POLAR_LINEAR)
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


class ConstellationTemporalTracker:
    """星座时域追踪器：EMA平滑 + 刚体异常帧剔除"""
    
    def __init__(self, alpha=0.35, max_jump=40):
        """
        :param alpha: 平滑系数 (0~1)。电机控制建议取 0.3~0.5 之间，平衡平滑度与响应延迟。
        :param max_jump: 允许的最大单帧像素跳跃距离，超过即判定特征匹配崩溃，剔除该帧。
        """
        self.alpha = alpha
        self.max_jump = max_jump
        self.history = None

    def update(self, measurements):
        """输入当前帧三个靶标的测量坐标，输出滤波后的稳定坐标"""
        if measurements is None:
            return None

        # 第一帧初始化
        if self.history is None:
            self.history = measurements.copy()
            return self.history.copy()

        filtered = {}
        is_outlier = False

        # 1. 异常点剔除逻辑
        # 因为三个靶标是刚体星座，如果其中任何一个点发生瞬时漂移(大概率是矩阵解算错乱)，
        # 整体应当被判定为异常帧，直接沿用历史稳定位置。
        for color, (cx, cy) in measurements.items():
            hx, hy = self.history[color]
            if get_distance((cx, cy), (hx, hy)) > self.max_jump:
                is_outlier = True
                break

        if is_outlier:
            return self.history.copy()

        # 2. 指数移动平均 (EMA) 平滑
        for color, (cx, cy) in measurements.items():
            hx, hy = self.history[color]
            fx = self.alpha * cx + (1 - self.alpha) * hx
            fy = self.alpha * cy + (1 - self.alpha) * hy
            filtered[color] = (int(fx), int(fy))

        # 更新历史并返回
        self.history = filtered.copy()
        return filtered

    def reset(self):
        """重置历史状态"""
        self.history = None


class TargetRecognizer:
    """靶标识别与追踪器"""

    def __init__(self):
        self.cam = None
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 模板数据
        self.ref_frame = None
        self.ref_binary = None
        self.ref_kp = None
        self.ref_des = None
        self.ref_targets_centers = None
        self.ref_mask_center = None
        self.ref_mask_radius = None
        self.ref_angular_sig = None
        self.mask = None

        # HSV颜色范围与绘图颜色
        self.hsv_ranges = {
            "Red": [(np.array([0, 50, 50]), np.array([15, 255, 255])),
                    (np.array([160, 50, 50]), np.array([180, 255, 255]))],
            "Green": [(np.array([40, 50, 50]), np.array([90, 255, 255]))],
            "Blue": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
        }
        self.colors_bgr = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}

        # 追踪状态与时域滤波器
        self.current_targets = None
        self.consecutive_tri_fail = 0
        self.depth_map = None
        self.tracker = ConstellationTemporalTracker(alpha=0.35, max_jump=40)

    # ==========================================
    # 相机管理
    # ==========================================
    def setup(self, camera_manager=None):
        if self.cam is not None:
            return 
        if camera_manager is None:
            self.cam = CameraManager()
            self.cam.start()
        else:
            self.cam = camera_manager

    def read_rgb_frame(self):
        return self.cam.read_rgb_frame()

    def get_depth_map(self):
        return self.cam.get_depth_map()

    # ==========================================
    # 图像预处理
    # ==========================================
    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, MEDIAN_KSIZE)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        return gray, binary

    def preprocess_edges(self, gray):
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.Canny(blurred, 40, 120)

    # ==========================================
    # Phase 1: 严苛初始化捕获
    # ==========================================
    def find_target_candidates(self, edges):
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
                if 0.7 < (float(tw) / th if th > 0 else 0) < 1.4 and 2000 < tw * th < 25000:
                    tcx, tcy = min_x + tw // 2, min_y + th // 2
                    target_candidates.append((min_x, min_y, tw, th, tcx, tcy))

        final_targets = []
        target_candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
        for cand in target_candidates:
            tcx, tcy = cand[4], cand[5]
            if not any(ft[0] < tcx < ft[0] + ft[2] and ft[1] < tcy < ft[1] + ft[3]
                       for ft in final_targets):
                final_targets.append(cand)

        return final_targets

    def identify_colors(self, frame, final_targets):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        identified = {}

        for x, y, w, h, cx, cy in final_targets:
            y1, y2 = max(0, y), min(frame.shape[0], y + h)
            x1, x2 = max(0, x), min(frame.shape[1], x + w)
            roi_hsv = hsv[y1:y2, x1:x2]
            if roi_hsv.size == 0:
                continue

            counts = {}
            for color_name, ranges in self.hsv_ranges.items():
                mask = cv2.inRange(roi_hsv, ranges[0][0], ranges[0][1])
                for rng in ranges[1:]:
                    mask = cv2.bitwise_or(mask, cv2.inRange(roi_hsv, rng[0], rng[1]))
                counts[color_name] = cv2.countNonZero(mask)

            best_color = max(counts, key=counts.get)
            if counts[best_color] > 100:
                identified[best_color] = (x, y, w, h, cx, cy)

        return identified

    def validate_geometry(self, identified):
        if len(identified) != 3 or not all(c in identified for c in ["Red", "Green", "Blue"]):
            return False

        tr, tg, tb = identified["Red"], identified["Green"], identified["Blue"]

        max_sq_diff = max(abs(tr[2] - tr[3]), abs(tg[2] - tg[3]), abs(tb[2] - tb[3]))
        if max_sq_diff > TOL_SQUARE:
            return False

        ws, hs = [tr[2], tg[2], tb[2]], [tr[3], tg[3], tb[3]]
        max_size_diff = max(max(ws) - min(ws), max(hs) - min(hs))
        if max_size_diff > TOL_SIZE:
            return False

        d_rg = get_distance((tr[4], tr[5]), (tg[4], tg[5]))
        d_gb = get_distance((tg[4], tg[5]), (tb[4], tb[5]))
        d_br = get_distance((tb[4], tb[5]), (tr[4], tr[5]))
        max_tri_diff = max(d_rg, d_gb, d_br) - min(d_rg, d_gb, d_br)
        if max_tri_diff > TOL_TRIANGLE:
            return False

        return True

    def capture_initial_targets(self):
        print("\n开始执行严苛的初始帧捕获逻辑...")
        cv2.namedWindow("Initialization")

        frame_count = 0
        while True:
            frame = self.read_rgb_frame()
            if frame is None:
                continue

            frame_count += 1
            _ = self.get_depth_map()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            edges = self.preprocess_edges(gray)

            final_targets = self.find_target_candidates(edges)
            identified = self.identify_colors(frame, final_targets)

            preview_img = frame.copy()
            cv2.putText(preview_img, f"Searching... Frame: {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Initialization", preview_img)

            binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
            cv2.imshow("Init Binary", binary_clean)
            cv2.waitKey(1)

            if not self.validate_geometry(identified):
                continue

            print("\n所有严苛几何条件均已满足！初始化捕获成功！")
            cv2.destroyWindow("Initialization")
            cv2.destroyWindow("Init Binary")
            return frame, identified

    # ==========================================
    # Phase 2: 构建ORB星座模板
    # ==========================================
    def build_template(self, frame, confirmed_targets):
        print("正在构建星座模板与特征库...")
        self.ref_frame = frame.copy()

        pts = np.array([[data[4], data[5]] for _, data in confirmed_targets.items()], dtype=np.float32)
        (center_x, center_y), strict_radius = cv2.minEnclosingCircle(pts)
        self.ref_mask_center = (int(center_x), int(center_y))
        self.ref_mask_radius = int(strict_radius * MASK_SCALE)

        self.mask = np.zeros(self.ref_frame.shape[:2], dtype=np.uint8)
        cv2.circle(self.mask, self.ref_mask_center, self.ref_mask_radius, 255, -1)

        ref_gray = cv2.cvtColor(self.ref_frame, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.medianBlur(ref_gray, MEDIAN_KSIZE)
        self.ref_binary = cv2.adaptiveThreshold(ref_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
        self.ref_binary = cv2.morphologyEx(self.ref_binary, cv2.MORPH_OPEN, self.kernel)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self.ref_binary, self.mask)
        self.ref_targets_centers = {color: (data[4], data[5])
                                     for color, data in confirmed_targets.items()}

        self.ref_angular_sig = extract_angular_signature(self.ref_binary, self.ref_mask_center,
                                                          self.ref_mask_radius)

        # 重置时域滤波器并注入初始值
        self.tracker.reset()
        self.current_targets = self.tracker.update(self.ref_targets_centers)
        self.consecutive_tri_fail = 0

        print(f"模板构建完毕，共提取 {len(self.ref_kp)} 个特征点。")
        return len(self.ref_kp)

    # ==========================================
    # Phase 3: 实时追踪
    # ==========================================
    def track_frame(self, frame, binary):
        result_img = frame.copy()
        cur_angular_sig = extract_angular_signature(binary, self.ref_mask_center, self.ref_mask_radius)
        rot_angle = estimate_rotation_angle(self.ref_angular_sig, cur_angular_sig)

        kp, des = self.orb.detectAndCompute(binary, self.mask)
        tracking_success = False
        M = None

        if des is not None and len(des) > 10:
            matches = self.matcher.match(self.ref_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:max(10, int(len(matches) * 0.2))]

            if len(good_matches) >= 10:
                src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC)

                if M is not None:
                    raw_new_targets = {}
                    for color, (orig_cx, orig_cy) in self.ref_targets_centers.items():
                        nx = M[0, 0] * orig_cx + M[0, 1] * orig_cy + M[0, 2]
                        ny = M[1, 0] * orig_cx + M[1, 1] * orig_cy + M[1, 2]
                        raw_new_targets[color] = (int(nx), int(ny))

                    # 基础的物理边界约束：防止坐标溢出计算圈
                    mc_x, mc_y = self.ref_mask_center
                    for color in raw_new_targets:
                        nx, ny = raw_new_targets[color]
                        if get_distance((mc_x, mc_y), (nx, ny)) > self.ref_mask_radius * 0.85:
                            dx, dy = nx - mc_x, ny - mc_y
                            dist = math.hypot(dx, dy) + 1e-8
                            scale = (self.ref_mask_radius * 0.85) / dist
                            raw_new_targets[color] = (int(mc_x + dx * scale), int(mc_y + dy * scale))

                    # === 引入核心的时域滤波器，替代原先简单的帧间约束 ===
                    self.current_targets = self.tracker.update(raw_new_targets)
                    tracking_success = True

        # 正三角形约束 (基于滤波后的平滑坐标进行判断)
        if tracking_success:
            target_pts = [(cx, cy) for cx, cy in self.current_targets.values()]
            if len(target_pts) == 3:
                d01 = get_distance(target_pts[0], target_pts[1])
                d12 = get_distance(target_pts[1], target_pts[2])
                d20 = get_distance(target_pts[2], target_pts[0])
                tri_diff = max(d01, d12, d20) - min(d01, d12, d20)
                
                if tri_diff > TOL_TRIANGLE_TRACK:
                    self.consecutive_tri_fail += 1
                    if self.consecutive_tri_fail >= MAX_TRI_FAIL:
                        return False, None, result_img, rot_angle, None  # 需要重新初始化
                else:
                    self.consecutive_tri_fail = 0

        affine_angle = math.degrees(math.atan2(M[1, 0], M[0, 0])) if M is not None else 0.0

        # 绘制结果
        if tracking_success:
            status = f"TRACKING | Rot: {rot_angle:.1f}deg (affine: {affine_angle:.1f}deg)"
            if self.consecutive_tri_fail > 0:
                status += f" | TriFail:{self.consecutive_tri_fail}"
            cv2.putText(result_img, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            pts = []
            for color, (cx, cy) in self.current_targets.items():
                bgr = self.colors_bgr[color]
                pts.append((cx, cy))
                cv2.circle(result_img, (cx, cy), 15, bgr, 3)
                cv2.circle(result_img, (cx, cy), 4, (255, 255, 255), -1)

                d_val = (self.depth_map[cy, cx] if self.depth_map is not None and
                         0 <= cy < self.depth_map.shape[0] and
                         0 <= cx < self.depth_map.shape[1] else 0)
                cv2.putText(result_img, f"{color} {d_val:.0f}mm", (cx - 30, cy - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

            cv2.circle(result_img, self.ref_mask_center, self.ref_mask_radius,
                       (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(result_img, self.ref_mask_center, int(self.ref_mask_radius * 0.85),
                       (0, 180, 180), 1, cv2.LINE_AA)

            if len(pts) == 3:
                cv2.polylines(result_img, [np.array(pts)], isClosed=True,
                              color=(255, 255, 255), thickness=1)
        else:
            cv2.putText(result_img, "TRACKING LOST!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return tracking_success, self.current_targets, result_img, rot_angle, affine_angle

    def track_loop(self):
        print("进入实时追踪！按 q 退出。\n")
        cv2.namedWindow("Constellation Tracking")
        cv2.namedWindow("Tracking Binary (Masked)")

        while True:
            frame = self.read_rgb_frame()
            if frame is None:
                break

            self.depth_map = self.get_depth_map()
            _, binary = self.preprocess(frame)

            success, targets, result_img, rot_angle, affine_angle = self.track_frame(frame, binary)

            if not success and self.consecutive_tri_fail >= MAX_TRI_FAIL:
                print(f"\n连续 {MAX_TRI_FAIL} 帧不满足正三角形约束，追踪失败。")
                cv2.destroyAllWindows()
                return False

            binary_masked = np.zeros_like(binary)
            cv2.circle(binary_masked, self.ref_mask_center, self.ref_mask_radius, 255, -1)
            binary_masked = cv2.bitwise_and(binary, binary_masked)

            cv2.imshow("Constellation Tracking", result_img)
            cv2.imshow("Tracking Binary (Masked)", binary_masked)
            if cv2.waitKey(1) == ord("q"):
                return True 

        return False

    # ==========================================
    # 便捷方法：自动完成全套流程
    # ==========================================
    def run_full_pipeline(self):
        self.setup()

        while True:
            frame, targets = self.capture_initial_targets()
            if frame is None:
                break

            self.build_template(frame, targets)

            user_quit = self.track_loop()
            if user_quit:
                break

        self.release()

    def release(self):
        if self.cam is not None:
            self.cam.release()
            self.cam = None


# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    cam = CameraManager()
    cam.start()

    tr = TargetRecognizer()
    tr.setup(cam)
    tr.run_full_pipeline()