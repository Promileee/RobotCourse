"""
抗干扰整合靶标追踪模块 (Integrated Robust Tracker)
融合了“静态圆柱体黑名单剔除”与“刚体星座时域滤波追踪”。
利用圆柱体的绝对物理位置生成 Exclusion Mask，彻底免疫静态障碍物的反光与纹理干扰。

使用方法:
    from camera_manager import CameraManager
    from integrated_tracker import IntegratedTracker

    cam = CameraManager()
    cam.start()
    
    tracker = IntegratedTracker()
    tracker.setup(cam)
    tracker.run_full_pipeline()
"""

import cv2
import numpy as np
import math
from backup_2026_06_05.camera_manager import CameraManager

# ==========================================
# 参数配置
# ==========================================
# 1. 圆柱体识别参数 (生成黑名单用)
CYLINDER_DEPTH_MAX = 1300
CYLINDER_DEPTH_RATIO = 0.2
CYLINDER_MIN_AREA = 500
MASK_EXPAND = 15  # 黑名单遮罩向外膨胀的像素，彻底屏蔽边缘反光

# 圆柱体的 HSV 颜色范围 (参考原 rect_recognize.py)
CYLINDER_COLORS = {
    "Red": [(np.array([0, 100, 50]), np.array([15, 255, 255])),
            (np.array([160, 100, 50]), np.array([180, 255, 255]))],
    "Blue": [(np.array([100, 120, 70]), np.array([130, 255, 255]))],
    "Green": [(np.array([40, 70, 70]), np.array([80, 255, 255]))],
}

# 2. 靶标星座追踪参数 (参考原 target_recognize.py)
TARGET_HSV_RANGES = {
    "Red": [(np.array([0, 50, 50]), np.array([15, 255, 255])),
            (np.array([160, 50, 50]), np.array([180, 255, 255]))],
    "Green": [(np.array([40, 50, 50]), np.array([90, 255, 255]))],
    "Blue": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
}
TARGET_COLORS_BGR = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}

TOL_SQUARE = 15
TOL_SIZE = 15
TOL_TRIANGLE = 20
MASK_SCALE = 1.25
MEDIAN_KSIZE = 5
TOL_TRIANGLE_TRACK = 8
MAX_TRI_FAIL = 7


def get_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def extract_angular_signature(binary_img, center, radius):
    """提取极坐标角度签名 (保持不变)"""
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
    corr = np.fft.ifft(np.fft.fft(sig_ref) * np.conj(np.fft.fft(sig_cur))).real
    lag = np.argmax(corr)
    if lag > 180: lag -= 360
    return float(-lag)


class ConstellationTemporalTracker:
    """星座时域追踪器：EMA平滑 + 刚体异常帧剔除"""
    def __init__(self, alpha=0.35, max_jump=40):
        self.alpha = alpha
        self.max_jump = max_jump
        self.history = None

    def update(self, measurements):
        if measurements is None: return None
        if self.history is None:
            self.history = measurements.copy()
            return self.history.copy()

        filtered = {}
        # 异常跳跃检测
        for color, (cx, cy) in measurements.items():
            hx, hy = self.history[color]
            if get_distance((cx, cy), (hx, hy)) > self.max_jump:
                return self.history.copy()

        # EMA 平滑
        for color, (cx, cy) in measurements.items():
            hx, hy = self.history[color]
            filtered[color] = (int(self.alpha * cx + (1 - self.alpha) * hx),
                               int(self.alpha * cy + (1 - self.alpha) * hy))
        self.history = filtered.copy()
        return filtered

    def reset(self):
        self.history = None


class IntegratedTracker:
    """抗干扰整合追踪器"""
    def __init__(self):
        self.cam = None
        # ORB 特征提取器
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.temporal_tracker = ConstellationTemporalTracker(alpha=0.35, max_jump=40)

        # 核心：物理黑名单遮罩 (0为排除区域，255为保留区域)
        self.exclusion_mask = None

        # 靶标模板状态
        self.ref_frame = None
        self.ref_binary = None
        self.ref_kp = None
        self.ref_des = None
        self.ref_targets_centers = None
        self.ref_mask_center = None
        self.ref_mask_radius = None
        self.ref_angular_sig = None
        self.target_search_mask = None  # ORB 用的有效特征搜寻圈
        
        self.current_targets = None
        self.consecutive_tri_fail = 0
        self.depth_map = None

    def setup(self, camera_manager=None):
        if self.cam is not None: return
        if camera_manager is None:
            self.cam = CameraManager()
            self.cam.start()
        else:
            self.cam = camera_manager

    # ==========================================
    # Phase 1: 圆柱体环境标定 (生成黑名单遮罩)
    # ==========================================
    def _create_color_mask(self, hsv, ranges):
        mask = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
        if len(ranges) > 1:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, ranges[1][0], ranges[1][1]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def calibrate_environment(self):
        """阻塞循环，基于深度和颜色识别三个圆柱体，生成全局物理黑名单遮罩"""
        print("\n[Phase 1] 正在锁定圆柱体绝对位置，生成黑名单遮罩...")
        cv2.namedWindow("Environment Calibration")
        history = []
        stable_frames_required = 5

        while True:
            frame = self.cam.read_rgb_frame()
            if frame is None: continue
            
            dpt = self.cam.get_depth_map()
            depth_mask = np.where((dpt > 0) & (dpt < CYLINDER_DEPTH_MAX), 255, 0).astype(np.uint8)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            current_rects = {}
            vis = frame.copy()

            for name, ranges in CYLINDER_COLORS.items():
                c_mask = self._create_color_mask(hsv, ranges)
                contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                best_area, best_rect = 0, None
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < CYLINDER_MIN_AREA: continue
                    
                    x, y, w, h = cv2.boundingRect(cnt)
                    x1, y1 = max(x, 0), max(y, 0)
                    x2, y2 = min(x+w, c_mask.shape[1]), min(y+h, c_mask.shape[0])
                    
                    roi_color = c_mask[y1:y2, x1:x2]
                    roi_depth = depth_mask[y1:y2, x1:x2]
                    color_px = cv2.countNonZero(roi_color)
                    if color_px == 0: continue
                    
                    in_range_px = cv2.countNonZero(cv2.bitwise_and(roi_color, roi_depth))
                    if in_range_px / color_px > CYLINDER_DEPTH_RATIO:
                        if area > best_area:
                            best_area = area
                            best_rect = (x, y, w, h)
                
                if best_rect:
                    current_rects[name] = best_rect
                    x, y, w, h = best_rect
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 165, 255), 2)  # 橘色框代表检测到的圆柱体

            # 稳定性验证
            if len(current_rects) == 3:
                history.append(current_rects)
                if len(history) > stable_frames_required:
                    history.pop(0)
                
                if len(history) == stable_frames_required:
                    is_stable = True
                    for i in range(1, stable_frames_required):
                        for name in current_rects:
                            if abs(history[i][name][0] - history[i-1][name][0]) > 5 or \
                               abs(history[i][name][1] - history[i-1][name][1]) > 5:
                                is_stable = False; break
                    
                    if is_stable:
                        # === 生成反向黑名单遮罩 ===
                        # 初始全为白(255)，遇到圆柱体涂黑(0)
                        self.exclusion_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
                        for name, (x, y, w, h) in history[-1].items():
                            # 向外膨胀 MASK_EXPAND 像素，确保边缘反光也被吃掉
                            cv2.rectangle(self.exclusion_mask, 
                                          (max(0, x - MASK_EXPAND), max(0, y - MASK_EXPAND)), 
                                          (min(frame.shape[1], x + w + MASK_EXPAND), min(frame.shape[0], y + h + MASK_EXPAND)), 
                                          0, -1)
                        print("圆柱体环境干扰已屏蔽！黑名单建立成功。")
                        cv2.destroyWindow("Environment Calibration")
                        return
            else:
                history.clear()

            cv2.putText(vis, "Locating static cylinders...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Environment Calibration", vis)
            cv2.waitKey(1)

    # ==========================================
    # Phase 2: 严格初始化靶标捕获 (应用黑名单)
    # ==========================================
    def capture_initial_targets(self):
        print("\n[Phase 2] 正在进行靶标的初始化捕获...")
        cv2.namedWindow("Target Initialization")

        while True:
            frame = self.cam.read_rgb_frame()
            if frame is None: continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 【核心逻辑】：将边缘检测前的灰度图，在黑名单区域强行抹平（变黑）
            # 这样 Canny 算子根本提取不到圆柱体的轮廓，彻底消除干扰！
            safe_gray = cv2.bitwise_and(gray, gray, mask=self.exclusion_mask)
            
            blurred = cv2.GaussianBlur(safe_gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 40, 120)

            final_targets = self._find_target_candidates(edges)
            identified = self._identify_target_colors(frame, final_targets)

            vis = frame.copy()
            # 顺便把黑名单区域半透明涂黑，方便调试观察
            vis[self.exclusion_mask == 0] = vis[self.exclusion_mask == 0] * 0.3
            cv2.putText(vis, "Waiting for constellation geometry...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Target Initialization", vis)
            cv2.waitKey(1)

            if not self._validate_target_geometry(identified):
                continue

            print("\n星座几何条件满足！初始化捕获成功！")
            cv2.destroyWindow("Target Initialization")
            self._build_template(frame, identified)
            return

    def _find_target_candidates(self, edges):
        # 内部寻找轮廓逻辑（与原程序完全一致，但输入的 edge 已去除了干扰）
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 100 < w * h < 40000:
                bboxes.append((x + w // 2, y + h // 2, x, y, w, h))

        clusters = []
        for box in bboxes:
            added = False
            for cluster in clusters:
                if abs(box[0] - cluster[0][0]) < 15 and abs(box[1] - cluster[0][1]) < 15:
                    cluster.append(box); added = True; break
            if not added: clusters.append([box])

        target_cands = []
        for cluster in clusters:
            if len(cluster) >= 4:
                min_x = min(b[2] for b in cluster)
                min_y = min(b[3] for b in cluster)
                max_r = max(b[2] + b[4] for b in cluster)
                max_b = max(b[3] + b[5] for b in cluster)
                tw, th = max_r - min_x, max_b - min_y
                if 0.7 < (float(tw) / th if th > 0 else 0) < 1.4 and 2000 < tw * th < 25000:
                    target_cands.append((min_x, min_y, tw, th, min_x + tw // 2, min_y + th // 2))

        final_targets = []
        target_cands.sort(key=lambda b: b[2] * b[3], reverse=True)
        for cand in target_cands:
            tcx, tcy = cand[4], cand[5]
            if not any(ft[0] < tcx < ft[0] + ft[2] and ft[1] < tcy < ft[1] + ft[3] for ft in final_targets):
                final_targets.append(cand)
        return final_targets

    def _identify_target_colors(self, frame, candidates):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        identified = {}
        for x, y, w, h, cx, cy in candidates:
            roi_hsv = hsv[max(0, y):min(frame.shape[0], y + h), max(0, x):min(frame.shape[1], x + w)]
            if roi_hsv.size == 0: continue
            
            counts = {}
            for name, ranges in TARGET_HSV_RANGES.items():
                mask = cv2.inRange(roi_hsv, ranges[0][0], ranges[0][1])
                if len(ranges) > 1:
                    mask = cv2.bitwise_or(mask, cv2.inRange(roi_hsv, ranges[1][0], ranges[1][1]))
                counts[name] = cv2.countNonZero(mask)

            best_color = max(counts, key=counts.get)
            if counts[best_color] > 100: identified[best_color] = (x, y, w, h, cx, cy)
        return identified

    def _validate_target_geometry(self, identified):
        if len(identified) != 3 or not all(c in identified for c in ["Red", "Green", "Blue"]): return False
        tr, tg, tb = identified["Red"], identified["Green"], identified["Blue"]
        if max(abs(tr[2]-tr[3]), abs(tg[2]-tg[3]), abs(tb[2]-tb[3])) > TOL_SQUARE: return False
        
        ws, hs = [tr[2], tg[2], tb[2]], [tr[3], tg[3], tb[3]]
        if max(max(ws)-min(ws), max(hs)-min(hs)) > TOL_SIZE: return False

        d_rg, d_gb, d_br = get_distance((tr[4], tr[5]), (tg[4], tg[5])), get_distance((tg[4], tg[5]), (tb[4], tb[5])), get_distance((tb[4], tb[5]), (tr[4], tr[5]))
        if max(d_rg, d_gb, d_br) - min(d_rg, d_gb, d_br) > TOL_TRIANGLE: return False
        return True

    def _build_template(self, frame, confirmed_targets):
        self.ref_frame = frame.copy()
        pts = np.array([[data[4], data[5]] for _, data in confirmed_targets.items()], dtype=np.float32)
        (center_x, center_y), strict_radius = cv2.minEnclosingCircle(pts)
        self.ref_mask_center, self.ref_mask_radius = (int(center_x), int(center_y)), int(strict_radius * MASK_SCALE)

        self.target_search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(self.target_search_mask, self.ref_mask_center, self.ref_mask_radius, 255, -1)

        ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.medianBlur(ref_gray, MEDIAN_KSIZE)
        self.ref_binary = cv2.adaptiveThreshold(ref_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.ref_binary = cv2.morphologyEx(self.ref_binary, cv2.MORPH_OPEN, self.kernel)
        
        # 【核心操作】：计算模板时，有效区域既要在星座圆圈内，又不能在圆柱体黑名单内
        final_orb_mask = cv2.bitwise_and(self.target_search_mask, self.exclusion_mask)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self.ref_binary, final_orb_mask)
        
        self.ref_targets_centers = {color: (data[4], data[5]) for color, data in confirmed_targets.items()}
        self.ref_angular_sig = extract_angular_signature(self.ref_binary, self.ref_mask_center, self.ref_mask_radius)
        
        self.temporal_tracker.reset()
        self.current_targets = self.temporal_tracker.update(self.ref_targets_centers)
        self.consecutive_tri_fail = 0
        print(f"模板构建完毕，共提取 {len(self.ref_kp)} 个干净的特征点。")

    # ==========================================
    # Phase 3: 实时抗干扰追踪
    # ==========================================
    def track_loop(self):
        print("\n[Phase 3] 进入高速时域追踪！按 q 退出。")
        cv2.namedWindow("Robust Tracking")
        cv2.namedWindow("Clean Binary View")

        while True:
            frame = self.cam.read_rgb_frame()
            if frame is None: break
            self.depth_map = self.cam.get_depth_map()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, MEDIAN_KSIZE)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
            
            # 【核心逻辑】：应用黑名单屏蔽噪点
            # 提取特征时，把黑名单区域和星座圈结合，确保 ORB 绝对不会去碰圆柱体区域
            final_orb_mask = cv2.bitwise_and(self.target_search_mask, self.exclusion_mask)
            
            # 同样用来算旋转签名
            safe_binary_for_sig = cv2.bitwise_and(binary, binary, mask=self.exclusion_mask)
            cur_angular_sig = extract_angular_signature(safe_binary_for_sig, self.ref_mask_center, self.ref_mask_radius)
            rot_angle = estimate_rotation_angle(self.ref_angular_sig, cur_angular_sig)

            kp, des = self.orb.detectAndCompute(binary, final_orb_mask)
            tracking_success = False

            if des is not None and len(des) > 10:
                matches = self.matcher.match(self.ref_des, des)
                good_matches = sorted(matches, key=lambda x: x.distance)[:max(10, int(len(matches) * 0.2))]

                if len(good_matches) >= 10:
                    src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC)

                    if M is not None:
                        raw_new = {}
                        mc_x, mc_y = self.ref_mask_center
                        for color, (orig_cx, orig_cy) in self.ref_targets_centers.items():
                            nx, ny = int(M[0, 0]*orig_cx + M[0, 1]*orig_cy + M[0, 2]), int(M[1, 0]*orig_cx + M[1, 1]*orig_cy + M[1, 2])
                            # 防止飞出界
                            if get_distance((mc_x, mc_y), (nx, ny)) > self.ref_mask_radius * 0.85:
                                dist = math.hypot(nx - mc_x, ny - mc_y) + 1e-8
                                scale = (self.ref_mask_radius * 0.85) / dist
                                nx, ny = int(mc_x + (nx - mc_x)*scale), int(mc_y + (ny - mc_y)*scale)
                            raw_new[color] = (nx, ny)

                        self.current_targets = self.temporal_tracker.update(raw_new)
                        tracking_success = True

            # 等边约束验证
            if tracking_success:
                pts = list(self.current_targets.values())
                if len(pts) == 3:
                    d01, d12, d20 = get_distance(pts[0], pts[1]), get_distance(pts[1], pts[2]), get_distance(pts[2], pts[0])
                    if max(d01, d12, d20) - min(d01, d12, d20) > TOL_TRIANGLE_TRACK:
                        self.consecutive_tri_fail += 1
                        if self.consecutive_tri_fail >= MAX_TRI_FAIL:
                            print(f"\n连续 {MAX_TRI_FAIL} 帧不满足正三角形约束，需要重新初始化！")
                            cv2.destroyAllWindows()
                            return False  # 跳出循环，触发重新初始化
                    else:
                        self.consecutive_tri_fail = 0

            # 可视化绘制
            vis = frame.copy()
            # 绘制阴影覆盖的黑名单区域
            vis[self.exclusion_mask == 0] = vis[self.exclusion_mask == 0] * 0.3
            
            if tracking_success:
                cv2.putText(vis, f"ROBUST TRACKING | Rot: {rot_angle:.1f}deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                poly_pts = []
                for name, (cx, cy) in self.current_targets.items():
                    bgr = TARGET_COLORS_BGR[name]
                    poly_pts.append((cx, cy))
                    cv2.circle(vis, (cx, cy), 15, bgr, 3)
                    cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)
                    
                    d_val = (self.depth_map[cy, cx] if self.depth_map is not None and 0 <= cy < self.depth_map.shape[0] and 0 <= cx < self.depth_map.shape[1] else 0)
                    cv2.putText(vis, f"{name} {d_val:.0f}mm", (cx - 30, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
                
                if len(poly_pts) == 3:
                    cv2.polylines(vis, [np.array(poly_pts)], isClosed=True, color=(255, 255, 255), thickness=1)
            else:
                cv2.putText(vis, "TRACKING LOST!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 展示给算法看的“干净”二值图
            show_binary = cv2.bitwise_and(binary, binary, mask=final_orb_mask)
            cv2.imshow("Clean Binary View", show_binary)
            cv2.imshow("Robust Tracking", vis)

            if cv2.waitKey(1) == ord("q"):
                return True

    def run_full_pipeline(self):
        self.setup()
        
        # Phase 1: 系统启动时，首先找准圆柱体障碍物的位置，拉黑它们。
        self.calibrate_environment()

        while True:
            # Phase 2: 等待并捕获靶标 (此时圆柱体已被屏蔽，不会发生误捕获)
            self.capture_initial_targets()

            # Phase 3: 进入追踪主循环 (如果追踪崩溃返回False，则回到 Phase 2 重新捕获)
            user_quit = self.track_loop()
            if user_quit: break
            
        if self.cam is not None:
            self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = IntegratedTracker()
    tracker.run_full_pipeline()