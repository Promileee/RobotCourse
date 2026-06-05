"""
靶标识别与追踪模块
基于ORB特征匹配和极坐标旋转估计的刚体追踪系统。
用于识别和追踪红、绿、蓝三色靶标构成的等边三角形成像星座。

使用方法:
    from target_recognize import TargetRecognizer
    tr = TargetRecognizer()
    tr.setup()
    targets = tr.capture_initial_targets()   # Phase 1: 严苛初始化捕获
    tr.build_template(targets)               # Phase 2: 构建ORB模板
    tr.track_loop()                          # Phase 3: 实时追踪
"""

import cv2
import numpy as np
import math
from openni import openni2


# ==========================================
# 几何约束阈值
# ==========================================
TOL_SQUARE = 15
TOL_SIZE = 15
TOL_TRIANGLE = 20
MASK_SCALE = 1.25
MAX_FRAME_SHIFT = 10
MAX_FRAME_SHIFT_RELAX = 20
TOL_TRIANGLE_TRACK = 5
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


class TargetRecognizer:
    """靶标识别与追踪器"""

    def __init__(self):
        self.cap = None
        self.dev = None
        self.depth_stream = None
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

        # HSV颜色范围
        self.hsv_ranges = {
            "Red": [(np.array([0, 50, 50]), np.array([15, 255, 255])),
                    (np.array([160, 50, 50]), np.array([180, 255, 255]))],
            "Green": [(np.array([40, 50, 50]), np.array([90, 255, 255]))],
            "Blue": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
        }
        self.colors_bgr = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}

        self.current_targets = None
        self.prev_targets = None
        self.was_constrained = False
        self.consecutive_tri_fail = 0
        self.depth_map = None

    # ==========================================
    # 相机管理
    # ==========================================
    def setup(self, skip_n=5):
        """初始化相机并跳过初始帧以稳定传感器"""
        print("正在初始化相机...")
        openni2.initialize()
        self.dev = openni2.Device.open_any()
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.start()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开RGB摄像头")

        print(f"正在跳过前{skip_n}帧以稳定传感器...")
        for _ in range(skip_n):
            self.cap.read()
            self.depth_stream.read_frame()
            cv2.waitKey(100)

    def read_rgb_frame(self):
        """读取RGB帧"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def get_depth_map(self):
        """读取深度帧并返回深度图"""
        depth_frame = self.depth_stream.read_frame()
        dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype="float32")
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype="float32")
        dpt2 *= 255
        dpt = dpt1 + dpt2
        return dpt[:, ::-1]

    # ==========================================
    # 图像预处理
    # ==========================================
    def preprocess(self, frame):
        """对帧进行自适应二值化 + 形态学去噪"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 2)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        return gray, binary

    def preprocess_edges(self, gray):
        """Canny边缘检测"""
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.Canny(blurred, 40, 120)

    # ==========================================
    # Phase 1: 严苛初始化捕获
    # ==========================================
    def find_target_candidates(self, edges):
        """从边缘图中找到候选靶标区域"""
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

        # NMS: 按面积排序，移除被已选区域包含的候选
        final_targets = []
        target_candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
        for cand in target_candidates:
            tcx, tcy = cand[4], cand[5]
            if not any(ft[0] < tcx < ft[0] + ft[2] and ft[1] < tcy < ft[1] + ft[3]
                       for ft in final_targets):
                final_targets.append(cand)

        return final_targets

    def identify_colors(self, frame, final_targets):
        """识别每个候选区域的颜色 (Red/Green/Blue)"""
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
        """验证三靶标的几何约束: 正方形、尺寸一致、等边三角形"""
        if len(identified) != 3 or not all(c in identified for c in ["Red", "Green", "Blue"]):
            return False

        tr, tg, tb = identified["Red"], identified["Green"], identified["Blue"]

        # 正方形约束
        max_sq_diff = max(abs(tr[2] - tr[3]), abs(tg[2] - tg[3]), abs(tb[2] - tb[3]))
        if max_sq_diff > TOL_SQUARE:
            return False

        # 尺寸一致约束
        ws, hs = [tr[2], tg[2], tb[2]], [tr[3], tg[3], tb[3]]
        max_size_diff = max(max(ws) - min(ws), max(hs) - min(hs))
        if max_size_diff > TOL_SIZE:
            return False

        # 等边三角形约束
        d_rg = get_distance((tr[4], tr[5]), (tg[4], tg[5]))
        d_gb = get_distance((tg[4], tg[5]), (tb[4], tb[5]))
        d_br = get_distance((tb[4], tb[5]), (tr[4], tr[5]))
        max_tri_diff = max(d_rg, d_gb, d_br) - min(d_rg, d_gb, d_br)
        if max_tri_diff > TOL_TRIANGLE:
            return False

        return True

    def capture_initial_targets(self):
        """Phase 1: 严苛的初始化捕获循环，返回 confirmed_targets"""
        print("\n开始执行严苛的初始帧捕获逻辑...")
        cv2.namedWindow("Initialization")

        frame_count = 0
        while True:
            frame = self.read_rgb_frame()
            if frame is None:
                continue

            frame_count += 1
            _ = self.get_depth_map()  # consume depth frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            edges = self.preprocess_edges(gray)

            final_targets = self.find_target_candidates(edges)
            identified = self.identify_colors(frame, final_targets)

            # 可视化
            preview_img = frame.copy()
            cv2.putText(preview_img, f"Searching... Frame: {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Initialization", preview_img)

            binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
            cv2.imshow("Init Binary", binary_clean)
            cv2.waitKey(1)

            # 严苛校验
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
        """Phase 2: 基于捕获的靶标构建ORB特征模板"""
        print("正在构建星座模板与特征库...")
        self.ref_frame = frame.copy()

        # 计算最小外接圆
        pts = np.array([[data[4], data[5]] for _, data in confirmed_targets.items()],
                       dtype=np.float32)
        (center_x, center_y), strict_radius = cv2.minEnclosingCircle(pts)
        self.ref_mask_center = (int(center_x), int(center_y))
        self.ref_mask_radius = int(strict_radius * MASK_SCALE)

        # 生成模板遮罩
        self.mask = np.zeros(self.ref_frame.shape[:2], dtype=np.uint8)
        cv2.circle(self.mask, self.ref_mask_center, self.ref_mask_radius, 255, -1)

        # 提取参考特征
        ref_gray = cv2.cvtColor(self.ref_frame, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.medianBlur(ref_gray, 2)
        self.ref_binary = cv2.adaptiveThreshold(ref_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
        self.ref_binary = cv2.morphologyEx(self.ref_binary, cv2.MORPH_OPEN, self.kernel)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self.ref_binary, self.mask)
        self.ref_targets_centers = {color: (data[4], data[5])
                                     for color, data in confirmed_targets.items()}

        # 提取参考角度签名
        self.ref_angular_sig = extract_angular_signature(self.ref_binary,
                                                          self.ref_mask_center,
                                                          self.ref_mask_radius)

        # 初始化追踪状态
        self.current_targets = self.ref_targets_centers.copy()
        self.prev_targets = self.ref_targets_centers.copy()
        self.was_constrained = False
        self.consecutive_tri_fail = 0

        print(f"模板构建完毕，共提取 {len(self.ref_kp)} 个特征点。")
        return len(self.ref_kp)

    # ==========================================
    # Phase 3: 实时追踪
    # ==========================================
    def track_frame(self, frame, binary):
        """追踪单帧，返回 (success, targets_dict, result_img, rot_angle, affine_angle)"""
        result_img = frame.copy()
        cur_angular_sig = extract_angular_signature(binary, self.ref_mask_center,
                                                     self.ref_mask_radius)
        rot_angle = estimate_rotation_angle(self.ref_angular_sig, cur_angular_sig)

        kp, des = self.orb.detectAndCompute(binary, self.mask)
        tracking_success = False
        M = None

        if des is not None and len(des) > 10:
            matches = self.matcher.match(self.ref_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:max(10, int(len(matches) * 0.2))]

            if len(good_matches) >= 10:
                src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]
                                     ).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]
                                     ).reshape(-1, 1, 2)

                M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC)

                if M is not None:
                    new_targets = {}
                    for color, (orig_cx, orig_cy) in self.ref_targets_centers.items():
                        nx = M[0, 0] * orig_cx + M[0, 1] * orig_cy + M[0, 2]
                        ny = M[1, 0] * orig_cx + M[1, 1] * orig_cy + M[1, 2]
                        new_targets[color] = (int(nx), int(ny))

                    # 帧间位移约束
                    effective_max_shift = (MAX_FRAME_SHIFT_RELAX if self.was_constrained
                                           else MAX_FRAME_SHIFT)
                    frame_constrained = False
                    for color in new_targets:
                        px, py = self.prev_targets[color]
                        nx, ny = new_targets[color]
                        if get_distance((px, py), (nx, ny)) > effective_max_shift:
                            new_targets[color] = (px, py)
                            frame_constrained = True
                    self.was_constrained = frame_constrained

                    # 圆周约束
                    mc_x, mc_y = self.ref_mask_center
                    for color in new_targets:
                        nx, ny = new_targets[color]
                        if get_distance((mc_x, mc_y), (nx, ny)) > self.ref_mask_radius * 0.85:
                            dx, dy = nx - mc_x, ny - mc_y
                            dist = math.hypot(dx, dy) + 1e-8
                            scale = (self.ref_mask_radius * 0.85) / dist
                            new_targets[color] = (int(mc_x + dx * scale),
                                                  int(mc_y + dy * scale))

                    tracking_success = True
                    self.current_targets = new_targets

        # 正三角形约束
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

            self.prev_targets = self.current_targets.copy()
        else:
            cv2.putText(result_img, "TRACKING LOST!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return tracking_success, self.current_targets, result_img, rot_angle, affine_angle

    def track_loop(self):
        """Phase 3: 实时追踪主循环 (阻塞，按q退出或追踪失败返回False)"""
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

            # 绘制特征计算圈内的二值化内容
            binary_masked = np.zeros_like(binary)
            cv2.circle(binary_masked, self.ref_mask_center, self.ref_mask_radius, 255, -1)
            binary_masked = cv2.bitwise_and(binary, binary_masked)

            cv2.imshow("Constellation Tracking", result_img)
            cv2.imshow("Tracking Binary (Masked)", binary_masked)
            if cv2.waitKey(1) == ord("q"):
                return True  # 用户主动退出

        return False

    # ==========================================
    # 便捷方法：自动完成全套流程
    # ==========================================
    def run_full_pipeline(self):
        """运行完整的 初始化→模板构建→追踪 流程，支持追踪失败后自动重试"""
        self.setup()

        while True:
            # Phase 1
            frame, targets = self.capture_initial_targets()
            if frame is None:
                break

            # Phase 2
            self.build_template(frame, targets)

            # Phase 3 (track_loop 返回 True=用户退出, False=需要重新初始化)
            user_quit = self.track_loop()
            if user_quit:
                break

        self.release()

    def release(self):
        """释放所有资源"""
        if self.cap is not None:
            self.cap.release()
        if self.depth_stream is not None:
            self.depth_stream.stop()
        if self.dev is not None:
            self.dev.close()
        cv2.destroyAllWindows()


# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    tr = TargetRecognizer()
    tr.run_full_pipeline()
