"""
laser_recognize_v3.py
改进的激光点识别模块 — 基于 v2。

相比 v2 的改进:
  1. 深度邻域搜索 — 深度检查时在以激光点为中心的圆形邻域内搜索
     最接近靶标平面深度的像素值，解决深度/RGB摄像头位差导致的边缘
     激光点被深度约束拒绝的问题。
     depth_search_radius 参数控制搜索半径（像素），设为 0 等价于 v2 行为。

继承 v2 的所有改进:
  1. 靶标平面深度过滤 — 只接受深度在靶标平面附近的亮斑
  2. 靶标圆盘 ROI — 只在靶标外接圆放大后的圆盘范围内搜索
  3. 形状与颜色特征过滤 — 排除非圆形、过大、非红色的轮廓
  4. 多通道检测 — 灰度图 + HSV V通道取交集
  5. 中值 + 卡尔曼时域滤波 — 平滑激光点坐标
  6. 多潜在点跟踪 — 同时跟踪多个候选点，选择最稳定的
  7. 检测评分 — 综合亮度、形状、深度一致性打分

使用方法:
    from camera_manager import CameraManager
    from target_recognize_v2 import TargetRecognizerV2
    from laser_recognize_v3 import LaserRecognizerV3

    cam = CameraManager()
    cam.start()

    tr = TargetRecognizerV2()
    tr.setup(cam)
    # ... 靶标初始化完成后:
    lr = LaserRecognizerV3()
    lr.setup(cam)
    lr.init_from_target_data(tr.ref_targets_centers, depth_map)

    # 每帧:
    result, combined, spot, debug = lr.process_frame(frame, dpt)
"""

import cv2
import numpy as np
import math
from collections import deque


# ==========================================
# 时域滤波器 (自包含，避免循环导入)
# ==========================================

class MedianFilter:
    """滑动窗口中值滤波器 — 用于前置剔除偶发跳变"""

    def __init__(self, window_size=5):
        self.window_size = max(1, window_size)
        self.buffer = None

    def update(self, value):
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if self.buffer is None:
            self.buffer = deque([arr.copy()] * self.window_size, maxlen=self.window_size)
        else:
            self.buffer.append(arr.copy())
        stacked = np.array(list(self.buffer))
        med = np.median(stacked, axis=0)
        if np.isscalar(value):
            return float(med[0]) if med.shape == (1,) else float(med)
        return med

    def reset(self):
        self.buffer = None


class KalmanFilter2D:
    """2D 恒速卡尔曼滤波器 — 状态: [x, y, vx, vy]^T，观测: [x, y]^T"""

    def __init__(self, dt=1.0, q_val=0.05, r_val=10.0):
        self.dt = dt
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 500.0
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.eye(4, dtype=np.float64) * q_val
        self.R = np.eye(2, dtype=np.float64) * r_val
        self.initialized = False

    def init(self, x, y):
        self.x[0, 0] = float(x)
        self.x[1, 0] = float(y)
        self.initialized = True

    def predict(self):
        """只预测，不更新观测"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, z_x, z_y):
        if not self.initialized:
            self.init(z_x, z_y)
            return np.array([z_x, z_y], dtype=np.float64)

        z = np.array([[z_x], [z_y]], dtype=np.float64)

        # 预测
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # 更新
        y_innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y_innov
        self.P = (np.eye(4, dtype=np.float64) - K @ self.H) @ self.P

        return np.array([self.x[0, 0], self.x[1, 0]], dtype=np.float64)

    def reset(self):
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 500.0
        self.initialized = False


# ==========================================
# 多候选点跟踪状态
# ==========================================

class TrackedCandidate:
    """单个潜在激光点的跨帧跟踪状态，内置卡尔曼滤波器"""

    def __init__(self, cx, cy, score, frame_id=0):
        self.kf = KalmanFilter2D(dt=1.0, q_val=0.1, r_val=5.0)
        self.kf.init(cx, cy)
        self.score = score
        self.missed_frames = 0
        self.total_frames = 1
        self.frame_id = frame_id
        self.position_history = deque([(cx, cy)], maxlen=10)
        self.stability = 1.0

    def predict(self):
        return self.kf.predict()

    def update(self, z_x, z_y, score, frame_id):
        self.kf.update(z_x, z_y)
        self.score = 0.6 * self.score + 0.4 * score
        self.missed_frames = 0
        self.total_frames += 1
        self.frame_id = frame_id
        self.position_history.append((float(self.kf.x[0, 0]), float(self.kf.x[1, 0])))
        self._update_stability()

    def mark_missed(self):
        self.missed_frames += 1

    def _update_stability(self):
        if len(self.position_history) < 3:
            self.stability = 1.0
            return
        positions = np.array(list(self.position_history))
        displacements = np.diff(positions, axis=0)
        mean_disp = float(np.mean(np.linalg.norm(displacements, axis=1)))
        self.stability = 1.0 / (1.0 + mean_disp)

    @property
    def position(self):
        return (int(round(self.kf.x[0, 0])), int(round(self.kf.x[1, 0])))

    @property
    def quality(self):
        """综合质量: 评分 × 稳定性，成熟跟踪有加成"""
        age_bonus = min(1.0, self.total_frames / 30.0)
        return self.score * (0.5 + 0.5 * self.stability) * (0.7 + 0.3 * age_bonus)


# ==========================================
# LaserRecognizerV3 主类
# ==========================================

class LaserRecognizerV3:
    """改进版激光点识别器 (v3 — 深度邻域搜索)"""

    def __init__(self,
                 # 亮度检测
                 thresh_val=225,
                 blur_size=7,
                 # 深度
                 depth_max=2000,
                 depth_tolerance=40.0,
                 depth_search_radius=5,
                 # ROI
                 roi_scale=1.85,
                 # 形状过滤
                 min_area=3,
                 max_area=500,
                 circularity_min=0.45,
                 # HSV 红色色相 (OpenCV 0-180, 极宽范围)
                 h_low_range=(0, 55),
                 h_high_range=(125, 180),
                 # V 通道阈值 (对红光响应弱于灰度, 阈值略低)
                 v_thresh_val=210,
                 # 时域滤波
                 median_window=5,
                 kalman_q=0.05,
                 kalman_r=10.0,
                 # 多候选跟踪
                 max_candidates=5,
                 max_missed_frames=10,
                 association_dist=30,
                 # 评分权重
                 w_brightness=0.3,
                 w_circularity=0.3,
                 w_depth=0.4):
        self.thresh_val = thresh_val
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self.depth_max = depth_max
        self.depth_tolerance = depth_tolerance
        self.depth_search_radius = depth_search_radius
        self.roi_scale = roi_scale
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_min = circularity_min
        self.h_low_range = h_low_range
        self.h_high_range = h_high_range
        self.v_thresh_val = v_thresh_val
        self.median_window = median_window
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        self.max_candidates = max_candidates
        self.max_missed_frames = max_missed_frames
        self.association_dist = association_dist
        self.w_brightness = w_brightness
        self.w_circularity = w_circularity
        self.w_depth = w_depth

        # 靶标初始化状态 (由 init_from_target_data 设置)
        self.initialized = False
        self.target_depth = None
        self.roi_center = None
        self.roi_radius = None

        # 时域滤波器
        self._median_x = MedianFilter(window_size=median_window)
        self._median_y = MedianFilter(window_size=median_window)

        # 多候选跟踪
        self.tracked_candidates = []
        self._frame_count = 0

        # 相机
        self.cam = None

    # ==========================================
    # 初始化
    # ==========================================

    def setup(self, camera_manager=None):
        """绑定相机管理器"""
        if self.cam is not None:
            return
        if camera_manager is None:
            from camera_manager import CameraManager
            self.cam = CameraManager()
            self.cam.start()
        else:
            self.cam = camera_manager

    def init_from_target_data(self, target_centers, depth_map):
        """
        从靶标识别的参考中心一次性初始化 ROI 和目标平面深度。

        Args:
            target_centers: dict {color: (cx, cy)}，通常来自 TargetRecognizerV2.ref_targets_centers
            depth_map: 对应的深度图
        """
        # 1. 计算三个靶标中心的外接圆 → ROI 圆盘
        pts = np.array([[cx, cy] for cx, cy in target_centers.values()], dtype=np.float32)
        (center_x, center_y), radius = cv2.minEnclosingCircle(pts)
        self.roi_center = (int(center_x), int(center_y))
        self.roi_radius = int(radius * self.roi_scale)

        # 2. 读取三个靶标中心的深度 → 靶标平面平均深度
        depths = []
        for cx, cy in target_centers.values():
            if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                d = depth_map[cy, cx]
                if d > 0:
                    depths.append(float(d))

        self.target_depth = float(np.mean(depths)) if depths else None
        self.initialized = True

        if self.target_depth is not None:
            print(f"[LaserV3] 靶标平面深度: {self.target_depth:.1f}mm (容差 ±{self.depth_tolerance:.0f}mm)")
        else:
            print("[LaserV3] 警告: 未能从靶标中心读取到有效深度值")
        print(f"[LaserV3] 搜索ROI: 圆心({self.roi_center[0]}, {self.roi_center[1]}), "
              f"半径={self.roi_radius}px (外接圆×{self.roi_scale})")

    def reset(self):
        """重置所有状态"""
        self.initialized = False
        self.target_depth = None
        self.roi_center = None
        self.roi_radius = None
        self._median_x.reset()
        self._median_y.reset()
        self.tracked_candidates.clear()
        self._frame_count = 0

    # ==========================================
    # 相机委托
    # ==========================================

    def read_rgb_frame(self):
        return self.cam.read_rgb_frame()

    def get_depth_map(self):
        return self.cam.get_depth_map()

    # ==========================================
    # 深度邻域搜索 (v3 新增)
    # ==========================================

    def _get_best_depth_in_neighborhood(self, dpt, cx, cy, h, w):
        """
        在以 (cx, cy) 为中心、depth_search_radius 为半径的圆形邻域内，
        搜索最接近靶标平面深度的有效深度值。

        当 depth_search_radius == 0 时，等价于直接读取 dpt[cy, cx]。

        Returns:
            best_depth: 邻域内最接近 target_depth 的深度值，若找不到有效深度返回 -1.0
        """
        if self.target_depth is None or self.depth_search_radius <= 0:
            return float(dpt[cy, cx]) if (0 <= cy < h and 0 <= cx < w) else -1.0

        best_depth = -1.0
        best_diff = float('inf')
        r = self.depth_search_radius

        y_min = max(0, cy - r)
        y_max = min(h, cy + r + 1)
        x_min = max(0, cx - r)
        x_max = min(w, cx + r + 1)
        r_sq = r * r

        for ny in range(y_min, y_max):
            for nx in range(x_min, x_max):
                if (nx - cx) ** 2 + (ny - cy) ** 2 > r_sq:
                    continue
                d_val = float(dpt[ny, nx])
                if d_val <= 0:
                    continue
                diff = abs(d_val - self.target_depth)
                if diff < best_diff:
                    best_diff = diff
                    best_depth = d_val

        return best_depth

    # ==========================================
    # 候选点检测
    # ==========================================

    def detect_candidates(self, frame, dpt):
        """
        在当前帧中检测所有潜在激光点候选。
        返回按评分降序排列的候选列表 (最多 max_candidates 个)。

        改进点:
          1. ROI 圆盘约束
          2. 多通道检测 (灰度 + V 通道交集)
          3. 形状过滤 (圆形度、面积)
          4. 深度预过滤 (靶标平面 ± 宽松容差) — v3 使用邻域搜索
        """
        if not self.initialized:
            return []

        h, w = frame.shape[:2]

        # --- 1. ROI 圆盘遮罩 ---
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(roi_mask, self.roi_center, self.roi_radius, 255, -1)

        # --- 2. 多通道亮度检测 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        _, bright_gray = cv2.threshold(blurred_gray, self.thresh_val, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        blurred_v = cv2.GaussianBlur(v_channel, (self.blur_size, self.blur_size), 0)
        _, bright_v = cv2.threshold(blurred_v, self.v_thresh_val, 255, cv2.THRESH_BINARY)

        # 灰度 ∩ V通道 → 必须在两个通道中都足够亮
        bright = cv2.bitwise_and(bright_gray, bright_v)

        # --- 3. 应用 ROI ---
        bright = cv2.bitwise_and(bright, roi_mask)

        # --- 4. 形态学去噪 ---
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # --- 5. 找轮廓 → 形状过滤 + 评分 ---
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            # 圆形度: 4π·area / perimeter², 正圆=1.0
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            if circularity < self.circularity_min:
                continue

            M = cv2.moments(cnt)
            if M["m00"] <= 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 深度检查 — 使用邻域搜索 (v3 改进)
            # 在激光点周围圆形邻域内寻找最接近靶标平面深度的像素
            d_val = self._get_best_depth_in_neighborhood(dpt, cx, cy, h, w)
            if d_val <= 0:
                continue
            if self.target_depth is not None:
                if abs(d_val - self.target_depth) > self.depth_tolerance * 5.0:
                    continue  # 极宽松: 排除明显不在靶标平面上的点

            # --- 评分 ---
            # 亮度分: 轮廓内灰度均值归一化
            cnt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            mean_brightness = float(np.mean(gray[cnt_mask == 255])) / 255.0 if np.any(cnt_mask == 255) else 0.0

            # 深度分: 使用邻域搜索到的最佳深度
            if self.target_depth is not None and d_val > 0:
                depth_score = max(0.0, 1.0 - abs(d_val - self.target_depth) / self.depth_tolerance)
            else:
                depth_score = 0.5

            # 综合评分
            score = (self.w_brightness * mean_brightness +
                     self.w_circularity * circularity +
                     self.w_depth * depth_score)

            candidates.append({
                'cx': cx, 'cy': cy,
                'area': area,
                'circularity': circularity,
                'depth': d_val,
                'brightness': mean_brightness,
                'depth_score': depth_score,
                'score': score,
                'contour': cnt,
            })

        candidates.sort(key=lambda c: c['score'], reverse=True)
        return candidates[:self.max_candidates]

    # ==========================================
    # 多候选关联与跟踪
    # ==========================================

    def _associate_and_update(self, candidates):
        """
        将检测到的候选点与已有的 TrackedCandidate 关联。
        返回最佳激光点 (cx, cy, depth, quality) 或 None。

        关联策略: 贪心最近邻
        选择策略: 质量(quality)最高且成熟的跟踪
        """
        # 预测所有已有跟踪的下一位置
        predictions = {tc: tc.predict() for tc in self.tracked_candidates}

        assigned_tracks = set()
        assigned_cands = set()

        # 贪心关联: 按候选评分降序，每个候选找最近的未匹配跟踪
        for i, cand in enumerate(candidates):
            best_tc = None
            best_dist = float('inf')
            for tc in self.tracked_candidates:
                if tc in assigned_tracks:
                    continue
                px, py = predictions[tc]
                dist = math.hypot(cand['cx'] - px, cand['cy'] - py)
                if dist < best_dist and dist < self.association_dist:
                    best_dist = dist
                    best_tc = tc

            if best_tc is not None:
                best_tc.update(cand['cx'], cand['cy'], cand['score'], self._frame_count)
                assigned_tracks.add(best_tc)
                assigned_cands.add(i)

        # 未匹配的候选 → 创建新跟踪
        for i, cand in enumerate(candidates):
            if i not in assigned_cands:
                tc = TrackedCandidate(cand['cx'], cand['cy'], cand['score'], self._frame_count)
                self.tracked_candidates.append(tc)

        # 未匹配的跟踪 → 标记丢失
        for tc in self.tracked_candidates:
            if tc not in assigned_tracks:
                tc.mark_missed()

        # 清理丢失太久的跟踪
        self.tracked_candidates = [tc for tc in self.tracked_candidates
                                   if tc.missed_frames <= self.max_missed_frames]

        # 按质量排序并限制数量
        self.tracked_candidates.sort(key=lambda tc: tc.quality, reverse=True)
        self.tracked_candidates = self.tracked_candidates[:self.max_candidates]

        # 选择最佳: 成熟(≥3帧)且质量最高
        mature = [tc for tc in self.tracked_candidates if tc.total_frames >= 3]
        if not mature:
            if candidates:
                best = candidates[0]
                return best['cx'], best['cy'], best['depth'], best['score']
            return None

        best_tc = max(mature, key=lambda tc: tc.quality)
        cx, cy = best_tc.position
        # 查找当前帧中与最佳跟踪最近的候选深度
        depth = best_tc.score  # fallback
        best_dist = float('inf')
        for cand in candidates:
            d = math.hypot(cand['cx'] - cx, cand['cy'] - cy)
            if d < best_dist:
                best_dist = d
                depth = cand['depth']
        if best_dist > self.association_dist:
            depth = -1.0

        return cx, cy, depth, best_tc.quality

    # ==========================================
    # 主处理入口
    # ==========================================

    def process_frame(self, frame, dpt):
        """
        处理一帧，返回:
            result_frame:  标注了激光点、ROI圆、候选跟踪的可视化图像
            combined_mask: 最佳候选的二值遮罩 (用于调试)
            spot_info:     (cx, cy, depth_val, score) 或 None
            debug_info:    dict，包含候选列表、跟踪数量等
        """
        self._frame_count += 1

        # 对齐深度图尺寸
        if dpt.shape[:2] != frame.shape[:2]:
            dpt = cv2.resize(dpt, (frame.shape[1], frame.shape[0]))

        candidates = self.detect_candidates(frame, dpt)
        track_result = self._associate_and_update(candidates)

        result = frame.copy()
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        spot_info = None

        if track_result is not None:
            cx, cy, depth_val, quality = track_result

            # 中值滤波作为最终飞点剔除 (卡尔曼已在 TrackedCandidate 内部完成)
            mx = self._median_x.update(cx)
            my = self._median_y.update(cy)
            if abs(mx - cx) < 30 and abs(my - cy) < 30:
                fx, fy = int(round(mx)), int(round(my))
            else:
                fx, fy = cx, cy

            spot_info = (fx, fy, depth_val, quality)

            # 绘制激光点
            cv2.circle(result, (fx, fy), 15, (0, 255, 0), 2)
            cv2.circle(result, (fx, fy), 3, (0, 255, 0), -1)
            cv2.putText(result, f"({fx},{fy}) d={depth_val:.0f} s={quality:.2f}",
                        (fx + 20, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # 绘制最佳候选轮廓
            if candidates:
                cv2.drawContours(result, [candidates[0]['contour']], -1, (0, 255, 255), 1)
                combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(combined_mask, [candidates[0]['contour']], -1, 255, -1)

        # 绘制 ROI 圆 (调试可视化)
        if self.initialized:
            cv2.circle(result, self.roi_center, self.roi_radius, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(result, f"ROI r={self.roi_radius}",
                        (self.roi_center[0] - 40,
                         self.roi_center[1] - self.roi_radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # 绘制所有被跟踪的候选 (黄色小圈)
        for tc in self.tracked_candidates:
            if tc.total_frames >= 2:
                px, py = tc.position
                cv2.circle(result, (px, py), 5, (255, 255, 0), 1)
                cv2.putText(result, f"q{tc.quality:.1f}", (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        # 靶标平面深度参考线
        if self.target_depth is not None:
            cv2.putText(result, f"TgtDepth: {self.target_depth:.0f}mm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        debug_info = {
            'candidates': candidates,
            'tracked_count': len(self.tracked_candidates),
            'roi_center': self.roi_center,
            'roi_radius': self.roi_radius,
        }

        return result, combined_mask, spot_info, debug_info

    # ==========================================
    # 可视化工具
    # ==========================================

    def visualize_depth(self, dpt, depth_max=None):
        """生成深度伪彩色图"""
        if depth_max is None:
            depth_max = self.depth_max
        vis = cv2.normalize(np.clip(dpt, 0, depth_max), None, 0, 255,
                            cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    def create_roi_visualization(self, frame):
        """
        生成 ROI 圆盘可视化图像 (用于调试 roi_scale 参数)。
        半透明紫色圆盘叠加在原图上。
        """
        vis = frame.copy()
        if not self.initialized:
            return vis
        overlay = vis.copy()
        cv2.circle(overlay, self.roi_center, self.roi_radius, (255, 0, 255), -1)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        cv2.circle(vis, self.roi_center, self.roi_radius, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"ROI r={self.roi_radius} (x{self.roi_scale})",
                    (self.roi_center[0] - 60,
                     self.roi_center[1] - self.roi_radius - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        return vis

    def release(self):
        """释放相机资源"""
        if self.cam is not None:
            self.cam.release()
            self.cam = None


# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    from camera_manager import CameraManager
    from target_recognize_v2 import TargetRecognizerV2

    cam = CameraManager()
    cam.start()

    tr = TargetRecognizerV2()
    tr.setup(cam)

    lr = LaserRecognizerV3()
    lr.setup(cam)

    # --- 调试窗口与 Trackbars ---
    cv2.namedWindow("Laser Result")
    cv2.createTrackbar("Threshold", "Laser Result", 225, 255, lambda _: None)
    cv2.createTrackbar("V-Threshold", "Laser Result", 210, 255, lambda _: None)
    cv2.createTrackbar("Blur", "Laser Result", 3, 10, lambda _: None)
    cv2.createTrackbar("ROI Scale x10", "Laser Result", 19, 30, lambda _: None)
    cv2.createTrackbar("Circularity x100", "Laser Result", 45, 95, lambda _: None)
    cv2.createTrackbar("Depth Tol mm", "Laser Result", 40, 100, lambda _: None)
    cv2.createTrackbar("Depth Search R", "Laser Result", 5, 20, lambda _: None)

    # --- 先初始化靶标 ---
    print("正在初始化靶标识别...")
    frame, targets = tr.capture_initial_targets()
    _ = tr.build_template(frame, targets)

    # 获取深度图并初始化激光识别
    dpt = tr.get_depth_map()
    lr.init_from_target_data(tr.ref_targets_centers, dpt)

    print("\n进入激光识别调试模式，按 q 退出。")

    while True:
        frame = lr.read_rgb_frame()
        if frame is None:
            print("Cannot read RGB frame")
            break

        dpt = lr.get_depth_map()

        # 读取调试参数
        lr.thresh_val = cv2.getTrackbarPos("Threshold", "Laser Result")
        lr.v_thresh_val = cv2.getTrackbarPos("V-Threshold", "Laser Result")
        blur_val = cv2.getTrackbarPos("Blur", "Laser Result")
        lr.blur_size = blur_val * 2 + 1
        lr.roi_scale = cv2.getTrackbarPos("ROI Scale x10", "Laser Result") / 10.0
        lr.circularity_min = cv2.getTrackbarPos("Circularity x100", "Laser Result") / 100.0
        lr.depth_tolerance = float(cv2.getTrackbarPos("Depth Tol mm", "Laser Result"))
        lr.depth_search_radius = cv2.getTrackbarPos("Depth Search R", "Laser Result")

        # 动态更新 ROI 半径 (圆心和深度不变)
        if hasattr(tr, 'ref_targets_centers'):
            pts = np.array([[cx, cy] for cx, cy in tr.ref_targets_centers.values()], dtype=np.float32)
            (_cx, _cy), radius = cv2.minEnclosingCircle(pts)
            lr.roi_center = (int(_cx), int(_cy))
            lr.roi_radius = int(radius * lr.roi_scale)

        result, combined, spot, debug = lr.process_frame(frame, dpt)
        roi_vis = lr.create_roi_visualization(frame)

        cv2.imshow("Laser Result", result)
        cv2.imshow("ROI Visualization", roi_vis)

        # 候选点轮廓可视化
        cand_viz = np.zeros(frame.shape[:2], dtype=np.uint8)
        for c in debug['candidates']:
            cv2.drawContours(cand_viz, [c['contour']], -1, 255, -1)
        cv2.imshow("Candidates", cand_viz)

        if spot:
            print(f"Laser: ({spot[0]},{spot[1]}) d={spot[2]:.0f}mm score={spot[3]:.2f} | "
                  f"tracks={debug['tracked_count']}")

        if cv2.waitKey(1) == ord("q"):
            break

    lr.release()
    cv2.destroyAllWindows()
