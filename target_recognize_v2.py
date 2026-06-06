"""
target_recognize_v2.py
在 target_recognize.py 基础上增加时间域组合滤波：
  1. 滑动窗口中值滤波 — 前置剔除 ORB 误匹配产生的离群飞点
  2. 卡尔曼滤波 (恒速模型) — 最优平滑 + 预测，消除高频小幅抖动

架构：继承 TargetRecognizer，覆写 track_frame 注入滤波逻辑。
      不修改 target_recognize.py 的任何代码。

使用方法:
    from camera_manager import CameraManager
    from target_recognize_v2 import TargetRecognizerV2

    cam = CameraManager()
    cam.start()
    tr = TargetRecognizerV2()
    tr.setup(cam)
    tr.run_full_pipeline()
"""

import cv2
import numpy as np
import math
from collections import deque
from target_recognize import TargetRecognizer, CameraManager


# ==========================================
# 滤波器组件
# ==========================================

class MedianFilter:
    """滑动窗口中值滤波器 — 用于前置剔除偶发的大幅度坐标跳变(飞点)"""

    def __init__(self, window_size=5):
        self.window_size = max(1, window_size)
        self.buffer = None

    def update(self, value):
        """value: 标量或一维数组。返回同形状的中值滤波结果。"""
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


class KalmanFilter1D:
    """1D 恒速卡尔曼滤波器 — 状态: [angle, angular_vel]^T，用于旋转角度平滑"""

    def __init__(self, dt=1.0, q_val=0.05, r_val=5.0):
        self.dt = dt
        self.x = np.zeros((2, 1), dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 500.0
        self.F = np.array([[1, dt],
                           [0,  1]], dtype=np.float64)
        self.H = np.array([[1, 0]], dtype=np.float64)
        self.Q = np.eye(2, dtype=np.float64) * q_val
        self.R = np.array([[r_val]], dtype=np.float64)
        self.initialized = False

    def init(self, angle):
        self.x[0, 0] = float(angle)
        self.initialized = True

    def update(self, z):
        if not self.initialized:
            self.init(z)
            return float(z)

        # 预测
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # 更新
        y_innov = float(z) - float(self.H @ self.x)
        S = float(self.H @ self.P @ self.H.T) + float(self.R[0, 0])
        K = self.P @ self.H.T / S
        self.x = self.x + K * y_innov
        self.P = (np.eye(2, dtype=np.float64) - K @ self.H) @ self.P

        return float(self.x[0, 0])

    def reset(self):
        self.x = np.zeros((2, 1), dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 500.0
        self.initialized = False


# ==========================================
# V2 主类
# ==========================================

class TargetRecognizerV2(TargetRecognizer):
    """
    在 TargetRecognizer 基础上叠加 中值滤波 + 卡尔曼滤波 的时域稳定版。

    可调参数:
        median_window  : 中值滤波窗口大小 (奇数, 默认5)。越大抗飞点越强但滞后越大。
        kalman_q_pos   : 位置卡尔曼的过程噪声 (默认0.05)。越小越平滑，但对机动响应越慢。
        kalman_r_pos   : 位置卡尔曼的观测噪声 (默认10.0)。越大越信任模型预测。
        kalman_q_rot   : 旋转卡尔曼的过程噪声 (默认0.05)。
        kalman_r_rot   : 旋转卡尔曼的观测噪声 (默认5.0)。
    """

    def __init__(self,
                 median_window=5,
                 kalman_q_pos=0.05,
                 kalman_r_pos=10.0,
                 kalman_q_rot=0.05,
                 kalman_r_rot=5.0):
        super().__init__()

        self.median_window = median_window
        self.kalman_q_pos = kalman_q_pos
        self.kalman_r_pos = kalman_r_pos
        self.kalman_q_rot = kalman_q_rot
        self.kalman_r_rot = kalman_r_rot

        # 滤波器实例 (build_template 时初始化)
        self._median_filters = {}    # color -> MedianFilter
        self._kalman_filters = {}    # color -> KalmanFilter2D
        self._rot_median = None
        self._rot_kalman = None
        self._filters_ready = False

    # ==========================================
    # 滤波器管理
    # ==========================================

    def _init_filters(self):
        """在模板构建完成后初始化所有滤波器，并用参考位置赋初值。"""
        for color in ["Red", "Green", "Blue"]:
            cx, cy = self.ref_targets_centers[color]

            mf = MedianFilter(self.median_window)
            mf.update((cx, cy))
            self._median_filters[color] = mf

            kf = KalmanFilter2D(dt=1.0, q_val=self.kalman_q_pos, r_val=self.kalman_r_pos)
            kf.init(cx, cy)
            self._kalman_filters[color] = kf

        self._rot_median = MedianFilter(self.median_window)
        self._rot_median.update(0.0)

        self._rot_kalman = KalmanFilter1D(dt=1.0, q_val=self.kalman_q_rot, r_val=self.kalman_r_rot)
        self._rot_kalman.init(0.0)

        self._filters_ready = True

    def _reset_filters(self):
        """追踪丢失时重置滤波器状态。"""
        self._median_filters.clear()
        self._kalman_filters.clear()
        self._rot_median = None
        self._rot_kalman = None
        self._filters_ready = False

    def _apply_filters(self, raw_targets, raw_rot):
        """
        两级滤波管线:
          1. 中值滤波: 对每个坐标分量独立取窗口内中值, 剔除偶发飞点
          2. 卡尔曼滤波: 恒速模型最优估计, 抑制高频抖动并给出预测位置

        返回: (filtered_targets_dict, filtered_rot_float)
        """
        # --- 旋转角度滤波 ---
        med_rot = self._rot_median.update(raw_rot)
        kf_rot = self._rot_kalman.update(med_rot)

        # --- 靶标坐标滤波 ---
        filtered = {}
        for color in ["Red", "Green", "Blue"]:
            if color not in raw_targets:
                continue
            rx, ry = raw_targets[color]

            # Step 1: 中值滤波去飞点
            med_pos = self._median_filters[color].update((rx, ry))
            mx, my = float(med_pos[0]), float(med_pos[1])

            # Step 2: 卡尔曼滤波平滑
            kf_pos = self._kalman_filters[color].update(mx, my)
            filtered[color] = (int(round(kf_pos[0])), int(round(kf_pos[1])))

        return filtered, kf_rot

    # ==========================================
    # 覆写父类方法
    # ==========================================

    def build_template(self, frame, confirmed_targets):
        """构建模板后自动初始化时域滤波器。"""
        result = super().build_template(frame, confirmed_targets)
        self._init_filters()
        return result

    def track_frame(self, frame, binary):
        """覆写父类 track_frame，在原始追踪结果上叠加中值+卡尔曼滤波。"""
        tracking_success, raw_targets, _parent_img, rot_angle, affine_angle = \
            super().track_frame(frame, binary)

        if not tracking_success:
            self._reset_filters()
            result_img = frame.copy()
            cv2.putText(result_img, "TRACKING LOST! [V2]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return False, None, result_img, rot_angle, None

        if not self._filters_ready:
            return tracking_success, raw_targets, _parent_img, rot_angle, affine_angle

        # 两级滤波
        filtered_targets, filtered_rot = self._apply_filters(
            {c: raw_targets[c] for c in ["Red", "Green", "Blue"] if c in raw_targets},
            rot_angle)

        # 用滤波结果覆写追踪状态，保证下一帧的帧间约束基于平滑坐标
        self.current_targets = filtered_targets
        self.prev_targets = filtered_targets.copy()

        # 基于滤波后坐标重新绘制结果图
        result_img = self._draw_tracking_result(frame, filtered_targets,
                                                 filtered_rot, affine_angle)

        return True, filtered_targets, result_img, filtered_rot, affine_angle

    # ==========================================
    # 可视化 (滤波后重绘，替代父类的原始绘制)
    # ==========================================

    def _draw_tracking_result(self, frame, targets, filtered_rot, affine_angle):
        """在 frame 拷贝上绘制滤波后的追踪可视化。"""
        result_img = frame.copy()

        status = (f"TRACKING V2 [MF+KF] | "
                  f"Rot: {filtered_rot:.1f}deg (affine: {affine_angle:.1f}deg)")
        if self.consecutive_tri_fail > 0:
            status += f" | TriFail:{self.consecutive_tri_fail}"
        cv2.putText(result_img, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        pts = []
        for color, (cx, cy) in targets.items():
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
        cv2.circle(result_img, self.ref_mask_center,
                   int(self.ref_mask_radius * 0.85), (0, 180, 180), 1, cv2.LINE_AA)

        if len(pts) == 3:
            cv2.polylines(result_img, [np.array(pts)], isClosed=True,
                          color=(255, 255, 255), thickness=1)

        return result_img


# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    cam = CameraManager()
    cam.start()

    tr = TargetRecognizerV2()
    tr.setup(cam)
    tr.run_full_pipeline()