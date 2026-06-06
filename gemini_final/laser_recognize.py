"""
高鲁棒性激光点识别模块 (结合形态学、几何特征与时域滤波)
适用于存在强烈环境光干扰、激光点亮度不均、过曝白晕的复杂场景。

使用方法:
    from camera_manager import CameraManager
    from laser_recognize import LaserRecognizer

    cam = CameraManager()
    cam.start()
    
    # 实例化并绑定相机
    lr = LaserRecognizer()
    lr.setup(cam)
    
    # 在循环中调用
    result_img, mask, spot = lr.process_frame(frame, dpt)
    if spot is not None:
        cx, cy, depth = spot
        # 将 cx, cy, depth 传入电机 PID 控制器
"""

import cv2
import numpy as np
import math
from backup_2026_06_05.camera_manager import CameraManager


class TemporalFilterTracker:
    """时域平滑与异常点剔除追踪器"""

    def __init__(self, alpha=0.4, max_jump=60):
        """
        :param alpha: 平滑系数 (0~1)。越小越平滑但滞后，越大响应越快。
        :param max_jump: 允许的最大两帧间像素跳跃距离，超过此距离视为噪点并忽略。
        """
        self.alpha = alpha
        self.max_jump = max_jump
        self.last_cx = None
        self.last_cy = None
        self.last_depth = None
        self.lost_frames = 0
        self.max_lost_frames = 5  # 连续丢失超过此帧数，则重置历史轨迹

    def update(self, measurement):
        """输入视觉测量值，输出滤波后的稳定值"""
        if measurement is None:
            self.lost_frames += 1
            if self.lost_frames >= self.max_lost_frames:
                self.reset()
            return None

        self.lost_frames = 0
        cx, cy, depth = measurement

        # 第一帧初始化
        if self.last_cx is None:
            self.last_cx, self.last_cy, self.last_depth = cx, cy, depth
            return (cx, cy, depth)

        # 异常点剔除 (距离突变检测)
        dist = math.hypot(cx - self.last_cx, cy - self.last_cy)
        if dist > self.max_jump:
            # 视为噪点，保持上一帧状态
            return (int(self.last_cx), int(self.last_cy), self.last_depth)

        # 指数移动平均 (EMA) 滤波
        filtered_cx = self.alpha * cx + (1 - self.alpha) * self.last_cx
        filtered_cy = self.alpha * cy + (1 - self.alpha) * self.last_cy
        filtered_depth = self.alpha * depth + (1 - self.alpha) * self.last_depth

        # 更新历史状态
        self.last_cx, self.last_cy, self.last_depth = filtered_cx, filtered_cy, filtered_depth

        return (int(filtered_cx), int(filtered_cy), filtered_depth)

    def reset(self):
        """重置追踪器状态"""
        self.last_cx = None
        self.last_cy = None
        self.last_depth = None
        self.lost_frames = 0


class LaserRecognizer:
    """激光点综合识别器"""

    def __init__(self, depth_max=1300, tophat_size=15, min_area=5, max_area=200, 
                 min_circularity=0.4, alpha=0.4, max_jump=60):
        # 相机与深度参数
        self.cam = None
        self.depth_max = depth_max
        
        # 视觉算法参数
        self.tophat_size = tophat_size if tophat_size % 2 == 1 else tophat_size + 1
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        
        # 实例化时域追踪器
        self.tracker = TemporalFilterTracker(alpha=alpha, max_jump=max_jump)

    def setup(self, camera_manager=None):
        """绑定相机管理器"""
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

    def create_depth_mask(self, dpt, depth_max=None):
        """生成有效深度遮罩，剔除背景墙壁的噪点"""
        if depth_max is None:
            depth_max = self.depth_max
        mask = np.where((dpt > 0) & (dpt < depth_max), 255, 0).astype(np.uint8)
        return mask

    def detect_bright_spots(self, frame):
        """核心视觉算法：顶帽变换 + 几何特征过滤"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. 顶帽变换：提取局部高光，消除大面积环境光和绝对亮度差异
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.tophat_size, self.tophat_size))
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # 2. 极低固定阈值二值化 (因为背景已由顶帽清零)
        _, thresh = cv2.threshold(top_hat, 30, 255, cv2.THRESH_BINARY)

        # 3. 形态学开运算，消除极细微的杂讯
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # 4. 几何特征过滤 (面积 + 圆度)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_mask = np.zeros_like(gray)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    # 计算圆度
                    circularity = (4 * np.pi * area) / (perimeter * perimeter)
                    if circularity > self.min_circularity:
                        cv2.drawContours(valid_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                        
        return valid_mask

    def _find_best_raw_spot(self, combined_mask):
        """从最终遮罩中找到最大的轮廓中心"""
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] <= 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    def get_depth_at(self, dpt, cx, cy):
        """安全获取深度值"""
        if 0 <= cy < dpt.shape[0] and 0 <= cx < dpt.shape[1]:
            return dpt[cy, cx]
        return -1

    def process_frame(self, frame, dpt, depth_max=None):
        """
        处理单帧，并经过时域滤波，输出最终稳定坐标
        返回: (可视化结果图, 最终识别二值图, (cx, cy, depth))
        """
        # 1. 深度图遮罩对齐
        depth_mask = self.create_depth_mask(dpt, depth_max)
        if depth_mask.shape[:2] != frame.shape[:2]:
            depth_mask = cv2.resize(depth_mask, (frame.shape[1], frame.shape[0]))

        # 2. 视觉特征提取
        bright_mask = self.detect_bright_spots(frame)

        # 3. 空间域融合 (视觉特征 ∩ 有效深度)
        combined = cv2.bitwise_and(bright_mask, depth_mask)

        # 4. 获取当前帧测量结果
        raw_spot_2d = self._find_best_raw_spot(combined)
        
        measurement = None
        if raw_spot_2d is not None:
            cx, cy = raw_spot_2d
            d_val = self.get_depth_at(dpt, cx, cy)
            measurement = (cx, cy, d_val)

        # 5. 时域滤波与异常点剔除
        filtered_spot = self.tracker.update(measurement)

        # 6. 可视化绘制
        result = frame.copy()
        if filtered_spot is not None:
            fcx, fcy, fd = filtered_spot
            # 绘制滤波后的稳定绿圈
            cv2.circle(result, (fcx, fcy), 15, (0, 255, 0), 2)
            cv2.circle(result, (fcx, fcy), 3, (0, 255, 0), -1)
            cv2.putText(result, f"({fcx}, {fcy}) D:{fd:.0f}mm", 
                        (fcx + 20, fcy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # (可选) 绘制浅红色的测量点，方便调试时观察噪点偏移
            if measurement is not None:
                rcx, rcy, _ = measurement
                cv2.circle(result, (rcx, rcy), 2, (0, 0, 255), -1)

        return result, combined, filtered_spot

    def visualize_depth(self, dpt, depth_max=None):
        """生成深度伪彩图"""
        if depth_max is None:
            depth_max = self.depth_max
        vis = cv2.normalize(np.clip(dpt, 0, depth_max), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    def release(self):
        """释放资源"""
        if self.cam is not None:
            self.cam.release()
            self.cam = None


# ==========================================
# 调试与调参入口
# ==========================================
if __name__ == "__main__":
    cam = CameraManager()
    cam.start()

    lr = LaserRecognizer()
    lr.setup(cam)

    cv2.namedWindow("Result")
    # 添加便于动态调参的滑动条
    cv2.createTrackbar("TopHatSize", "Result", 15, 50, lambda _: None)
    cv2.createTrackbar("MinCircular", "Result", 40, 100, lambda _: None) # 除以100即为0.4
    cv2.createTrackbar("DepthMax_mm", "Result", 1300, 5000, lambda _: None)

    while True:
        frame = lr.read_rgb_frame()
        if frame is None:
            break

        dpt = lr.get_depth_map()
        
        # 实时更新参数
        t_size = cv2.getTrackbarPos("TopHatSize", "Result")
        lr.tophat_size = t_size if t_size % 2 == 1 else t_size + 1
        lr.min_circularity = cv2.getTrackbarPos("MinCircular", "Result") / 100.0
        depth_max = cv2.getTrackbarPos("DepthMax_mm", "Result")

        # 核心处理
        result, combined, spot = lr.process_frame(frame, dpt, depth_max)
        
        # 获取基础图以便对比显示
        depth_mask = lr.create_depth_mask(dpt, depth_max)
        if depth_mask.shape[:2] != frame.shape[:2]:
            depth_mask = cv2.resize(depth_mask, (frame.shape[1], frame.shape[0]))

        # 显示
        cv2.imshow("Result", result)
        cv2.imshow("Combined Valid Mask", combined)

        if cv2.waitKey(1) == ord("q"):
            break

    lr.release()