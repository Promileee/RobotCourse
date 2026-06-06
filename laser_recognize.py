"""
激光点识别模块
结合RGB相机亮度检测和深度相机的距离信息，识别激光点位置。
使用方法:
    from camera_manager import CameraManager
    from laser_recognize import LaserRecognizer

    cam = CameraManager()
    cam.start()
    lr = LaserRecognizer()
    lr.setup(cam)
    result, combined, spot = lr.process_frame(frame, dpt)
"""

import cv2
import numpy as np
from camera_manager import CameraManager


class LaserRecognizer:
    """激光点识别器"""

    def __init__(self, depth_max=1300, thresh_val=240, blur_size=8):
        self.depth_max = depth_max
        self.thresh_val = thresh_val
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self.cam = None

    def setup(self, camera_manager=None):
        """绑定相机管理器。若为 None 则自动创建一个 (独立调试用)。"""
        if self.cam is not None:
            return  # 已绑定，避免重复初始化
        if camera_manager is None:
            self.cam = CameraManager()
            self.cam.start()
        else:
            self.cam = camera_manager

    def read_rgb_frame(self):
        """读取RGB帧 (委托给 CameraManager)"""
        return self.cam.read_rgb_frame()

    def get_depth_map(self):
        """读取深度帧并返回深度图 (委托给 CameraManager)"""
        return self.cam.get_depth_map()

    def create_depth_mask(self, dpt, depth_max=None):
        """创建深度遮罩：有效距离内的像素为255"""
        if depth_max is None:
            depth_max = self.depth_max
        mask = np.where((dpt > 0) & (dpt < depth_max), 255, 0).astype(np.uint8)
        return mask

    def detect_bright_spots(self, frame, thresh_val=None, blur_size=None):
        """检测亮度高的区域（激光点）"""
        if thresh_val is None:
            thresh_val = self.thresh_val
        if blur_size is None:
            blur_size = self.blur_size
        if blur_size % 2 == 0:
            blur_size += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        _, bright = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        return bright

    def find_laser_spot(self, combined_mask):
        """从合并的遮罩中找到最大的激光点轮廓，返回 (cx, cy, area) 或 None"""
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area <= 2:
            return None

        M = cv2.moments(largest)
        if M["m00"] <= 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy, area

    def get_depth_at(self, dpt, cx, cy):
        """获取指定像素位置的深度值"""
        if 0 <= cy < dpt.shape[0] and 0 <= cx < dpt.shape[1]:
            return dpt[cy, cx]
        return -1

    def process_frame(self, frame, dpt, depth_max=None, thresh_val=None, blur_size=None):
        """处理一帧，返回 (result_frame, combined_mask, spot_info)
        spot_info = (cx, cy, depth_val) 或 None
        """
        if depth_max is None:
            depth_max = self.depth_max

        depth_mask = self.create_depth_mask(dpt, depth_max)
        if depth_mask.shape[:2] != frame.shape[:2]:
            depth_mask = cv2.resize(depth_mask, (frame.shape[1], frame.shape[0]))

        bright = self.detect_bright_spots(frame, thresh_val, blur_size)

        combined = cv2.bitwise_and(bright, depth_mask)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        spot = self.find_laser_spot(combined)
        result = frame.copy()

        if spot is not None:
            cx, cy, area = spot
            d_val = self.get_depth_at(dpt, cx, cy)
            spot = (cx, cy, d_val)
            cv2.circle(result, (cx, cy), 15, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 3, (0, 255, 0), -1)
            cv2.putText(result, f"({cx}, {cy}) d={d_val:.0f}mm",
                        (cx + 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result, combined, spot

    def visualize_depth(self, dpt, depth_max=None):
        """生成深度可视化图像"""
        if depth_max is None:
            depth_max = self.depth_max
        vis = cv2.normalize(np.clip(dpt, 0, depth_max), None, 0, 255,
                            cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    def release(self):
        """释放相机资源 (委托给 CameraManager)"""
        if self.cam is not None:
            self.cam.release()
            self.cam = None


# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    cam = CameraManager()
    cam.start()

    lr = LaserRecognizer()
    lr.setup(cam)

    cv2.namedWindow("Result")
    cv2.createTrackbar("Threshold", "Result", 235, 255, lambda _: None)
    cv2.createTrackbar("Blur", "Result", 3, 20, lambda _: None)
    cv2.createTrackbar("DepthMax_mm", "Result", 1300, 5000, lambda _: None)

    while True:
        frame = lr.read_rgb_frame()
        if frame is None:
            print("Cannot read RGB frame")
            break

        dpt = lr.get_depth_map()
        depth_max = cv2.getTrackbarPos("DepthMax_mm", "Result")
        thresh_val = cv2.getTrackbarPos("Threshold", "Result")
        blur_size = cv2.getTrackbarPos("Blur", "Result")

        result, combined, spot = lr.process_frame(frame, dpt, depth_max, thresh_val, blur_size)
        depth_vis = lr.visualize_depth(dpt, depth_max)
        depth_mask = lr.create_depth_mask(dpt, depth_max)
        if depth_mask.shape[:2] != frame.shape[:2]:
            depth_mask = cv2.resize(depth_mask, (frame.shape[1], frame.shape[0]))

        bright = lr.detect_bright_spots(frame, thresh_val, blur_size)

        cv2.imshow("Result", result)
        cv2.imshow("Depth + Mask", np.hstack([depth_vis,
                   cv2.cvtColor(depth_mask, cv2.COLOR_GRAY2BGR) * 255]))
        cv2.imshow("Bright + Combined", np.hstack([bright, combined]))

        if spot:
            print(f"Laser spot: ({spot[0]}, {spot[1]}) depth={spot[2]:.0f}mm")

        if cv2.waitKey(1) == ord("q"):
            break

    lr.release()
