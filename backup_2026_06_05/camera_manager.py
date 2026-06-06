"""
统一相机管理模块
集中管理 RGB 相机和深度相机，供各识别模块共用。

使用方法:
    from camera_manager import CameraManager
    cam = CameraManager()
    cam.start()
    frame = cam.read_rgb_frame()
    dpt = cam.get_depth_map()
    cam.release()
"""

import cv2
import numpy as np
from openni import openni2


class CameraManager:
    """RGB + 深度相机统一管理器"""

    def __init__(self):
        self.cap = None
        self.dev = None
        self.depth_stream = None
        self._started = False

    def start(self, rgb_index=0, skip_n=5):
        """初始化相机并跳过稳定帧"""
        if self._started:
            print("CameraManager: 相机已启动，跳过重复初始化")
            return

        print("CameraManager: 正在初始化相机...")

        # RGB 相机
        self.cap = cv2.VideoCapture(rgb_index)
        if not self.cap.isOpened():
            raise RuntimeError("CameraManager: 无法打开RGB摄像头")

        # 深度相机
        openni2.initialize()
        self.dev = openni2.Device.open_any()
        print(f"CameraManager: 深度设备 — {self.dev.get_device_info()}")
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.start()

        # 跳过稳定帧
        print(f"CameraManager: 跳过前 {skip_n} 帧以稳定传感器...")
        for _ in range(skip_n):
            self.cap.read()
            self.depth_stream.read_frame()
            cv2.waitKey(100)

        self._started = True
        print("CameraManager: 就绪")

    def read_rgb_frame(self):
        """读取RGB帧，失败返回 None"""
        if self.cap is None:
            raise RuntimeError("CameraManager: 相机未启动，请先调用 start()")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def get_depth_map(self):
        """读取深度帧并返回深度图 (mm)，shape=(480, 640)"""
        if self.depth_stream is None:
            raise RuntimeError("CameraManager: 深度相机未启动，请先调用 start()")
        depth_frame = self.depth_stream.read_frame()
        dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype="float32")
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype="float32")
        dpt2 *= 255
        dpt = dpt1 + dpt2
        dpt = dpt[:, ::-1]
        return dpt

    def release(self):
        """释放所有相机资源"""
        print("CameraManager: 释放相机...")
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.depth_stream is not None:
            self.depth_stream.stop()
            self.depth_stream = None
        if self.dev is not None:
            self.dev.close()
            self.dev = None
        self._started = False
        cv2.destroyAllWindows()


# ==========================================
# 调试入口: 显示 RGB + 深度画面
# ==========================================
if __name__ == "__main__":
    cam = CameraManager()
    cam.start()

    cv2.namedWindow("RGB")
    cv2.namedWindow("Depth")

    while True:
        frame = cam.read_rgb_frame()
        dpt = cam.get_depth_map()

        if frame is None:
            print("无法读取RGB帧")
            break

        # 深度可视化
        dpt_clipped = np.clip(dpt, 0, 3000)
        depth_vis = cv2.normalize(dpt_clipped, None, 0, 255,
                                  cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.imshow("RGB", frame)
        cv2.imshow("Depth", depth_vis)

        if cv2.waitKey(1) == ord("q"):
            break

    cam.release()
