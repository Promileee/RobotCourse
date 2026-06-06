"""
统一相机管理模块 (客户端)
通过 ZeroMQ 订阅服务端 (Live 或 Playback) 发布的数据。
API 接口与原版完全一致，实现无感替换。
"""

import cv2
import numpy as np
import zmq
import pickle

class CameraManager:
    """RGB + 深度相机统一管理器 (ZMQ 客户端版本)"""

    def __init__(self):
        self.context = None
        self.socket = None
        self._started = False
        
        # 用于缓存同一帧的深度图
        self._current_depth = None

    def start(self, rgb_index=0, skip_n=5):
        """
        初始化网络接收端口。
        (保留 rgb_index 和 skip_n 参数是为了保持兼容性，实际在此版本中不再需要硬件初始化)
        """
        if self._started:
            return

        print("CameraManager (Client): 正在连接数据流端口...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
        # 订阅所有主题
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        # 核心设置：CONFLATE 设为 1，确保永远只接收最新的一帧，丢弃积压的旧帧，保证零延迟
        self.socket.setsockopt(zmq.CONFLATE, 1) 
        
        # 连接到本机的发布端口
        self.socket.connect("tcp://localhost:5555")
        self._started = True
        print("CameraManager (Client): 已成功连接！")

    def read_rgb_frame(self):
        """阻塞读取最新的 RGB 帧，并同时缓存配套的深度帧"""
        if not self._started:
            raise RuntimeError("CameraManager: 请先调用 start()")
        
        try:
            # recv 会阻塞，直到服务端发来新数据。这能完美同步算法和摄像头的频率
            raw_data = self.socket.recv()
            data_dict = pickle.loads(raw_data)
            
            # 保存这一帧对应的深度图，供 get_depth_map 调用
            self._current_depth = data_dict['depth']
            return data_dict['rgb']
            
        except Exception as e:
            print(f"CameraManager 读取数据异常: {e}")
            return None

    def get_depth_map(self):
        """返回与上一次 read_rgb_frame 完全同步的深度图"""
        if not self._started:
            raise RuntimeError("CameraManager: 请先调用 start()")
        if self._current_depth is None:
            # 如果算法在没读 RGB 的情况下强行先读深度，返回空矩阵防止报错
            return np.zeros((480, 640), dtype="float32")
        return self._current_depth

    def release(self):
        """断开网络连接"""
        print("CameraManager (Client): 断开连接...")
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None
        self._started = False
        cv2.destroyAllWindows()


# ==========================================
# 测试入口: 测试接收画面
# ==========================================
if __name__ == "__main__":
    cam = CameraManager()
    cam.start()

    cv2.namedWindow("RGB (Client)")
    cv2.namedWindow("Depth (Client)")

    print("正在等待服务端发送数据... (请确保已运行 server_live 或 server_playback)")

    while True:
        frame = cam.read_rgb_frame()
        dpt = cam.get_depth_map()

        if frame is None:
            break

        dpt_clipped = np.clip(dpt, 0, 3000)
        depth_vis = cv2.normalize(dpt_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.imshow("RGB (Client)", frame)
        cv2.imshow("Depth (Client)", depth_vis)

        if cv2.waitKey(1) == ord("q"):
            break

    cam.release()