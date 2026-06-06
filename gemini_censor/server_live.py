"""
硬件实时发布服务端 (server_live.py)
连接实体 RGB 和深度摄像头，将画面实时发布到本地端口。
这个程序可以挂在一个独立的终端里永远不关。
"""

import cv2
import numpy as np
from openni import openni2
import zmq
import pickle
import time

def main():
    # 1. 配置 ZeroMQ 发布者 (Publisher)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    # 绑定到本地 5555 端口
    socket.bind("tcp://*:5555")
    print("Live Server: 数据发布端口已绑定至 tcp://*:5555")

    # 2. 初始化硬件 (原 camera_manager 的启动逻辑)
    print("Live Server: 正在初始化实体相机硬件...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开RGB摄像头")

    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    print("Live Server: 硬件就绪，正在持续发布数据... (按 Ctrl+C 退出)")

    try:
        while True:
            # 读取 RGB
            ret, rgb = cap.read()
            if not ret or rgb is None:
                continue

            # 读取 Depth
            depth_frame = depth_stream.read_frame()
            dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
            dpt1 = np.asarray(dframe_data[:, :, 0], dtype="float32")
            dpt2 = np.asarray(dframe_data[:, :, 1], dtype="float32")
            dpt = (dpt1 + dpt2 * 255)[:, ::-1] # 水平翻转对齐RGB

            # 3. 将 RGB 和 Depth 打包并发布
            data_dict = {'rgb': rgb, 'depth': dpt}
            # 使用 pickle 序列化发送，局域网/本机传输速度极快
            socket.send(pickle.dumps(data_dict, protocol=-1))
            
    except KeyboardInterrupt:
        print("\nLive Server: 收到终止信号，正在安全释放硬件...")
    finally:
        cap.release()
        depth_stream.stop()
        dev.close()
        socket.close()
        context.term()
        print("Live Server: 已退出。")

if __name__ == "__main__":
    main()