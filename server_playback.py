"""
数据回放发布服务端 (server_playback.py)
读取本地记录的 RGB 和 Depth 数据，伪装成实时硬件发布到本地端口。
离开实验室时，运行此程序替代 server_live.py。
"""

import cv2
import numpy as np
import zmq
import pickle
import time
import os

# ============== 配置区 ==============
# 在这里填入你保存的数据文件夹相对路径
RECORD_DIR = r"debug\record_20260603_162503"  
# ====================================

def main():
    if not os.path.exists(RECORD_DIR):
        raise FileNotFoundError(f"找不到指定的录制文件夹: {RECORD_DIR}")

    # 1. 配置 ZeroMQ 发布者
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    print("Playback Server: 数据发布端口已绑定至 tcp://*:5555")

    # 2. 加载离线数据
    print(f"Playback Server: 正在加载数据 {RECORD_DIR} ...")
    avi_path = os.path.join(RECORD_DIR, "rgb.avi")
    npz_path = os.path.join(RECORD_DIR, "depth.npz")

    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开离线 RGB 视频")
    
    depth_data = np.load(npz_path)['depth']
    total_frames = depth_data.shape[0]
    print(f"Playback Server: 数据加载完毕，共 {total_frames} 帧。开始循环广播...")

    frame_idx = 0
    # 按照 30 fps 控制回放速度
    delay = 1.0 / 30.0 

    try:
        while True:
            ret, rgb = cap.read()
            
            # 视频播放完毕，循环重置
            if not ret or frame_idx >= total_frames:
                print("Playback Server: 播放完毕，从头开始循环...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            # 取出对应的深度图 (保存时是 uint16, 算法中用的是 float32)
            dpt = depth_data[frame_idx].astype("float32")

            # 3. 发布数据
            data_dict = {'rgb': rgb, 'depth': dpt}
            socket.send(pickle.dumps(data_dict, protocol=-1))

            frame_idx += 1
            time.sleep(delay)  # 模拟真实的相机帧率间隔

    except KeyboardInterrupt:
        print("\nPlayback Server: 停止回放。")
    finally:
        cap.release()
        socket.close()
        context.term()

if __name__ == "__main__":
    main()