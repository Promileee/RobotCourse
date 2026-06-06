"""
record_data_v1.py
通过 CameraManager (ZMQ 客户端) 订阅服务端的实时/回放数据流并录制。
运行前请先启动 server_live.py 或 server_playback.py。
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
from camera_manager import CameraManager


def main():
    # ── 创建保存目录 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"record_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # ── 连接数据流 ──
    cam = CameraManager()
    cam.start()
    print("等待服务端数据...")

    # ── 等待第一帧以确定分辨率 ──
    frame = cam.read_rgb_frame()
    if frame is None:
        print("未能接收到数据，请确认已启动 server_live.py 或 server_playback.py")
        cam.release()
        return
    h, w = frame.shape[:2]

    # ── RGB视频写入器 ──
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    rgb_writer = cv2.VideoWriter(
        os.path.join(save_dir, "rgb.avi"), fourcc, 30.0, (w, h))

    # ── 深度帧缓存 ──
    depth_frames = []
    timestamps = []

    DURATION = 60  # 秒
    print(f"开始录制 {DURATION} 秒... 保存到 {save_dir}/")
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > DURATION:
            break

        frame = cam.read_rgb_frame()
        dpt = cam.get_depth_map()

        if frame is None:
            continue

        # ── 保存 ──
        rgb_writer.write(frame)
        depth_frames.append(dpt.astype(np.uint16))
        timestamps.append(elapsed)

        # 实时显示
        dpt_clipped = np.clip(dpt, 0, 3000)
        dpt_norm = cv2.normalize(dpt_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dpt_color = cv2.applyColorMap(dpt_norm, cv2.COLORMAP_JET)
        combined = np.hstack([frame, dpt_color])
        cv2.putText(combined, f"Recording... {elapsed:.1f}s / {DURATION}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Recording", combined)

        if cv2.waitKey(1) == ord('q'):
            print("手动中断")
            break

    # ── 保存深度数据 ──
    print(f"录制完成，共 {len(depth_frames)} 帧，正在保存...")
    depth_array = np.stack(depth_frames, axis=0)  # (N, H, W) uint16
    np.savez_compressed(
        os.path.join(save_dir, "depth.npz"),
        depth=depth_array,
        timestamps=np.array(timestamps, dtype=np.float32))
    print(f"深度数据已保存: {save_dir}/depth.npz 形状={depth_array.shape}")

    # ── 清理 ──
    rgb_writer.release()
    cam.release()

    print(f"全部保存完毕: {save_dir}/")
    print(f"  rgb.avi      — RGB视频")
    print(f"  depth.npz     — 深度原始数据 (mm), 形状={depth_array.shape}")


if __name__ == "__main__":
    main()