import cv2
import numpy as np
from openni import openni2
import time
import os
from datetime import datetime


def main():
    # ── 创建保存目录 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"record_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # ── 初始化深度摄像头 ──
    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    # ── 初始化RGB摄像头 ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开RGB摄像头")
        exit()

    # ── RGB视频写入器 ──
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    rgb_writer = cv2.VideoWriter(
        os.path.join(save_dir, "rgb.avi"), fourcc, 30.0, (640, 480))

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

        # ── RGB帧 ──
        ret, rgb = cap.read()
        if not ret or rgb is None:
            continue

        # ── 深度帧（原始16bit，mm单位） ──
        frame = depth_stream.read_frame()
        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
        dpt2 *= 255
        dpt = dpt1 + dpt2
        dpt = dpt[:, ::-1]  # 水平翻转对齐RGB

        # ── 保存 ──
        rgb_writer.write(rgb)
        depth_frames.append(dpt.astype(np.uint16))
        timestamps.append(elapsed)

        # 实时显示
        dpt_norm = cv2.normalize(dpt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dpt_color = cv2.applyColorMap(dpt_norm, cv2.COLORMAP_JET)
        combined = np.hstack([rgb, dpt_color])
        cv2.putText(combined, f"Recording... {elapsed:.1f}s / {DURATION}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Recording", combined)

        if cv2.waitKey(1) == ord('q'):
            print("手动中断")
            break

    # ── 保存深度数据 ──
    print(f"录制完成，共 {len(depth_frames)} 帧，正在保存...")
    depth_array = np.stack(depth_frames, axis=0)  # (N, 480, 640) uint16
    np.savez_compressed(
        os.path.join(save_dir, "depth.npz"),
        depth=depth_array,
        timestamps=np.array(timestamps, dtype=np.float32))
    print(f"深度数据已保存: {save_dir}/depth.npz 形状={depth_array.shape}")

    # ── 清理 ──
    rgb_writer.release()
    depth_stream.stop()
    dev.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"全部保存完毕: {save_dir}/")
    print(f"  rgb.avi      — RGB视频")
    print(f"  depth.npz     — 深度原始数据 (mm), 形状={depth_array.shape}")


if __name__ == "__main__":
    main()
