"""
激光追踪绿色靶标 - 闭环控制系统
==============================================
严格调用三个模块, 通过 CameraManager 统一管理相机:

  CameraManager   → 统一管理 RGB + 深度相机, 分发给各模块
  TargetRecognizer → Phase1 识别三靶标 → Phase2 构建模板 → Phase3 追踪
                     从追踪结果中提取 Green 靶标坐标 (cx, cy)
  LaserRecognizer → 处理帧数据, 检测激光点位置 (cx, cy, depth)
  MotorController → 发送速度指令 (mode 2), 驱动激光追绿色靶标

控制逻辑: 像素误差 → P控制器 → 速度指令 → 闭环伺服

使用方法:
    python laser_track_green.py
    按 q 退出 | 按 r 复位电机 | 按 空格 暂停
"""

import cv2
import numpy as np
import time

from backup_2026_06_05.camera_manager import CameraManager
from backup_2026_06_05.laser_recognize import LaserRecognizer
from backup_2026_06_05.motor_control import MotorController
from backup_2026_06_05.target_recognize import TargetRecognizer


# ==========================================
# 控制参数
# ==========================================
KP_PAN = 0.2        # P: 比例增益 (度/秒 / 像素)
KP_TILT = 0.08
KD_PAN = 0.04        # D: 微分增益 (度/秒 / 像素/帧), 抑制震荡和过冲
KD_TILT = 0.04
TILT_INVERT = -1     # Y轴电机物理方向相反: -1 反转
DEAD_ZONE = 4
MAX_SPEED = 200       # 最大速度 (度/秒)
MAX_ACCEL = 100       # 最大加速度 (度/秒²), 限制速度变化率防止急起急停
SERIAL_PORT = 'COM7'
SERIAL_BAUD = 9600

LASER_THRESH = 240
LASER_BLUR = 1
DEPTH_MAX = 1300


def pixel_to_velocity(ex, ey, dex, dey):
    """像素误差 → 目标电机速度 (PD控制 + 死区 + 限幅)
    P项: 比例控制, error越大速度越大
    D项: 微分控制, error快速缩小时提前减速防止过冲
    TILT_INVERT 处理Y轴电机物理方向反转
    """
    # D项: error正在缩小(de与error同号表示远离, 异号表示靠近)
    # dex > 0 表示激光正在远离靶标 → D项加大速度追赶
    # dex < 0 表示激光正在靠近靶标 → D项减小速度刹车
    if abs(ex) < DEAD_ZONE and abs(ey) < DEAD_ZONE:
        return 0.0, 0.0
    pan = np.clip(ex * KP_PAN + dex * KD_PAN, -MAX_SPEED, MAX_SPEED)
    tilt = np.clip((ey * KP_TILT + dey * KD_TILT) * TILT_INVERT, -MAX_SPEED, MAX_SPEED)
    return pan, tilt


def main():
    print("=" * 60)
    print("  激光追踪绿色靶标 - 闭环视觉伺服控制")
    print("=" * 60)

    # ==========================================
    # 1. 创建 CameraManager (唯一相机入口)
    # ==========================================
    cam = CameraManager()
    cam.start()

    # ==========================================
    # 2. 创建三个模块实例, 统一绑定 CameraManager
    # ==========================================
    tr = TargetRecognizer()
    tr.setup(cam)

    lr = LaserRecognizer(depth_max=DEPTH_MAX,
                         thresh_val=LASER_THRESH,
                         blur_size=LASER_BLUR)
    lr.setup(cam)

    mc = MotorController(port=SERIAL_PORT, baudrate=SERIAL_BAUD)

    # ==========================================
    # 3. 连接电机
    # ==========================================
    print("\n连接电机控制器...")
    motor_ok = True
    try:
        mc.connect()
        mc.reset()
        time.sleep(1)
    except Exception as e:
        print(f"  警告: 无法连接电机 ({e}) → 仅检测模式")
        motor_ok = False

    # ==========================================
    # 4. 窗口 & 滑动条
    # ==========================================
    cv2.namedWindow("Laser Track Green")
    cv2.createTrackbar("Threshold", "Laser Track Green", LASER_THRESH, 255, lambda _: None)
    cv2.createTrackbar("Blur", "Laser Track Green", LASER_BLUR, 20, lambda _: None)
    cv2.createTrackbar("DepthMax_mm", "Laser Track Green", DEPTH_MAX, 5000, lambda _: None)

    print("\n" + "=" * 60)
    print("  按 q 退出 | 按 r 复位 | 按 空格 暂停")
    print("=" * 60 + "\n")

    paused = False
    prev_pan = 0.0
    prev_tilt = 0.0
    prev_ex = 0.0   # 上一帧水平像素误差 (用于D项)
    prev_ey = 0.0   # 上一帧垂直像素误差 (用于D项)

    # ==========================================
    # 5. 主循环: 追踪失败 → 自动重新初始化
    # ==========================================
    while True:

        # -------- Phase 1: TargetRecognizer 严苛初始化捕获三靶标 --------
        print(">>> Phase 1: 等待三靶标 (Red, Green, Blue) 初始化捕获...")
        ref_frame, confirmed_targets = tr.capture_initial_targets()
        if ref_frame is None:
            break

        # -------- Phase 2: TargetRecognizer 构建 ORB 星座模板 --------
        print(">>> Phase 2: 构建 ORB 星座模板...")
        n_kp = tr.build_template(ref_frame, confirmed_targets)
        print(f"  模板就绪, {n_kp} 个特征点")

        # -------- Phase 3: 追踪 + 激光伺服 --------
        print(">>> Phase 3: 实时追踪三靶标 → 激光追 Green\n")

        cv2.namedWindow("Tracking Binary (Masked)")

        while True:
            # --- 从 CameraManager 读取帧 ---
            frame = cam.read_rgb_frame()
            if frame is None:
                break
            dpt = cam.get_depth_map()
            tr.depth_map = dpt

            # --- 滑动条参数 ---
            thresh_val = cv2.getTrackbarPos("Threshold", "Laser Track Green")
            blur_val = cv2.getTrackbarPos("Blur", "Laser Track Green")
            depth_max_val = cv2.getTrackbarPos("DepthMax_mm", "Laser Track Green")
            if blur_val % 2 == 0:
                blur_val += 1

            # ==== TargetRecognizer: ORB追踪三靶标 ====
            gray, binary = tr.preprocess(frame)
            tracking_ok, all_targets, tr_result, rot_angle, _ = \
                tr.track_frame(frame, binary)

            # 追踪丢失 → 重新初始化
            if not tracking_ok and tr.consecutive_tri_fail >= 7:
                print("\n追踪丢失! 返回 Phase 1 重新初始化...\n")
                break

            # ==== LaserRecognizer: 检测激光点 ====
            _, _, laser_spot = lr.process_frame(frame, dpt, depth_max_val,
                                                 thresh_val, blur_val)

            # ==== 提取绿色靶标 → P控制计算目标速度 ====
            green_pos = all_targets.get("Green") if all_targets else None

            if laser_spot is not None and green_pos is not None and not paused:
                lx, ly, _ = laser_spot
                gx, gy = green_pos
                ex, ey = gx - lx, gy - ly
                dex, dey = ex - prev_ex, ey - prev_ey  # 误差变化量 (D项)
                target_pan, target_tilt = pixel_to_velocity(ex, ey, dex, dey)
                prev_ex, prev_ey = ex, ey
            else:
                target_pan, target_tilt = 0.0, 0.0
                prev_ex, prev_ey = 0.0, 0.0  # 目标丢失时重置D项

            # ==== 加速度限制: 平滑过渡到目标速度 ====
            # 每帧最大速度变化量 = MAX_ACCEL * dt (假设 ~30fps → dt≈0.033s)
            max_step = MAX_ACCEL * 0.033

            # Pan: 速率限制
            pan_diff = target_pan - prev_pan
            pan_diff = np.clip(pan_diff, -max_step, max_step)
            pan_speed = prev_pan + pan_diff

            # Tilt: 速率限制
            tilt_diff = target_tilt - prev_tilt
            tilt_diff = np.clip(tilt_diff, -max_step, max_step)
            tilt_speed = prev_tilt + tilt_diff

            # ==== MotorController: 发送速率限制后的速度指令 ====
            speed_changed = (abs(pan_speed - prev_pan) > 0.1 or
                             abs(tilt_speed - prev_tilt) > 0.1)
            if speed_changed and motor_ok and not paused:
                try:
                    mc.set_velocity(pan_speed, tilt_speed)
                except Exception as e:
                    print(f"电机通信错误: {e}")

            prev_pan = pan_speed
            prev_tilt = tilt_speed

            # ========== 可视化 ==========
            # 底图: track_frame 已绘制三靶标 + 观察圈
            display = tr_result

            # 叠加激光点
            if laser_spot is not None:
                lx, ly, ld = laser_spot
                cv2.circle(display, (lx, ly), 15, (0, 0, 255), 2)
                cv2.circle(display, (lx, ly), 3, (0, 0, 255), -1)
                cv2.putText(display, f"Laser ({lx},{ly}) d={ld:.0f}mm",
                            (lx + 20, ly - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 激光 → 绿色靶标连线
            if laser_spot is not None and green_pos is not None:
                lx, ly, _ = laser_spot
                gx, gy = green_pos
                cv2.line(display, (lx, ly), (gx, gy), (255, 255, 0), 1, cv2.LINE_AA)
                ex, ey = gx - lx, gy - ly
                mx, my = (lx + gx) // 2, (ly + gy) // 2
                cv2.putText(display, f"Err({ex},{ey})", (mx + 5, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            # 状态栏
            status = f"Pan:{pan_speed:+.1f}/s  Tilt:{tilt_speed:+.1f}/s"
            cv2.putText(display, status, (10, display.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if (abs(pan_speed) < 0.01 and abs(tilt_speed) < 0.01
                    and laser_spot and green_pos):
                cv2.putText(display, "ON TARGET", (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if paused:
                cv2.putText(display, "PAUSED",
                            (display.shape[1] // 2 - 50, display.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # 追踪二值图 (遮罩圈内)
            binary_masked = np.zeros_like(binary)
            cv2.circle(binary_masked, tr.ref_mask_center, tr.ref_mask_radius, 255, -1)
            binary_masked = cv2.bitwise_and(binary, binary_masked)

            cv2.imshow("Laser Track Green", display)
            cv2.imshow("Depth", lr.visualize_depth(dpt, depth_max_val))
            cv2.imshow("Tracking Binary (Masked)", binary_masked)

            # ========== 键盘 ==========
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n正在退出...")
                if motor_ok:
                    try:
                        mc.set_velocity(0.0, 0.0)
                        time.sleep(0.2)
                        mc.disconnect()
                    except Exception:
                        pass
                cam.release()
                print("已退出。")
                return
            elif key == ord('r'):
                print("复位电机...")
                if motor_ok:
                    try:
                        mc.reset()
                    except Exception as e:
                        print(f"复位失败: {e}")
                prev_pan = 0.0
                prev_tilt = 0.0
            elif key == ord(' '):
                paused = not paused
                print("已暂停" if paused else "继续追踪")
                if motor_ok:
                    try:
                        mc.set_velocity(0.0, 0.0)
                    except Exception:
                        pass
                prev_pan = 0.0
                prev_tilt = 0.0

        cv2.destroyWindow("Tracking Binary (Masked)")


if __name__ == "__main__":
    main()