"""
激光追踪绿色靶标 - 闭环控制系统 (更新版)
==============================================
严格调用三个模块, 通过 CameraManager 统一管理相机:

  CameraManager   → 统一管理 RGB + 深度相机, 分发给各模块
  TargetRecognizer → Phase1 识别三靶标 → Phase2 构建模板 → Phase3 追踪
                     从EMA平滑结果中提取 Green 靶标坐标 (cx, cy)
  LaserRecognizerV3 → 处理帧数据, 检测激光点位置 (cx, cy, depth, quality)
  MotorController   → 发送速度指令 (mode 2), 驱动激光追绿色靶标

控制逻辑: 像素误差 → PD控制器 → 速度指令 → 闭环伺服

与 laser_track_green.py 的差异:
  - TargetRecognizerV2 → TargetRecognizer (EMA平滑+异常帧剔除)
  - LaserRecognizer → LaserRecognizerV3 (深度邻域搜索 + 多候选跟踪)
  - LaserRecognizerV3 需要 init_from_target_data() 初始化靶标平面深度

使用方法:
    python close_loop_control.py
    按 q 退出 | 按 r 复位电机 | 按 空格 暂停
"""

import cv2
import numpy as np
import time

from camera_manager import CameraManager
from laser_recognize_v3 import LaserRecognizerV3
from motor_control import MotorController
from target_recognize_v1 import TargetRecognizer


# ==========================================
# 控制参数
# ==========================================
KP_PAN = 0.12
KP_TILT = 0.12
KD_PAN = 0.08
KD_TILT = 0.08
TILT_INVERT = -1
DEAD_ZONE = 5
MAX_SPEED = 200
MAX_ACCEL = 100
SERIAL_PORT = 'COM7'
SERIAL_BAUD = 9600

LASER_THRESH = 225
LASER_BLUR = 1
DEPTH_MAX = 1300


def pixel_to_velocity(ex, ey, dex, dey):
    """像素误差 → 目标电机速度 (PD控制 + 死区 + 限幅)"""
    if abs(ex) < DEAD_ZONE and abs(ey) < DEAD_ZONE:
        return 0.0, 0.0
    pan = np.clip(ex * KP_PAN + dex * KD_PAN, -MAX_SPEED, MAX_SPEED)
    tilt = np.clip((ey * KP_TILT + dey * KD_TILT) * TILT_INVERT, -MAX_SPEED, MAX_SPEED)
    return pan, tilt


def main():
    print("=" * 60)
    print("  激光追踪绿色靶标 - 闭环视觉伺服控制 (V2/V3)")
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

    lr = LaserRecognizerV3(depth_max=DEPTH_MAX,
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
    prev_ex = 0.0
    prev_ey = 0.0
    last_laser_time = time.time()

    # ==========================================
    # 5. 主循环: 追踪失败 → 自动重新初始化
    # ==========================================
    while True:

        # -------- Phase 1: TargetRecognizerV2 严苛初始化捕获三靶标 --------
        print(">>> Phase 1: 等待三靶标 (Red, Green, Blue) 初始化捕获...")
        ref_frame, confirmed_targets = tr.capture_initial_targets()
        if ref_frame is None:
            break

        # -------- Phase 2: TargetRecognizerV2 构建 ORB 星座模板 --------
        print(">>> Phase 2: 构建 ORB 星座模板...")
        n_kp = tr.build_template(ref_frame, confirmed_targets)
        print(f"  模板就绪, {n_kp} 个特征点")

        # -------- Phase 2.5: 初始化 LaserRecognizerV3 靶标平面数据 --------
        dpt_init = cam.get_depth_map()
        lr.init_from_target_data(tr.ref_targets_centers, dpt_init)

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

            # 更新 LaserRecognizerV3 实例参数 (替代旧版 per-frame 传参)
            lr.thresh_val = thresh_val
            lr.blur_size = blur_val
            lr.depth_max = depth_max_val

            # ==== TargetRecognizerV2: ORB追踪三靶标 (含中值+卡尔曼滤波) ====
            gray, binary = tr.preprocess(frame)
            tracking_ok, all_targets, tr_result, rot_angle, _ = \
                tr.track_frame(frame, binary)

            # 追踪丢失 → 重新初始化
            if not tracking_ok and tr.consecutive_tri_fail >= 7:
                print("\n追踪丢失! 返回 Phase 1 重新初始化...\n")
                lr.reset()
                break

            # ==== LaserRecognizerV3: 检测激光点 ====
            _, _, laser_spot, _ = lr.process_frame(frame, dpt)

            # ==== 提取绿色靶标 → PD控制计算目标速度 ====
            green_pos = all_targets.get("Green") if all_targets else None

            if laser_spot is not None and green_pos is not None and not paused:
                # LaserRecognizerV3 spot: (cx, cy, depth, quality)
                lx, ly = laser_spot[0], laser_spot[1]
                gx, gy = green_pos
                ex, ey = gx - lx, gy - ly
                dex, dey = ex - prev_ex, ey - prev_ey
                target_pan, target_tilt = pixel_to_velocity(ex, ey, dex, dey)
                prev_ex, prev_ey = ex, ey
                last_laser_time = time.time()
            else:
                target_pan, target_tilt = 0.0, 0.0
                prev_ex, prev_ey = 0.0, 0.0

            # ==== 激光超时 1s → 复位电机 ====
            if time.time() - last_laser_time > 1.0 and motor_ok and not paused:
                print("激光丢失超过1s，复位电机...")
                try:
                    mc.reset()
                except Exception as e:
                    print(f"复位失败: {e}")
                last_laser_time = time.time()
                prev_pan = 0.0
                prev_tilt = 0.0

            # ==== 加速度限制 ====
            max_step = MAX_ACCEL * 0.033

            pan_diff = target_pan - prev_pan
            pan_diff = np.clip(pan_diff, -max_step, max_step)
            pan_speed = prev_pan + pan_diff

            tilt_diff = target_tilt - prev_tilt
            tilt_diff = np.clip(tilt_diff, -max_step, max_step)
            tilt_speed = prev_tilt + tilt_diff

            # ==== MotorController: 发送速度指令 ====
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
            display = tr_result

            if laser_spot is not None:
                lx, ly = laser_spot[0], laser_spot[1]
                ld = laser_spot[2]
                cv2.circle(display, (lx, ly), 15, (0, 0, 255), 2)
                cv2.circle(display, (lx, ly), 3, (0, 0, 255), -1)
                cv2.putText(display, f"Laser ({lx},{ly}) d={ld:.0f}mm",
                            (lx + 20, ly - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 激光 → 绿色靶标连线
            if laser_spot is not None and green_pos is not None:
                lx, ly = laser_spot[0], laser_spot[1]
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