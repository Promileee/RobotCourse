"""
激光追踪靶标 - 闭环控制系统 (可选靶标版)
==============================================
严格调用三个模块, 通过 CameraManager 统一管理相机:

  CameraManager   → 统一管理 RGB + 深度相机, 分发给各模块
  TargetRecognizer → Phase1 识别三靶标 → Phase2 构建模板 → Phase3 追踪
                     从EMA平滑结果中提取选定靶标坐标 (cx, cy)
  LaserRecognizerV3 → 处理帧数据, 检测激光点位置 (cx, cy, depth, quality)
  MotorController   → 发送速度指令 (mode 2), 驱动激光追踪选定靶标

控制逻辑: 像素误差 → PD控制器 → 速度指令 → 闭环伺服

使用方法:
    python close_loop_control_v1.py
    启动后输入 r/g/b 选择追踪靶标
    追踪中按 1/2/3 实时切换靶标 | 按 q 输入新颜色 | 按 r 复位 | 按 空格 暂停
    颜色输入时直接回车或输入 quit 退出程序
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

COLOR_MAP = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}
NUM_KEY_MAP = {ord('1'): 'Red', ord('2'): 'Green', ord('3'): 'Blue'}


def pixel_to_velocity(ex, ey, dex, dey):
    """像素误差 → 目标电机速度 (PD控制 + 死区 + 限幅)"""
    if abs(ex) < DEAD_ZONE and abs(ey) < DEAD_ZONE:
        return 0.0, 0.0
    pan = np.clip(ex * KP_PAN + dex * KD_PAN, -MAX_SPEED, MAX_SPEED)
    tilt = np.clip((ey * KP_TILT + dey * KD_TILT) * TILT_INVERT, -MAX_SPEED, MAX_SPEED)
    return pan, tilt


def prompt_target_color(current=None):
    """阻塞式输入靶标颜色。回车或 quit 返回 None。"""
    hint = f" (当前: {current})" if current else ""
    raw = input(f"请输入靶标颜色 (r=Red, g=Green, b=Blue, quit=退出){hint}: ").strip().lower()
    if raw == '' or raw == 'quit':
        return None
    if raw in COLOR_MAP:
        return COLOR_MAP[raw]
    print(f"  无效输入 '{raw}'，保持当前靶标")
    return None


def main():
    print("=" * 60)
    print("  激光追踪靶标 - 闭环视觉伺服控制")
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
    WIN_NAME = "Laser Track"
    cv2.namedWindow(WIN_NAME)
    cv2.createTrackbar("Threshold", WIN_NAME, LASER_THRESH, 255, lambda _: None)
    cv2.createTrackbar("Blur", WIN_NAME, LASER_BLUR, 20, lambda _: None)
    cv2.createTrackbar("DepthMax_mm", WIN_NAME, DEPTH_MAX, 5000, lambda _: None)

    print("\n" + "=" * 60)
    print("  按 1/2/3 实时切换靶标 | 按 q 输入新颜色 | 按 r 复位 | 按 空格 暂停")
    print("=" * 60 + "\n")

    # ==========================================
    # 5. 初始颜色选择
    # ==========================================
    target_color = None
    while target_color is None:
        target_color = prompt_target_color()
        if target_color is None:
            print("正在退出...")
            cam.release()
            if motor_ok:
                try:
                    mc.disconnect()
                except Exception:
                    pass
            return
    print(f"  已选择: {target_color}\n")

    paused = False
    prev_pan = 0.0
    prev_tilt = 0.0
    prev_ex = 0.0
    prev_ey = 0.0
    last_laser_time = time.time()

    # ==========================================
    # 6. 追踪主循环: 失败自动重新初始化
    # ==========================================
    while True:

        # -------- Phase 1: 严苛初始化捕获三靶标 --------
        print(">>> Phase 1: 等待三靶标 (Red, Green, Blue) 初始化捕获...")
        ref_frame, confirmed_targets = tr.capture_initial_targets()
        if ref_frame is None:
            break

        # -------- Phase 2: 构建 ORB 星座模板 --------
        print(">>> Phase 2: 构建 ORB 星座模板...")
        n_kp = tr.build_template(ref_frame, confirmed_targets)
        print(f"  模板就绪, {n_kp} 个特征点")

        # -------- Phase 2.5: 初始化 LaserRecognizerV3 --------
        dpt_init = cam.get_depth_map()
        lr.init_from_target_data(tr.ref_targets_centers, dpt_init)

        # -------- Phase 3: 追踪 + 激光伺服 --------
        print(f">>> Phase 3: 实时追踪三靶标 → 激光追 {target_color}\n")

        cv2.namedWindow("Tracking Binary (Masked)")

        while True:
            frame = cam.read_rgb_frame()
            if frame is None:
                break
            dpt = cam.get_depth_map()
            tr.depth_map = dpt

            # --- 滑动条参数 ---
            thresh_val = cv2.getTrackbarPos("Threshold", WIN_NAME)
            blur_val = cv2.getTrackbarPos("Blur", WIN_NAME)
            depth_max_val = cv2.getTrackbarPos("DepthMax_mm", WIN_NAME)
            if blur_val % 2 == 0:
                blur_val += 1

            lr.thresh_val = thresh_val
            lr.blur_size = blur_val
            lr.depth_max = depth_max_val

            # ==== TargetRecognizer: ORB追踪三靶标 ====
            gray, binary = tr.preprocess(frame)
            tracking_ok, all_targets, tr_result, rot_angle, _ = \
                tr.track_frame(frame, binary)

            if not tracking_ok and tr.consecutive_tri_fail >= 7:
                print("\n追踪丢失! 返回 Phase 1 重新初始化...\n")
                lr.reset()
                break

            # ==== LaserRecognizerV3: 检测激光点 ====
            _, _, laser_spot, _ = lr.process_frame(frame, dpt)

            # ==== 提取选定靶标 → PD控制计算目标速度 ====
            target_pos = all_targets.get(target_color) if all_targets else None

            if laser_spot is not None and target_pos is not None and not paused:
                lx, ly = laser_spot[0], laser_spot[1]
                gx, gy = target_pos
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

            if laser_spot is not None and target_pos is not None:
                lx, ly = laser_spot[0], laser_spot[1]
                gx, gy = target_pos
                cv2.line(display, (lx, ly), (gx, gy), (255, 255, 0), 1, cv2.LINE_AA)
                ex, ey = gx - lx, gy - ly
                mx, my = (lx + gx) // 2, (ly + gy) // 2
                cv2.putText(display, f"Err({ex},{ey})", (mx + 5, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            status = f"Target:{target_color}  Pan:{pan_speed:+.1f}/s  Tilt:{tilt_speed:+.1f}/s"
            cv2.putText(display, status, (10, display.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if (abs(pan_speed) < 0.01 and abs(tilt_speed) < 0.01
                    and laser_spot and target_pos):
                cv2.putText(display, "ON TARGET", (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if paused:
                cv2.putText(display, "PAUSED",
                            (display.shape[1] // 2 - 50, display.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            binary_masked = np.zeros_like(binary)
            cv2.circle(binary_masked, tr.ref_mask_center, tr.ref_mask_radius, 255, -1)
            binary_masked = cv2.bitwise_and(binary, binary_masked)

            cv2.imshow(WIN_NAME, display)
            cv2.imshow("Depth", lr.visualize_depth(dpt, depth_max_val))
            cv2.imshow("Tracking Binary (Masked)", binary_masked)

            # ========== 键盘 ==========
            key = cv2.waitKey(1) & 0xFF
            if key in NUM_KEY_MAP:
                # 数字键 1/2/3: 实时切换靶标，无需重识别
                new_color = NUM_KEY_MAP[key]
                if new_color != target_color:
                    target_color = new_color
                    print(f"  切换靶标 → {target_color}")
                    prev_ex, prev_ey = 0.0, 0.0
                    prev_pan, prev_tilt = 0.0, 0.0
                    if motor_ok:
                        try:
                            mc.set_velocity(0.0, 0.0)
                        except Exception:
                            pass
            elif key == ord('q'):
                # q: 停止电机，阻塞式输入新颜色
                if motor_ok:
                    try:
                        mc.set_velocity(0.0, 0.0)
                    except Exception:
                        pass
                prev_pan, prev_tilt = 0.0, 0.0
                paused = True
                new_color = prompt_target_color(target_color)
                if new_color is None:
                    # 回车或 quit → 退出程序
                    print("\n正在退出...")
                    if motor_ok:
                        try:
                            mc.disconnect()
                        except Exception:
                            pass
                    cam.release()
                    print("已退出。")
                    return
                target_color = new_color
                prev_ex, prev_ey = 0.0, 0.0
                last_laser_time = time.time()
                paused = False
                print(f"  继续追踪: {target_color}\n")
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

    # ==========================================
    # 7. 清理退出
    # ==========================================
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


if __name__ == "__main__":
    main()
