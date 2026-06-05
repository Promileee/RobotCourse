"""
激光追踪绿色靶标 - 闭环控制系统
==============================================
严格遵循 target_recognize 的完整识别-追踪管线:
  Phase 1: 严苛初始化 → 识别红、绿、蓝三靶标
  Phase 2: 构建 ORB 星座模板
  Phase 3: 实时刚体追踪 → 提取绿色靶标坐标
  同时调用 laser_recognize 检测激光点, motor_control 闭环控制电机,
  驱动激光点追踪绿色靶标。

使用方法:
    python laser_track_green.py
    按 q 退出, 按 r 复位电机, 按 空格 暂停/继续
"""

import cv2
import numpy as np
import time

from laser_recognize import LaserRecognizer
from motor_control import MotorController
from target_recognize import TargetRecognizer


# ==========================================
# 控制参数
# ==========================================
KP_PAN = 0.08         # 水平比例增益 (度/秒 / 像素)
KP_TILT = 0.08        # 俯仰比例增益
DEAD_ZONE = 12        # 死区 (像素)，误差小于此值不运动
MAX_SPEED = 30        # 最大速度限制 (度/秒)
SERIAL_PORT = 'COM7'  # 电机串口
SERIAL_BAUD = 9600


def pixel_to_velocity(ex, ey, kp_pan=KP_PAN, kp_tilt=KP_TILT, dead_zone=DEAD_ZONE,
                      max_speed=MAX_SPEED):
    """将像素误差转换为电机速度指令 (pan_speed, tilt_speed)"""
    if abs(ex) < dead_zone and abs(ey) < dead_zone:
        return 0.0, 0.0

    pan_speed = np.clip(ex * kp_pan, -max_speed, max_speed)
    tilt_speed = np.clip(ey * kp_tilt, -max_speed, max_speed)
    return pan_speed, tilt_speed


class LaserTrackController:
    """整合三模块的激光追踪控制器"""

    def __init__(self):
        self.tr = TargetRecognizer()
        self.lr = LaserRecognizer(depth_max=1300, thresh_val=235, blur_size=3)
        self.mc = MotorController(port=SERIAL_PORT, baudrate=SERIAL_BAUD)

        self.paused = False
        self.prev_pan_speed = 0.0
        self.prev_tilt_speed = 0.0

    # ==========================================
    # 初始化
    # ==========================================
    def setup(self):
        """初始化相机和电机"""
        print("=" * 60)
        print("  激光追踪绿色靶标 - 闭环视觉伺服控制")
        print("=" * 60)

        # 相机由 TargetRecognizer 统一管理
        self.tr.setup()

        # 电机
        print("\n连接电机控制器...")
        try:
            self.mc.connect()
            print("  MotorController: 已连接")
            self.mc.reset()
            time.sleep(1)
        except Exception as e:
            print(f"  警告: 无法连接电机 ({e})")
            print("  将进入仅检测模式 (不发送电机指令)")
            self.mc = None

        # 窗口
        cv2.namedWindow("Laser Track Green")
        cv2.createTrackbar("Threshold", "Laser Track Green", 235, 255, lambda _: None)
        cv2.createTrackbar("Blur", "Laser Track Green", 3, 20, lambda _: None)
        cv2.createTrackbar("DepthMax_mm", "Laser Track Green", 1300, 5000, lambda _: None)

        print("\n" + "=" * 60)
        print("  按 q 退出 | 按 r 复位 | 按 空格 暂停")
        print("=" * 60 + "\n")

    # ==========================================
    # 可视化
    # ==========================================
    def draw_overlay(self, frame, all_targets, laser_spot, pan_speed, tilt_speed):
        """在图像上绘制三靶标、激光点、控制信息"""
        result = frame.copy()
        colors_bgr = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}

        # 绘制全部三个靶标
        if all_targets:
            pts = []
            for color, (cx, cy) in all_targets.items():
                bgr = colors_bgr.get(color, (255, 255, 255))
                pts.append((cx, cy))
                cv2.circle(result, (cx, cy), 15, bgr, 2)
                cv2.circle(result, (cx, cy), 4, (255, 255, 255), -1)
                label = f"{color}"
                if color == "Green":
                    # 绿色靶标特别标注：这是追踪目标
                    d_val = (self.tr.depth_map[cy, cx] if self.tr.depth_map is not None
                             and 0 <= cy < self.tr.depth_map.shape[0]
                             and 0 <= cx < self.tr.depth_map.shape[1] else 0)
                    label = f"TARGET {color} {d_val:.0f}mm"
                    cv2.circle(result, (cx, cy), 22, (0, 255, 255), 2)  # 额外黄色圈
                cv2.putText(result, label, (cx - 30, cy - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
            if len(pts) == 3:
                cv2.polylines(result, [np.array(pts)], isClosed=True,
                              color=(255, 255, 255), thickness=1)

        # 绘制固定观察圈
        if self.tr.ref_mask_center is not None:
            cv2.circle(result, self.tr.ref_mask_center, self.tr.ref_mask_radius,
                       (0, 255, 255), 1, cv2.LINE_AA)

        # 绘制激光点
        if laser_spot is not None:
            lx, ly, _ = laser_spot
            cv2.circle(result, (lx, ly), 15, (0, 0, 255), 2)
            cv2.circle(result, (lx, ly), 3, (0, 0, 255), -1)
            cv2.putText(result, f"Laser ({lx},{ly})", (lx + 20, ly - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 激光 → 绿色靶标连线
            if all_targets and "Green" in all_targets:
                gx, gy = all_targets["Green"]
                cv2.line(result, (lx, ly), (gx, gy), (255, 255, 0), 1, cv2.LINE_AA)
                ex, ey = gx - lx, gy - ly
                mid_x, mid_y = (lx + gx) // 2, (ly + gy) // 2
                cv2.putText(result, f"Err({ex},{ey})", (mid_x + 5, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # 状态栏
        status = f"Pan:{pan_speed:+.1f}/s  Tilt:{tilt_speed:+.1f}/s"
        cv2.putText(result, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if abs(pan_speed) < 0.01 and abs(tilt_speed) < 0.01:
            cv2.putText(result, "ON TARGET", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        return result

    # ==========================================
    # 电机控制
    # ==========================================
    def control_motor(self, laser_spot, green_pos):
        """根据激光和绿色靶标位置计算并发送电机速度指令"""
        if laser_spot is None or green_pos is None or self.paused:
            self._stop_motor_if_needed()
            return 0.0, 0.0

        lx, ly, _ = laser_spot
        gx, gy = green_pos
        ex = gx - lx
        ey = gy - ly

        pan_speed, tilt_speed = pixel_to_velocity(ex, ey)

        # 仅在速度变化超过阈值时发送
        if (abs(pan_speed - self.prev_pan_speed) > 0.5 or
                abs(tilt_speed - self.prev_tilt_speed) > 0.5):
            if self.mc is not None:
                try:
                    self.mc.set_velocity(pan_speed, tilt_speed)
                except Exception as e:
                    print(f"电机通信错误: {e}")
            self.prev_pan_speed = pan_speed
            self.prev_tilt_speed = tilt_speed

        # 到达死区时发送零速
        if abs(pan_speed) < 0.01 and abs(tilt_speed) < 0.01:
            if abs(self.prev_pan_speed) > 0.01 or abs(self.prev_tilt_speed) > 0.01:
                if self.mc is not None:
                    try:
                        self.mc.set_velocity(0.0, 0.0)
                    except Exception:
                        pass
                self.prev_pan_speed = 0.0
                self.prev_tilt_speed = 0.0

        return pan_speed, tilt_speed

    def _stop_motor_if_needed(self):
        """目标丢失时停止电机"""
        if abs(self.prev_pan_speed) > 0.01 or abs(self.prev_tilt_speed) > 0.01:
            if self.mc is not None and not self.paused:
                try:
                    self.mc.set_velocity(0.0, 0.0)
                except Exception:
                    pass
            self.prev_pan_speed = 0.0
            self.prev_tilt_speed = 0.0

    def reset_motor(self):
        """复位电机"""
        print("复位电机...")
        if self.mc is not None:
            try:
                self.mc.reset()
            except Exception as e:
                print(f"复位失败: {e}")
        self.prev_pan_speed = 0.0
        self.prev_tilt_speed = 0.0

    def toggle_pause(self):
        """暂停/继续"""
        self.paused = not self.paused
        if self.paused:
            print("已暂停")
            self._stop_motor_if_needed()
        else:
            print("继续追踪")

    # ==========================================
    # 读取传感器数据
    # ==========================================
    def read_sensors(self):
        """读取RGB帧和深度图"""
        frame = self.tr.read_rgb_frame()
        if frame is None:
            return None, None

        self.tr.depth_map = self.tr.get_depth_map()
        return frame, self.tr.depth_map

    # ==========================================
    # 键盘处理
    # ==========================================
    def handle_key(self, key):
        """处理键盘输入，返回 True 表示退出，False 表示重新初始化"""
        if key == ord('q'):
            return "quit"
        elif key == ord('r'):
            self.reset_motor()
        elif key == ord(' '):
            self.toggle_pause()
        return None

    # ==========================================
    # 主流程
    # ==========================================
    def run(self):
        """运行完整的闭环控制流程"""
        self.setup()

        while True:
            # ==========================================
            # Phase 1: TargetRecognizer 严苛初始化捕获三靶标
            # ==========================================
            print("\n>>> Phase 1: 等待三靶标 (Red, Green, Blue) 初始化捕获...")
            ref_frame, confirmed_targets = self.tr.capture_initial_targets()
            if ref_frame is None:
                break

            # ==========================================
            # Phase 2: TargetRecognizer 构建 ORB 星座模板
            # ==========================================
            print("\n>>> Phase 2: 构建 ORB 星座模板...")
            n_kp = self.tr.build_template(ref_frame, confirmed_targets)
            print(f"  模板就绪, {n_kp} 个特征点")

            # ==========================================
            # Phase 3: 追踪 + 激光闭环控制
            # ==========================================
            print("\n>>> Phase 3: 实时追踪 + 激光伺服")
            print("  追踪三靶标刚体, 提取绿色靶标 → 驱动激光点追踪\n")

            cv2.namedWindow("Tracking Binary (Masked)")

            while True:
                # --- 读取帧 ---
                frame, dpt = self.read_sensors()
                if frame is None:
                    break

                # --- TargetRecognizer: ORB追踪三靶标 ---
                _, binary = self.tr.preprocess(frame)
                tracking_success, all_targets, tr_result, rot_angle, aff_angle = \
                    self.tr.track_frame(frame, binary)

                # 追踪失败 → 回到 Phase 1 重新初始化
                if not tracking_success:
                    if self.tr.consecutive_tri_fail >= 7:  # MAX_TRI_FAIL
                        print("\n追踪丢失! 返回 Phase 1 重新初始化...\n")
                        break
                elif all_targets is None:
                    print("\n靶标丢失! 返回 Phase 1 重新初始化...\n")
                    break

                # --- LaserRecognizer: 检测激光点 ---
                thresh_val = cv2.getTrackbarPos("Threshold", "Laser Track Green")
                blur_val = cv2.getTrackbarPos("Blur", "Laser Track Green")
                depth_max_val = cv2.getTrackbarPos("DepthMax_mm", "Laser Track Green")
                if blur_val % 2 == 0:
                    blur_val += 1

                lr_result, combined_mask, laser_spot = self.lr.process_frame(
                    frame, dpt, depth_max_val, thresh_val, blur_val
                )

                # --- 提取绿色靶标坐标 → 控制电机 ---
                green_pos = all_targets.get("Green") if all_targets else None
                pan_speed, tilt_speed = self.control_motor(laser_spot, green_pos)

                # --- 可视化 ---
                display = self.draw_overlay(frame, all_targets, laser_spot,
                                            pan_speed, tilt_speed)

                # ORB追踪状态文字
                if tracking_success:
                    status = f"TRACKING | Rot:{rot_angle:.1f}deg"
                    cv2.putText(display, status, (10, display.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if self.paused:
                    cv2.putText(display, "PAUSED",
                                (display.shape[1] // 2 - 50, display.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                # 显示
                cv2.imshow("Laser Track Green", display)

                # 追踪用二值图 (遮罩圈内)
                binary_masked = np.zeros_like(binary)
                cv2.circle(binary_masked, self.tr.ref_mask_center,
                           self.tr.ref_mask_radius, 255, -1)
                binary_masked = cv2.bitwise_and(binary, binary_masked)
                cv2.imshow("Tracking Binary (Masked)", binary_masked)

                depth_vis = self.lr.visualize_depth(dpt, depth_max_val)
                cv2.imshow("Depth", depth_vis)

                # --- 键盘 ---
                key = cv2.waitKey(1) & 0xFF
                action = self.handle_key(key)
                if action == "quit":
                    self.shutdown()
                    return

            # 内层循环退出 = 追踪失败, 回到外层 while → Phase 1 重新初始化
            cv2.destroyWindow("Tracking Binary (Masked)")

    def shutdown(self):
        """清理资源"""
        print("\n正在退出...")
        if self.mc is not None:
            try:
                self.mc.set_velocity(0.0, 0.0)
                time.sleep(0.2)
                self.mc.disconnect()
            except Exception:
                pass
        self.tr.release()
        print("已退出。")


# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    controller = LaserTrackController()
    controller.run()