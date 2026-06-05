"""
绿色靶标闭环跟踪程序
通过视觉伺服 (Visual Servoing) 控制二自由度云台，使绿色靶标保持在画面中心。
"""

import cv2
import numpy as np
import serial
import time
from openni import openni2
from collections import deque

# ==========================================
# 串口 / 电机参数
# ==========================================
SERIAL_PORT = 'COM7'
BAUDRATE = 9600

# ==========================================
# 图像参数
# ==========================================
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CENTER_X = IMAGE_WIDTH // 2   # 320
CENTER_Y = IMAGE_HEIGHT // 2  # 240

# ==========================================
# 绿色 HSV 阈值
# ==========================================
GREEN_LOWER = np.array([40, 50, 50])
GREEN_UPPER = np.array([90, 255, 255])

# ==========================================
# PID 控制器参数 (可通过滑动条实时调节)
# ==========================================
KP_X_INIT = 0.18
KI_X_INIT = 0.004
KD_X_INIT = 0.06
KP_Y_INIT = 0.18
KI_Y_INIT = 0.004
KD_Y_INIT = 0.06

MAX_SPEED = 90       # 最大电机速度 (度/秒)
DEAD_ZONE = 6        # 死区 (像素), 误差小于此值不输出
MIN_AREA = 200       # 绿色区域最小面积 (过滤噪点)

# ==========================================
# PID 控制器实现
# ==========================================
class PIDController:
    """带积分抗饱和和死区的 PID 控制器"""

    def __init__(self, kp, ki, kd, max_output, dead_zone=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.dead_zone = dead_zone
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update(self, error, current_time):
        if self.prev_time is None:
            self.prev_time = current_time
            self.prev_error = error
            return 0.0

        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 0.033  # 假设 30fps

        # 死区
        if abs(error) < self.dead_zone:
            self.integral *= 0.9  # 死区内缓慢衰减积分
            self.prev_error = error
            self.prev_time = current_time
            return 0.0

        # 比例项
        p = self.kp * error

        # 积分项 (带抗饱和)
        max_integral = self.max_output / (self.ki + 1e-8)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        i = self.ki * self.integral

        # 微分项
        d = self.kd * (error - self.prev_error) / dt

        output = p + i + d
        output = np.clip(output, -self.max_output, self.max_output)

        self.prev_error = error
        self.prev_time = current_time

        return output


# ==========================================
# 串口通信
# ==========================================
def setup_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        time.sleep(2)
        print("串口已连接")
        return ser
    except serial.SerialException as e:
        print(f"串口连接失败: {e}")
        return None


def send_speed_command(ser, pan_speed, tilt_speed):
    """发送模式2 (闭环速度控制) 命令"""
    if ser is None:
        return
    cmd = f"2,{pan_speed:.1f},{tilt_speed:.1f}\n"
    ser.write(cmd.encode())
    # 读取响应 (非阻塞)
    while ser.in_waiting:
        line = ser.readline().decode().strip()
        if line:
            print("Arduino:", line)


def send_stop(ser):
    """停止电机"""
    if ser is None:
        return
    ser.write(b"2,0,0\n")
    time.sleep(0.05)


def send_reset(ser):
    """复位云台"""
    if ser is None:
        return
    ser.write(b"0\n")
    time.sleep(0.05)


# ==========================================
# 绿色靶标检测
# ==========================================
def detect_green_target(hsv, min_area=MIN_AREA):
    """在 HSV 图像中检测最大的绿色区域, 返回其质心和包围盒"""
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    # 形态学去噪
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None, mask

    M = cv2.moments(largest)
    if M["m00"] <= 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(largest)

    return (cx, cy, x, y, w, h, area), mask


# ==========================================
# 主程序
# ==========================================
def main():
    print("=" * 55)
    print("  绿色靶标闭环跟踪系统 (Visual Servoing)")
    print("=" * 55)

    # --- 初始化相机 ---
    print("正在初始化相机...")
    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开RGB摄像头")
        depth_stream.stop()
        dev.close()
        exit()

    print("正在跳过前5帧以稳定传感器...")
    for _ in range(5):
        cap.read()
        depth_stream.read_frame()
        cv2.waitKey(100)

    # --- 初始化串口 ---
    ser = setup_serial()

    # --- PID 控制器 ---
    pid_x = PIDController(KP_X_INIT, KI_X_INIT, KD_X_INIT, MAX_SPEED, DEAD_ZONE)
    pid_y = PIDController(KP_Y_INIT, KI_Y_INIT, KD_Y_INIT, MAX_SPEED, DEAD_ZONE)

    # --- 滑动条 ---
    cv2.namedWindow("Visual Servoing")
    cv2.createTrackbar("Kp_X", "Visual Servoing", int(KP_X_INIT * 1000), 2000, lambda _: None)
    cv2.createTrackbar("Ki_X", "Visual Servoing", int(KI_X_INIT * 1000), 200, lambda _: None)
    cv2.createTrackbar("Kd_X", "Visual Servoing", int(KD_X_INIT * 1000), 2000, lambda _: None)
    cv2.createTrackbar("Kp_Y", "Visual Servoing", int(KP_Y_INIT * 1000), 2000, lambda _: None)
    cv2.createTrackbar("Ki_Y", "Visual Servoing", int(KI_Y_INIT * 1000), 200, lambda _: None)
    cv2.createTrackbar("Kd_Y", "Visual Servoing", int(KD_Y_INIT * 1000), 2000, lambda _: None)
    cv2.createTrackbar("DeadZone", "Visual Servoing", DEAD_ZONE, 50, lambda _: None)
    cv2.createTrackbar("MaxSpeed", "Visual Servoing", MAX_SPEED, 180, lambda _: None)

    # --- 状态变量 ---
    lost_count = 0
    MAX_LOST = 30          # 连续丢失帧数超过此值则停止电机
    last_send_time = 0     # 上次发送指令的时间
    SEND_INTERVAL = 0.08   # 最小发送间隔 (秒)
    target_found = False
    current_pan_speed = 0.0
    current_tilt_speed = 0.0

    print("\n控制器就绪！按 'q' 退出, 'r' 复位云台, 's' 急停\n")

    while True:
        # --- 读取帧 ---
        ret, frame = cap.read()
        if not ret:
            print("无法读取RGB帧")
            break

        depth_frame = depth_stream.read_frame()
        dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        depth_map = (np.asarray(dframe_data[:, :, 0], dtype="float32") +
                     np.asarray(dframe_data[:, :, 1], dtype="float32") * 255)[:, ::-1]

        now = time.time()

        # --- 读取滑动条参数 ---
        pid_x.kp = cv2.getTrackbarPos("Kp_X", "Visual Servoing") / 1000.0
        pid_x.ki = cv2.getTrackbarPos("Ki_X", "Visual Servoing") / 1000.0
        pid_x.kd = cv2.getTrackbarPos("Kd_X", "Visual Servoing") / 1000.0
        pid_y.kp = cv2.getTrackbarPos("Kp_Y", "Visual Servoing") / 1000.0
        pid_y.ki = cv2.getTrackbarPos("Ki_Y", "Visual Servoing") / 1000.0
        pid_y.kd = cv2.getTrackbarPos("Kd_Y", "Visual Servoing") / 1000.0
        pid_x.dead_zone = cv2.getTrackbarPos("DeadZone", "Visual Servoing")
        pid_y.dead_zone = cv2.getTrackbarPos("DeadZone", "Visual Servoing")
        pid_x.max_output = cv2.getTrackbarPos("MaxSpeed", "Visual Servoing")
        pid_y.max_output = cv2.getTrackbarPos("MaxSpeed", "Visual Servoing")

        # --- 绿色靶标检测 ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result, green_mask = detect_green_target(hsv)

        result_img = frame.copy()

        if result is not None:
            cx, cy, x, y, w, h, area = result

            target_found = True
            lost_count = 0

            # 像素误差: 靶标相对于画面中心的偏移
            error_x = cx - CENTER_X   # +x = 偏右, 需向右转
            error_y = CENTER_Y - cy   # +y = 偏上, 需向上转 (图像y轴向下)

            # PID 计算速度
            pan_speed = pid_x.update(error_x, now)
            tilt_speed = pid_y.update(error_y, now)

            current_pan_speed = pan_speed
            current_tilt_speed = tilt_speed

            # 深度值
            d_val = depth_map[cy, cx] if (0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]) else 0

            # --- 可视化 ---
            # 靶标包围盒和质心
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(result_img, (cx, cy), 8, (0, 255, 0), -1)
            cv2.circle(result_img, (cx, cy), 12, (255, 255, 255), 2)

            # 画面中心十字
            cv2.drawMarker(result_img, (CENTER_X, CENTER_Y), (0, 0, 255),
                           cv2.MARKER_CROSS, 30, 2)

            # 质心到中心的连线
            cv2.line(result_img, (cx, cy), (CENTER_X, CENTER_Y), (255, 0, 0), 2)

            # 信息
            cv2.putText(result_img, f"Target: ({cx}, {cy}) Area: {area:.0f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(result_img, f"Error: ({error_x:+d}, {error_y:+d}) px",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(result_img, f"Depth: {d_val:.0f} mm",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(result_img, f"Speed: Pan={pan_speed:+.1f} Tilt={tilt_speed:+.1f} deg/s",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # PID 内部状态
            cv2.putText(result_img,
                        f"P=({pid_x.kp*error_x:+.1f},{pid_y.kp*error_y:+.1f}) "
                        f"I=({pid_x.integral*pid_x.ki:+.1f},{pid_y.integral*pid_y.ki:+.1f})",
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        else:
            target_found = False
            lost_count += 1

            if lost_count >= MAX_LOST:
                # 超时丢失, 停止电机
                if current_pan_speed != 0.0 or current_tilt_speed != 0.0:
                    current_pan_speed = 0.0
                    current_tilt_speed = 0.0
                    send_speed_command(ser, 0.0, 0.0)
                    pid_x.reset()
                    pid_y.reset()

            cv2.drawMarker(result_img, (CENTER_X, CENTER_Y), (0, 0, 255),
                           cv2.MARKER_CROSS, 30, 2)
            status = "TARGET LOST" if lost_count < MAX_LOST else "STOPPED (target lost)"
            cv2.putText(result_img, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(result_img, f"Lost frames: {lost_count}/{MAX_LOST}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # --- 发送电机指令 ---
        if target_found and (now - last_send_time) >= SEND_INTERVAL:
            send_speed_command(ser, current_pan_speed, current_tilt_speed)
            last_send_time = now

        # --- 显示 ---
        cv2.imshow("Visual Servoing", result_img)
        cv2.imshow("Green Mask", green_mask)

        # --- 键盘 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("复位云台...")
            send_reset(ser)
            pid_x.reset()
            pid_y.reset()
            current_pan_speed = 0.0
            current_tilt_speed = 0.0
            lost_count = 0
        elif key == ord('s'):
            print("急停！")
            send_stop(ser)
            pid_x.reset()
            pid_y.reset()
            current_pan_speed = 0.0
            current_tilt_speed = 0.0

    # --- 清理 ---
    print("\n正在退出...")
    send_stop(ser)
    time.sleep(0.1)
    send_reset(ser)
    time.sleep(0.2)

    if ser:
        ser.close()
    cap.release()
    depth_stream.stop()
    dev.close()
    cv2.destroyAllWindows()
    print("已退出。")


if __name__ == "__main__":
    main()
