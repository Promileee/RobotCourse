"""
激光追踪靶标 - 闭环控制系统 V2 (类封装版)
==========================================
将 V1 重构为 CloseLoopController 类，支持灵活的颜色指定方式，
便于 import 调用。

颜色指定方式:
  - 名称:     'r' / 'g' / 'b' 或 'Red' / 'Green' / 'Blue'
  - 编码:     1=Red, 2=Green, 3=Blue
  - BGR元组:  (0,0,255) / (0,255,0) / (255,0,0)

使用示例:
  >>> from close_loop_control_v2 import CloseLoopController
  >>> ctrl = CloseLoopController()
  >>> ctrl.run('r')         # 追踪红色靶标 (阻塞)
  >>> ctrl.run(1)           # 等效
  >>> ctrl.run((0,0,255))   # 等效 (BGR)

  >>> # 非阻塞单步模式 (自行控制循环):
  >>> ctrl = CloseLoopController()
  >>> ctrl.init_tracking('g')
  >>> while True:
  ...     frame = ...  # 自行获取帧
  ...     result = ctrl.step(frame)
  ...     if result is None: break
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

# 颜色映射: 各种输入格式 → 内部颜色名
_COLOR_NAME_MAP = {
    # 简写
    'r': 'Red', 'g': 'Green', 'b': 'Blue',
    # 全称小写
    'red': 'Red', 'green': 'Green', 'blue': 'Blue',
    # 全称首字母大写
    'Red': 'Red', 'Green': 'Green', 'Blue': 'Blue',
}

# BGR值 → 颜色名
_BGR_TO_COLOR = {
    (0, 0, 255): 'Red',
    (0, 255, 0): 'Green',
    (255, 0, 0): 'Blue',
}

# 编码 → 颜色名
_ENCODING_TO_COLOR = {
    1: 'Red',
    2: 'Green',
    3: 'Blue',
}

# 键盘映射 (用于交互模式的实时切换)
_NUM_KEY_MAP = {ord('1'): 'Red', ord('2'): 'Green', ord('3'): 'Blue'}


def parse_color(spec):
    """将各种颜色输入格式解析为内部颜色名 ('Red' / 'Green' / 'Blue')。

    支持:
      - 字符串: 'r', 'g', 'b', 'red', 'green', 'blue', 'Red', 'Green', 'Blue'
      - 整数编码: 1 → Red, 2 → Green, 3 → Blue
      - BGR元组: (0,0,255) → Red, (0,255,0) → Green, (255,0,0) → Blue
      - RGB元组: (255,0,0) → Blue, (0,255,0) → Green, (0,0,255) → Red
                (注意: OpenCV使用BGR顺序，但RGB也会被自动兼容)

    返回: 'Red' / 'Green' / 'Blue'，解析失败返回 None。
    """
    if spec is None:
        return None

    # 整数编码
    if isinstance(spec, int):
        return _ENCODING_TO_COLOR.get(spec)

    # 字符串
    if isinstance(spec, str):
        return _COLOR_NAME_MAP.get(spec)

    # 元组/列表 (BGR 或 RGB)
    if isinstance(spec, (tuple, list)) and len(spec) == 3:
        t = tuple(int(c) for c in spec)
        # 先尝试按 BGR 匹配
        color = _BGR_TO_COLOR.get(t)
        if color:
            return color
        # 再尝试按 RGB 匹配 (交换第一和第三通道)
        t_bgr = (t[2], t[1], t[0])
        return _BGR_TO_COLOR.get(t_bgr)

    return None


def pixel_to_velocity(ex, ey, dex, dey):
    """像素误差 → 目标电机速度 (PD控制 + 死区 + 限幅)"""
    if abs(ex) < DEAD_ZONE and abs(ey) < DEAD_ZONE:
        return 0.0, 0.0
    pan = np.clip(ex * KP_PAN + dex * KD_PAN, -MAX_SPEED, MAX_SPEED)
    tilt = np.clip((ey * KP_TILT + dey * KD_TILT) * TILT_INVERT, -MAX_SPEED, MAX_SPEED)
    return pan, tilt


class CloseLoopController:
    """激光追踪闭环控制器。

    封装 CameraManager、TargetRecognizer、LaserRecognizerV3、MotorController
    四个模块，通过 PD 控制器驱动激光追踪指定颜色靶标。

    常用方法:
      - run(color):    阻塞式运行完整追踪流程 (含GUI窗口)
      - init_tracking(color): 初始化追踪管线，返回成功与否
      - step(frame):   单步处理一帧 (需先调用 init_tracking)
      - stop():        停止并释放资源
      - parse_color(): 静态方法，解析颜色输入
    """

    def __init__(self,
                 kp_pan=KP_PAN, kp_tilt=KP_TILT,
                 kd_pan=KD_PAN, kd_tilt=KD_TILT,
                 max_speed=MAX_SPEED, max_accel=MAX_ACCEL,
                 dead_zone=DEAD_ZONE, tilt_invert=TILT_INVERT,
                 serial_port=SERIAL_PORT, serial_baud=SERIAL_BAUD,
                 laser_thresh=LASER_THRESH, laser_blur=LASER_BLUR,
                 depth_max=DEPTH_MAX):
        """
        参数:
          kp_pan, kp_tilt:    PD比例系数
          kd_pan, kd_tilt:    PD微分系数
          max_speed:          电机最大速度
          max_accel:          电机最大加速度
          dead_zone:          死区像素
          tilt_invert:        tilt方向反转 (-1或1)
          serial_port:        电机串口号
          serial_baud:        电机波特率
          laser_thresh:       激光检测阈值
          laser_blur:         激光检测模糊核大小
          depth_max:          深度最大值 (mm)
        """
        # PD参数
        self.kp_pan = kp_pan
        self.kp_tilt = kp_tilt
        self.kd_pan = kd_pan
        self.kd_tilt = kd_tilt
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.dead_zone = dead_zone
        self.tilt_invert = tilt_invert

        # 硬件参数
        self.serial_port = serial_port
        self.serial_baud = serial_baud
        self.laser_thresh = laser_thresh
        self.laser_blur = laser_blur
        self.depth_max = depth_max

        # 模块实例 (延迟初始化)
        self.cam = None
        self.tr = None
        self.lr = None
        self.mc = None
        self.motor_ok = False

        # 追踪状态
        self.target_color = None
        self.paused = False
        self.prev_pan = 0.0
        self.prev_tilt = 0.0
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.last_laser_time = 0.0

        # 追踪管线就绪标志 (用于 step 模式)
        self._phase3_ready = False

    # ==========================================
    # 颜色解析 (静态方法 + 实例方法)
    # ==========================================

    @staticmethod
    def parse_color(spec):
        """解析颜色输入，返回内部颜色名。详见模块级 parse_color 函数。"""
        return parse_color(spec)

    # ==========================================
    # 生命周期
    # ==========================================

    def setup(self):
        """初始化相机、识别器、电机控制器。

        返回: (motor_ok: bool) 电机是否连接成功。
        """
        # 1. CameraManager
        self.cam = CameraManager()
        self.cam.start()

        # 2. TargetRecognizer
        self.tr = TargetRecognizer()
        self.tr.setup(self.cam)

        # 3. LaserRecognizerV3
        self.lr = LaserRecognizerV3(depth_max=self.depth_max,
                                    thresh_val=self.laser_thresh,
                                    blur_size=self.laser_blur)
        self.lr.setup(self.cam)

        # 4. MotorController
        self.mc = MotorController(port=self.serial_port, baudrate=self.serial_baud)
        self.motor_ok = True
        try:
            self.mc.connect()
            self.mc.reset()
            time.sleep(1)
        except Exception as e:
            print(f"  警告: 无法连接电机 ({e}) → 仅检测模式")
            self.motor_ok = False

        return self.motor_ok

    def stop(self):
        """停止追踪，释放所有资源。"""
        self._phase3_ready = False

        if self.motor_ok and self.mc:
            try:
                self.mc.set_velocity(0.0, 0.0)
                time.sleep(0.2)
                self.mc.disconnect()
            except Exception:
                pass
            self.motor_ok = False

        if self.cam:
            try:
                self.cam.release()
            except Exception:
                pass

        self.tr = None
        self.lr = None
        self.mc = None
        self.cam = None

    def reset_motor(self):
        """复位电机并清零内部速度状态。"""
        self.prev_pan = 0.0
        self.prev_tilt = 0.0
        if self.motor_ok and self.mc:
            try:
                self.mc.reset()
            except Exception as e:
                print(f"复位失败: {e}")

    # ==========================================
    # 追踪管线 (三步初始化 → 持续追踪)
    # ==========================================

    def init_tracking(self, color_spec):
        """执行 Phase 1 → Phase 2 → Phase 2.5 初始化流程，准备进入 Phase 3 追踪。

        参数:
          color_spec: 目标颜色，支持字符串/整数/元组，参见 parse_color()。

        返回: (success: bool, target_color_name: str | None)
        """
        # 解析颜色
        color = parse_color(color_spec)
        if color is None:
            print(f"  无法解析颜色: {color_spec}")
            return False, None

        self.target_color = color
        self._phase3_ready = False

        if self.cam is None:
            print("  错误: 请先调用 setup()")
            return False, None

        # Phase 1: 捕获三靶标
        print(">>> Phase 1: 等待三靶标 (Red, Green, Blue) 初始化捕获...")
        ref_frame, confirmed_targets = self.tr.capture_initial_targets()
        if ref_frame is None:
            print("  Phase 1 失败")
            return False, None

        # Phase 2: 构建 ORB 星座模板
        print(">>> Phase 2: 构建 ORB 星座模板...")
        n_kp = self.tr.build_template(ref_frame, confirmed_targets)
        print(f"  模板就绪, {n_kp} 个特征点")

        # Phase 2.5: 初始化 LaserRecognizerV3
        dpt_init = self.cam.get_depth_map()
        self.lr.init_from_target_data(self.tr.ref_targets_centers, dpt_init)

        # 重置追踪状态
        self.prev_pan = 0.0
        self.prev_tilt = 0.0
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.last_laser_time = time.time()

        self._phase3_ready = True
        print(f">>> Phase 3 就绪, 目标: {color}\n")
        return True, color

    def step(self, frame, depth_map=None):
        """单步追踪处理。需先成功调用 init_tracking()。

        参数:
          frame:     RGB图像 (numpy array, BGR)
          depth_map: 深度图 (可选，为None时自动获取)

        返回: dict 或 None (追踪丢失需重新初始化时返回None)。
          dict 包含:
            'display':       可视化结果图
            'binary_masked': 掩码二值图
            'target_pos':    (gx, gy) 目标像素坐标
            'laser_spot':    (lx, ly, depth) 激光点
            'error':         (ex, ey) 像素误差
            'pan_speed':     云台速度
            'tilt_speed':    倾斜速度
            'tracking_ok':   追踪是否正常
        """
        if not self._phase3_ready:
            print("  错误: 请先调用 init_tracking()")
            return None

        if depth_map is None:
            depth_map = self.cam.get_depth_map()
        self.tr.depth_map = depth_map

        # --- TargetRecognizer: ORB追踪三靶标 ---
        gray, binary = self.tr.preprocess(frame)
        tracking_ok, all_targets, tr_result, rot_angle, _ = \
            self.tr.track_frame(frame, binary)

        if not tracking_ok and self.tr.consecutive_tri_fail >= 7:
            print("\n追踪丢失! 需重新初始化。\n")
            self.lr.reset()
            self._phase3_ready = False
            return None

        # --- LaserRecognizerV3: 检测激光点 ---
        _, _, laser_spot, _ = self.lr.process_frame(frame, depth_map)

        # --- PD控制 ---
        target_pos = all_targets.get(self.target_color) if all_targets else None

        if laser_spot is not None and target_pos is not None and not self.paused:
            lx, ly = laser_spot[0], laser_spot[1]
            gx, gy = target_pos
            ex, ey = gx - lx, gy - ly
            dex, dey = ex - self.prev_ex, ey - self.prev_ey
            target_pan, target_tilt = self._pd_control(ex, ey, dex, dey)
            self.prev_ex, self.prev_ey = ex, ey
            self.last_laser_time = time.time()
        else:
            target_pan, target_tilt = 0.0, 0.0
            self.prev_ex, self.prev_ey = 0.0, 0.0

        # --- 激光超时保护 ---
        if time.time() - self.last_laser_time > 1.0 and self.motor_ok and not self.paused:
            print("激光丢失超过1s，复位电机...")
            self.reset_motor()
            self.last_laser_time = time.time()

        # --- 加速度限制 ---
        pan_speed, tilt_speed = self._apply_accel_limit(target_pan, target_tilt)

        # --- 发送电机指令 ---
        self._send_motor_command(pan_speed, tilt_speed)

        # --- 可视化 ---
        display, binary_masked = self._visualize(
            tr_result, binary, laser_spot, target_pos, pan_speed, tilt_speed)

        return {
            'display': display,
            'binary_masked': binary_masked,
            'target_pos': target_pos,
            'laser_spot': laser_spot,
            'error': (self.prev_ex, self.prev_ey),
            'pan_speed': pan_speed,
            'tilt_speed': tilt_speed,
            'tracking_ok': tracking_ok,
        }

    def switch_target(self, color_spec):
        """运行时切换追踪目标 (不清除追踪状态，仅切换颜色)。

        返回: 新的颜色名，失败返回 None。
        """
        color = parse_color(color_spec)
        if color is None:
            print(f"  无法解析颜色: {color_spec}")
            return None
        if color == self.target_color:
            return color
        self.target_color = color
        self.prev_ex, self.prev_ey = 0.0, 0.0
        self.prev_pan, self.prev_tilt = 0.0, 0.0
        if self.motor_ok and self.mc:
            try:
                self.mc.set_velocity(0.0, 0.0)
            except Exception:
                pass
        print(f"  切换靶标 → {color}")
        return color

    # ==========================================
    # 阻塞式完整运行 (兼容 V1 的交互模式)
    # ==========================================

    def run(self, color_spec=None):
        """阻塞式运行完整追踪流程 (含 GUI 窗口和键盘交互)。

        参数:
          color_spec: 目标颜色。为 None 时进入交互式颜色选择。

        键盘:
          1/2/3  切换靶标
          q      输入新颜色 (回车/quit退出)
          r      复位电机
          空格   暂停/继续
          Esc    退出
        """
        # --- 确保已 setup ---
        if self.cam is None:
            self.setup()

        # --- 窗口 & 滑动条 ---
        WIN_NAME = "Laser Track V2"
        cv2.namedWindow(WIN_NAME)
        cv2.createTrackbar("Threshold", WIN_NAME, self.laser_thresh, 255, lambda _: None)
        cv2.createTrackbar("Blur", WIN_NAME, self.laser_blur, 20, lambda _: None)
        cv2.createTrackbar("DepthMax_mm", WIN_NAME, self.depth_max, 5000, lambda _: None)

        # --- 颜色选择 ---
        if color_spec is not None:
            target = parse_color(color_spec)
            if target is None:
                print(f"  无法解析颜色: {color_spec}")
                self.stop()
                return
            self.target_color = target
            print(f"  目标靶标: {target}\n")
        else:
            self.target_color = None
            while self.target_color is None:
                self.target_color = self._interactive_color_prompt()
                if self.target_color is None:
                    print("正在退出...")
                    self.stop()
                    return
            print(f"  已选择: {self.target_color}\n")

        print("=" * 60)
        print("  按 1/2/3 实时切换靶标 | 按 q 输入新颜色 | 按 r 复位 | 按 空格 暂停")
        print("=" * 60 + "\n")

        self.paused = False
        self.prev_pan = 0.0
        self.prev_tilt = 0.0
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.last_laser_time = time.time()

        # --- 主循环 ---
        while True:
            # Phase 1
            print(">>> Phase 1: 等待三靶标 (Red, Green, Blue) 初始化捕获...")
            ref_frame, confirmed_targets = self.tr.capture_initial_targets()
            if ref_frame is None:
                break

            # Phase 2
            print(">>> Phase 2: 构建 ORB 星座模板...")
            n_kp = self.tr.build_template(ref_frame, confirmed_targets)
            print(f"  模板就绪, {n_kp} 个特征点")

            # Phase 2.5
            dpt_init = self.cam.get_depth_map()
            self.lr.init_from_target_data(self.tr.ref_targets_centers, dpt_init)

            # Phase 3
            print(f">>> Phase 3: 实时追踪三靶标 → 激光追 {self.target_color}\n")
            self._phase3_ready = True

            cv2.namedWindow("Tracking Binary (Masked)")

            while True:
                frame = self.cam.read_rgb_frame()
                if frame is None:
                    break
                dpt = self.cam.get_depth_map()
                self.tr.depth_map = dpt

                # 滑动条参数
                thresh_val = cv2.getTrackbarPos("Threshold", WIN_NAME)
                blur_val = cv2.getTrackbarPos("Blur", WIN_NAME)
                depth_max_val = cv2.getTrackbarPos("DepthMax_mm", WIN_NAME)
                if blur_val % 2 == 0:
                    blur_val += 1

                self.lr.thresh_val = thresh_val
                self.lr.blur_size = blur_val
                self.lr.depth_max = depth_max_val

                # 单步追踪
                result = self.step(frame, dpt)
                if result is None:
                    # 追踪丢失，返回 Phase 1
                    self._phase3_ready = False
                    break

                # 显示
                cv2.imshow(WIN_NAME, result['display'])
                cv2.imshow("Depth", self.lr.visualize_depth(dpt, depth_max_val))
                cv2.imshow("Tracking Binary (Masked)", result['binary_masked'])

                # 键盘
                key = cv2.waitKey(1) & 0xFF
                if self._handle_key(key):
                    cv2.destroyWindow("Tracking Binary (Masked)")
                    self.stop()
                    return

                # 如果 Phase 3 被键盘中断 (需重新初始化)
                if not self._phase3_ready:
                    cv2.destroyWindow("Tracking Binary (Masked)")
                    break

            cv2.destroyWindow("Tracking Binary (Masked)")

        # 清理
        self.stop()

    # ==========================================
    # 内部方法
    # ==========================================

    def _pd_control(self, ex, ey, dex, dey):
        """PD控制计算目标速度。"""
        if abs(ex) < self.dead_zone and abs(ey) < self.dead_zone:
            return 0.0, 0.0
        pan = np.clip(ex * self.kp_pan + dex * self.kd_pan, -self.max_speed, self.max_speed)
        tilt = np.clip((ey * self.kp_tilt + dey * self.kd_tilt) * self.tilt_invert,
                       -self.max_speed, self.max_speed)
        return pan, tilt

    def _apply_accel_limit(self, target_pan, target_tilt):
        """应用加速度限制。"""
        max_step = self.max_accel * 0.033

        pan_diff = target_pan - self.prev_pan
        pan_diff = np.clip(pan_diff, -max_step, max_step)
        pan_speed = self.prev_pan + pan_diff

        tilt_diff = target_tilt - self.prev_tilt
        tilt_diff = np.clip(tilt_diff, -max_step, max_step)
        tilt_speed = self.prev_tilt + tilt_diff

        self.prev_pan = pan_speed
        self.prev_tilt = tilt_speed
        return pan_speed, tilt_speed

    def _send_motor_command(self, pan_speed, tilt_speed):
        """发送电机速度指令 (含变化检测避免重复发送)。"""
        if not self.motor_ok or self.paused:
            return
        try:
            self.mc.set_velocity(pan_speed, tilt_speed)
        except Exception as e:
            print(f"电机通信错误: {e}")

    def _visualize(self, tr_result, binary, laser_spot, target_pos, pan_speed, tilt_speed):
        """绘制可视化叠加层。"""
        display = tr_result

        # 激光点
        if laser_spot is not None:
            lx, ly = laser_spot[0], laser_spot[1]
            ld = laser_spot[2]
            cv2.circle(display, (lx, ly), 15, (0, 0, 255), 2)
            cv2.circle(display, (lx, ly), 3, (0, 0, 255), -1)
            cv2.putText(display, f"Laser ({lx},{ly}) d={ld:.0f}mm",
                        (lx + 20, ly - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 误差线
        if laser_spot is not None and target_pos is not None:
            lx, ly = laser_spot[0], laser_spot[1]
            gx, gy = target_pos
            cv2.line(display, (lx, ly), (gx, gy), (255, 255, 0), 1, cv2.LINE_AA)
            mx, my = (lx + gx) // 2, (ly + gy) // 2
            cv2.putText(display, f"Err({gx-lx},{gy-ly})", (mx + 5, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # 状态栏
        status = f"Target:{self.target_color}  Pan:{pan_speed:+.1f}/s  Tilt:{tilt_speed:+.1f}/s"
        cv2.putText(display, status, (10, display.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if (abs(pan_speed) < 0.01 and abs(tilt_speed) < 0.01
                and laser_spot is not None and target_pos is not None):
            cv2.putText(display, "ON TARGET", (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.paused:
            cv2.putText(display, "PAUSED",
                        (display.shape[1] // 2 - 50, display.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 掩码二值图
        binary_masked = np.zeros_like(binary)
        cv2.circle(binary_masked, self.tr.ref_mask_center, self.tr.ref_mask_radius, 255, -1)
        binary_masked = cv2.bitwise_and(binary, binary_masked)

        return display, binary_masked

    def _handle_key(self, key):
        """处理键盘输入。返回 True 表示需要退出程序。"""
        if key in _NUM_KEY_MAP:
            new_color = _NUM_KEY_MAP[key]
            if new_color != self.target_color:
                self.switch_target(new_color)

        elif key == ord('q'):
            if self.motor_ok and self.mc:
                try:
                    self.mc.set_velocity(0.0, 0.0)
                except Exception:
                    pass
            self.prev_pan, self.prev_tilt = 0.0, 0.0
            self.paused = True
            new_color = self._interactive_color_prompt(self.target_color)
            if new_color is None:
                return True  # 退出
            self.target_color = new_color
            self.prev_ex, self.prev_ey = 0.0, 0.0
            self.last_laser_time = time.time()
            self.paused = False
            print(f"  继续追踪: {self.target_color}\n")

        elif key == ord('r'):
            print("复位电机...")
            self.reset_motor()
            self.prev_pan = 0.0
            self.prev_tilt = 0.0

        elif key == ord(' '):
            self.paused = not self.paused
            print("已暂停" if self.paused else "继续追踪")
            if self.motor_ok and self.mc:
                try:
                    self.mc.set_velocity(0.0, 0.0)
                except Exception:
                    pass
            self.prev_pan, self.prev_tilt = 0.0, 0.0

        elif key == 27:  # Esc
            return True

        return False

    def _interactive_color_prompt(self, current=None):
        """阻塞式输入靶标颜色。回车或 quit 返回 None。"""
        hint = f" (当前: {current})" if current else ""
        raw = input(f"请输入靶标颜色 (r=Red, g=Green, b=Blue, quit=退出){hint}: ").strip().lower()
        if raw == '' or raw == 'quit':
            return None
        if raw in _COLOR_NAME_MAP:
            return _COLOR_NAME_MAP[raw]
        # 尝试作为整数解析
        try:
            code = int(raw)
            if code in _ENCODING_TO_COLOR:
                return _ENCODING_TO_COLOR[code]
        except ValueError:
            pass
        print(f"  无效输入 '{raw}'，保持当前靶标")
        return None


# ==========================================
# 独立运行入口
# ==========================================
def main():
    print("=" * 60)
    print("  激光追踪靶标 - 闭环视觉伺服控制 V2")
    print("=" * 60)

    ctrl = CloseLoopController()
    ctrl.setup()
    ctrl.run()  # 无参数 → 交互式颜色选择


if __name__ == "__main__":
    main()