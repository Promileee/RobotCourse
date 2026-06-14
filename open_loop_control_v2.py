"""
开环激光指向控制器 V2 (Class-based)
=====================================
将 V1 重构为类，核心功能：输入摄像头像素坐标 → 开环指向该坐标。

Usage:
    from open_loop_control_v2 import OpenLoopController

    ctrl = OpenLoopController(serial_port='COM7')
    ctrl.setup()

    # 方式1: 现场标定
    ctrl.calibrate()

    # 方式2: 加载已有模型
    # ctrl.load_model("calib_model.npz")

    # 指向指定像素坐标
    ctrl.point_to(320, 240)

    # 持续追踪绿色靶标
    # ctrl.start_tracking()

    ctrl.shutdown()
"""

import cv2
import math
import numpy as np
import time
import os

from camera_manager import CameraManager
from laser_recognize import LaserRecognizer
from laser_recognize_v3 import LaserRecognizerV3
from motor_control import MotorController
from target_recognize_v1 import TargetRecognizer


# ==========================================
# 工具函数
# ==========================================
def uniform_grid_2d(n, x_range, y_range):
    """在2D范围内均匀网格采点"""
    w = x_range[1] - x_range[0]
    h = y_range[1] - y_range[0]

    n_x = max(2, int(np.ceil(np.sqrt(n * w / h))))
    n_y = max(2, int(np.ceil(n / n_x)))

    xs = np.linspace(x_range[0], x_range[1], n_x)
    ys = np.linspace(y_range[0], y_range[1], n_y)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])

    if len(pts) > n:
        idx = np.linspace(0, len(pts) - 1, n, dtype=int)
        pts = pts[idx]
    return pts


# ==========================================
# 简单MLP回归器
# ==========================================
class SimpleMLP:
    """2层全连接神经网络，学习 摄像头坐标→电机角度 的映射"""

    def __init__(self, hidden_layers=(32, 16), lr=0.01, max_iter=2000):
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.max_iter = max_iter
        self.weights = []
        self.biases = []
        self.in_mean = None
        self.in_std = None
        self.out_mean = None
        self.out_std = None

    def _init_params(self, sizes):
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            w = np.random.randn(sizes[i], sizes[i + 1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros((1, sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def _forward(self, X):
        activations = [X]
        pre_acts = []
        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_acts.append(z)
            if i < len(self.weights) - 1:
                a = np.maximum(0, z)
            else:
                a = z
            activations.append(a)
        return activations, pre_acts

    def _backward(self, activations, pre_acts, y):
        m = y.shape[0]
        delta = (activations[-1] - y) * (2.0 / m)

        grads_w = []
        grads_b = []
        for i in reversed(range(len(self.weights))):
            if i < len(self.weights) - 1:
                delta = delta * (pre_acts[i] > 0)
            dw = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            if i > 0:
                delta = delta @ self.weights[i].T
        return grads_w, grads_b

    def fit(self, X, y, verbose=True):
        self.in_mean = X.mean(axis=0)
        self.in_std = X.std(axis=0) + 1e-8
        self.out_mean = y.mean(axis=0)
        self.out_std = y.std(axis=0) + 1e-8

        Xn = (X - self.in_mean) / self.in_std
        yn = (y - self.out_mean) / self.out_std

        sizes = [X.shape[1]] + list(self.hidden_layers) + [y.shape[1]]
        self._init_params(sizes)

        for epoch in range(self.max_iter):
            act, pre = self._forward(Xn)
            gw, gb = self._backward(act, pre, yn)
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * gw[i]
                self.biases[i] -= self.lr * gb[i]

            if verbose and epoch % 500 == 0:
                loss = np.mean((act[-1] - yn) ** 2)
                print(f"    Epoch {epoch:4d}: loss={loss:.6f}")

        final = np.mean((self._forward(Xn)[0][-1] - yn) ** 2)
        if verbose:
            print(f"    训练完成: final_loss={final:.6f}")

    def predict(self, X):
        Xn = (X - self.in_mean) / self.in_std
        yn = self._forward(Xn)[0][-1]
        return yn * self.out_std + self.out_mean

    def save(self, filepath):
        """保存模型参数到 .npz 文件"""
        np.savez(filepath,
                 hidden_layers=np.array(self.hidden_layers),
                 lr=self.lr, max_iter=self.max_iter,
                 *[self.weights[i] for i in range(len(self.weights))],
                 *[self.biases[i] for i in range(len(self.biases))],
                 in_mean=self.in_mean, in_std=self.in_std,
                 out_mean=self.out_mean, out_std=self.out_std)
        print(f"模型已保存: {filepath}")

    @classmethod
    def load(cls, filepath):
        """从 .npz 文件加载模型"""
        data = np.load(filepath, allow_pickle=True)
        hidden_layers = tuple(data['hidden_layers'].tolist())
        obj = cls(hidden_layers=hidden_layers,
                  lr=float(data['lr']), max_iter=int(data['max_iter']))

        num_layers = len(hidden_layers) + 1
        obj.weights = [data[f'arr_{i}'] for i in range(num_layers - 1)]
        obj.biases = [data[f'arr_{i + num_layers - 1}']
                      for i in range(num_layers - 1)]
        obj.in_mean = data['in_mean']
        obj.in_std = data['in_std']
        obj.out_mean = data['out_mean']
        obj.out_std = data['out_std']
        print(f"模型已加载: {filepath}")
        return obj


# ==========================================
# 开环指向控制器
# ==========================================
class OpenLoopController:
    """开环激光指向控制器

    核心流程：
        1. setup()       —— 初始化硬件、靶标识别、激光ROI
        2. calibrate()   —— 网格采样 + 训练NN（或 load_model() 加载已有模型）
        3. point_to(x,y) —— 输入像素坐标，开环指向

    Parameters:
        serial_port:     电机串口号
        serial_baud:     串口波特率
        pan_range:       Pan轴采样范围 (min, max) 度
        tilt_range:      Tilt轴采样范围 (min, max) 度
        num_calib_points: 标定采样点数
        settle_time:     电机稳定等待时间(秒)
        laser_thresh:    激光检测阈值
        laser_blur:      激光检测模糊核大小
        depth_max:       深度最大有效值(mm)
        adjust_gain:     方向调整增益 (°/px)
        max_adjust_attempts: 单点最大调整次数
        nn_hidden:       NN隐藏层大小
        nn_lr:           NN学习率
        nn_epochs:       NN训练轮数
        track_interval:  追踪指向间隔(秒)
    """

    def __init__(self,
                 serial_port='COM7',
                 serial_baud=9600,
                 pan_range=(-10.0, 8.0),
                 tilt_range=(-3.0, 12.0),
                 num_calib_points=16,
                 settle_time=2.0,
                 laser_thresh=225,
                 laser_blur=3,
                 depth_max=1300,
                 adjust_gain=0.05,
                 max_adjust_attempts=5,
                 nn_hidden=(32, 16),
                 nn_lr=0.01,
                 nn_epochs=2000,
                 track_interval=5.0):
        # 电机
        self.serial_port = serial_port
        self.serial_baud = serial_baud
        # 标定
        self.pan_range = pan_range
        self.tilt_range = tilt_range
        self.num_calib_points = num_calib_points
        self.settle_time = settle_time
        self.adjust_gain = adjust_gain
        self.max_adjust_attempts = max_adjust_attempts
        # 激光检测
        self.laser_thresh = laser_thresh
        self.laser_blur = laser_blur
        self.depth_max = depth_max
        # NN
        self.nn_hidden = nn_hidden
        self.nn_lr = nn_lr
        self.nn_epochs = nn_epochs
        # 追踪
        self.track_interval = track_interval

        # 运行时状态
        self.cam = None
        self.mc = None
        self.lr = None
        self.tr = None
        self.model = None
        self.calib_data = None
        self._motor_ok = False
        self._setup_done = False

    # ==========================================
    # 初始化 / 清理
    # ==========================================
    def setup(self):
        """初始化硬件：相机、电机、激光识别器、靶标识别器，并初始化激光ROI"""
        print("=" * 60)
        print("  开环激光指向控制器 V2 初始化")
        print("=" * 60)

        self.cam = CameraManager()
        self.cam.start()

        self.lr = LaserRecognizerV3(
            thresh_val=self.laser_thresh,
            blur_size=self.laser_blur,
            depth_max=self.depth_max,
        )
        self.lr.setup(self.cam)

        self.tr = TargetRecognizer()
        self.tr.setup(self.cam)

        self.mc = MotorController(port=self.serial_port, baudrate=self.serial_baud)
        try:
            self.mc.connect()
            print(f"电机已连接: {self.serial_port}")
            self._motor_ok = True
        except Exception as e:
            print(f"警告: 无法连接电机 ({e}) → 仅视觉模式")
            self._motor_ok = False

        if not self._motor_ok:
            print("电机未连接, 无法执行标定和指向。")
            return False

        # 靶标初始化 → 获取激光ROI圆盘
        print("\n>>> 等待三靶标初始化...")
        ref_frame, confirmed = self.tr.capture_initial_targets()
        if ref_frame is None:
            print("靶标初始化失败。")
            return False
        print(">>> 构建ORB模板...")
        self.tr.build_template(ref_frame, confirmed)

        dpt_init = self.cam.get_depth_map()
        self.lr.init_from_target_data(self.tr.ref_targets_centers, dpt_init)
        print(f"激光ROI已初始化: 圆心({self.lr.roi_center[0]}, {self.lr.roi_center[1]}), "
              f"半径={self.lr.roi_radius}px")

        self._setup_done = True
        print("初始化完成。\n")
        return True

    def shutdown(self):
        """清理资源：复位电机、断开连接、释放相机"""
        print("正在清理...")
        if self.mc is not None:
            try:
                self.mc.reset()
                time.sleep(1.0)
                self.mc.disconnect()
            except Exception:
                pass
        if self.cam is not None:
            self.cam.release()
        cv2.destroyAllWindows()
        self._setup_done = False
        print("已退出。")

    # ==========================================
    # 标定: 网格采样 + 训练
    # ==========================================
    def calibrate(self, verbose=True):
        """执行完整标定流程：Phase 1 网格采样 + Phase 2 训练NN

        Returns:
            self (支持链式调用)
        """
        if not self._setup_done:
            raise RuntimeError("请先调用 setup() 初始化")

        calib_data = self._phase1_sample()
        if len(calib_data) < 3:
            raise RuntimeError(f"标定数据不足 ({len(calib_data)} 组), 至少需要3组")

        self.calib_data = calib_data
        self.model = self._phase2_train(calib_data, verbose=verbose)
        return self

    def _phase1_sample(self):
        """均匀网格采样 → 开环指向 → 采集(摄像头坐标, 电机角度)对"""
        print("\n" + "=" * 60)
        print("  Phase 1: 开环标定采样 (ROI约束 + 方向调整)")
        print("=" * 60)

        if self.lr.initialized:
            print(f"  ROI圆盘: 圆心({self.lr.roi_center[0]}, {self.lr.roi_center[1]}), "
                  f"半径={self.lr.roi_radius}px")
        else:
            print("  警告: LaserRecognizerV3 未初始化ROI")

        raw_lr = LaserRecognizer(depth_max=self.depth_max,
                                 thresh_val=self.laser_thresh,
                                 blur_size=self.laser_blur)
        raw_lr.setup(self.cam)

        print("复位电机...")
        self.mc.reset()
        time.sleep(10)

        motor_pts = uniform_grid_2d(self.num_calib_points,
                                    self.pan_range,
                                    self.tilt_range)
        print(f"均匀网格生成 {self.num_calib_points} 个采样点:")
        for i, (p, t) in enumerate(motor_pts):
            print(f"  [{i:2d}] pan={p:+7.2f}°, tilt={t:+7.2f}°")

        calib_data = []
        retry_queue = list(motor_pts)

        cv2.namedWindow("Calibration")

        while len(calib_data) < self.num_calib_points:
            if not retry_queue:
                extra = uniform_grid_2d(self.num_calib_points - len(calib_data),
                                        (self.pan_range[0] - 2.0, self.pan_range[1] + 2.0),
                                        (self.tilt_range[0] - 2.0, self.tilt_range[1] + 2.0))
                retry_queue = list(extra)
                print(f"队列耗尽, 补充 {len(extra)} 个点")

            base_pan, base_tilt = retry_queue.pop(0)
            pan, tilt = base_pan, base_tilt
            success = False

            for attempt in range(self.max_adjust_attempts):
                print(f"指向 pan={pan:+6.2f}°, tilt={tilt:+6.2f}° ... ", end="", flush=True)
                self.mc.set_position(pan, tilt)
                time.sleep(self.settle_time)

                # V3检测 (ROI约束)
                spot_info = None
                for _ in range(15):
                    frame = self.cam.read_rgb_frame()
                    if frame is None:
                        continue
                    dpt = self.cam.get_depth_map()
                    _, _, spot_info, _ = self.lr.process_frame(frame, dpt)
                    if spot_info is not None:
                        break
                    time.sleep(0.1)

                if spot_info is not None:
                    cx, cy, depth_val, quality = spot_info
                    calib_data.append((cx, cy, pan, tilt))
                    print(f"OK → cam=({cx:4d},{cy:4d})  "
                          f"d={depth_val:.0f}mm q={quality:.2f}  "
                          f"[{len(calib_data)}/{self.num_calib_points}]")
                    disp = frame.copy()
                    cv2.circle(disp, (cx, cy), 15, (0, 0, 255), 2)
                    cv2.circle(disp, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(disp, f"L({cx},{cy}) -> M({pan:.1f},{tilt:.1f})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(disp, f"Progress: {len(calib_data)}/{self.num_calib_points}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if self.lr.initialized:
                        cv2.circle(disp, self.lr.roi_center, self.lr.roi_radius,
                                   (255, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow("Calibration", disp)
                    cv2.waitKey(1)
                    success = True
                    break

                # V3未找到 → 用原始检测器定位
                frame = self.cam.read_rgb_frame()
                if frame is None:
                    print("无画面")
                    break
                dpt = self.cam.get_depth_map()
                _, _, raw_spot = raw_lr.process_frame(frame, dpt,
                                                      self.depth_max,
                                                      self.laser_thresh,
                                                      self.laser_blur)

                if raw_spot is None:
                    print(f"未检测到任何激光点 (第{attempt+1}次)")
                    break

                lx, ly, _ = raw_spot

                if not self.lr.initialized:
                    print(f"激光({lx},{ly}) 但ROI未初始化, 无法调整")
                    break

                err_x = self.lr.roi_center[0] - lx
                err_y = self.lr.roi_center[1] - ly
                dist_to_roi = math.hypot(err_x, err_y)

                if dist_to_roi <= self.lr.roi_radius:
                    calib_data.append((lx, ly, pan, tilt))
                    print(f"OK(RAW) → cam=({lx:4d},{ly:4d})  "
                          f"[{len(calib_data)}/{self.num_calib_points}]")
                    success = True
                    break

                pan += err_x * self.adjust_gain
                tilt += err_y * self.adjust_gain
                print(f"激光({lx},{ly})距ROI中心{dist_to_roi:.0f}px, "
                      f"调整→ pan={pan:+.2f} tilt={tilt:+.2f}")

            if not success:
                print("  调整次数用尽, 跳过此点")

            cv2.waitKey(1)

        cv2.destroyWindow("Calibration")
        print(f"\n标定完成! 共计 {len(calib_data)} 组有效数据\n")
        return calib_data

    def _phase2_train(self, calib_data, verbose=True):
        """训练MLP: 摄像头坐标(cx, cy) → 电机开环角度(pan, tilt)"""
        print("=" * 60)
        print("  Phase 2: 训练神经网络")
        print("=" * 60)

        X = np.array([[cx, cy] for cx, cy, _, _ in calib_data], dtype=np.float32)
        y = np.array([[pan, tilt] for _, _, pan, tilt in calib_data], dtype=np.float32)

        if verbose:
            print(f"输入 (摄像头坐标):  {X.tolist()}")
            print(f"输出 (电机角度):    {y.tolist()}")
            print(f"\n网络结构: 2 → {self.nn_hidden[0]} → {self.nn_hidden[1]} → 2")
            print(f"学习率: {self.nn_lr}, 迭代: {self.nn_epochs}\n")

        model = SimpleMLP(hidden_layers=self.nn_hidden,
                          lr=self.nn_lr, max_iter=self.nn_epochs)
        model.fit(X, y, verbose=verbose)

        if verbose:
            y_pred = model.predict(X)
            print("\n训练数据拟合对比:")
            print(f"{'摄像头坐标':<16s} {'实际角度':<22s} {'预测角度':<22s} {'误差':<16s}")
            print("-" * 76)
            for i in range(len(X)):
                err_pan = y[i, 0] - y_pred[i, 0]
                err_tilt = y[i, 1] - y_pred[i, 1]
                print(f"({X[i,0]:4.0f},{X[i,1]:4.0f})        "
                      f"({y[i,0]:+7.2f},{y[i,1]:+7.2f})         "
                      f"({y_pred[i,0]:+7.2f},{y_pred[i,1]:+7.2f})         "
                      f"({err_pan:+5.2f},{err_tilt:+5.2f})")
            print()

        return model

    # ==========================================
    # 核心: 输入坐标 → 开环指向
    # ==========================================
    def point_to(self, pixel_x, pixel_y, clip=True):
        """输入摄像头像素坐标，开环指向该位置。

        Args:
            pixel_x: 像素X坐标
            pixel_y: 像素Y坐标
            clip: 是否限幅电机角度到标定范围的2倍

        Returns:
            (pan, tilt): 实际发送的电机角度(度)
        """
        if self.model is None:
            raise RuntimeError("模型未训练, 请先调用 calibrate() 或 load_model()")
        if not self._motor_ok:
            raise RuntimeError("电机未连接")

        cam_pt = np.array([[pixel_x, pixel_y]], dtype=np.float32)
        pred = self.model.predict(cam_pt)
        pan, tilt = float(pred[0, 0]), float(pred[0, 1])

        if clip:
            pan = np.clip(pan, self.pan_range[0] * 2, self.pan_range[1] * 2)
            tilt = np.clip(tilt, self.tilt_range[0] * 2, self.tilt_range[1] * 2)

        self.mc.set_position(pan, tilt)
        return pan, tilt

    def predict(self, pixel_x, pixel_y, clip=True):
        """仅预测电机角度，不实际移动（用于调试/验证）。

        Args:
            pixel_x: 像素X坐标
            pixel_y: 像素Y坐标
            clip: 是否限幅

        Returns:
            (pan, tilt): 预测的电机角度(度)
        """
        if self.model is None:
            raise RuntimeError("模型未训练, 请先调用 calibrate() 或 load_model()")

        cam_pt = np.array([[pixel_x, pixel_y]], dtype=np.float32)
        pred = self.model.predict(cam_pt)
        pan, tilt = float(pred[0, 0]), float(pred[0, 1])

        if clip:
            pan = np.clip(pan, self.pan_range[0] * 2, self.pan_range[1] * 2)
            tilt = np.clip(tilt, self.tilt_range[0] * 2, self.tilt_range[1] * 2)

        return pan, tilt

    # ==========================================
    # 模型持久化
    # ==========================================
    def save_model(self, filepath):
        """保存标定数据和模型"""
        if self.model is None:
            raise RuntimeError("无模型可保存")
        self.model.save(filepath)

    def load_model(self, filepath):
        """加载已有模型（跳过标定）"""
        if not self._setup_done:
            raise RuntimeError("请先调用 setup() 初始化")
        self.model = SimpleMLP.load(filepath)
        print("模型加载完成，可直接调用 point_to()。")

    # ==========================================
    # 持续追踪
    # ==========================================
    def start_tracking(self):
        """持续追踪绿色靶标，定时开环指向"""
        if self.model is None:
            raise RuntimeError("模型未训练, 请先调用 calibrate() 或 load_model()")

        print("=" * 60)
        print("  开环追踪绿色靶标")
        print("=" * 60)

        cv2.namedWindow("Open-Loop Track")
        cv2.namedWindow("Tracking Binary (Masked)")

        last_point_time = 0.0
        paused = False

        while True:
            print("\n>>> 等待三靶标初始化...")
            ref_frame, confirmed = self.tr.capture_initial_targets()
            if ref_frame is None:
                break
            print(">>> 构建ORB模板...")
            self.tr.build_template(ref_frame, confirmed)

            dpt_init = self.cam.get_depth_map()
            self.lr.init_from_target_data(self.tr.ref_targets_centers, dpt_init)

            while True:
                frame = self.cam.read_rgb_frame()
                if frame is None:
                    break
                dpt = self.cam.get_depth_map()
                self.tr.depth_map = dpt

                gray, binary = self.tr.preprocess(frame)
                track_ok, all_targets, tr_result, rot_angle, _ = \
                    self.tr.track_frame(frame, binary)

                if not track_ok and self.tr.consecutive_tri_fail >= 7:
                    print("\n追踪丢失! 重新初始化...\n")
                    break

                _, _, laser_spot, _ = self.lr.process_frame(frame, dpt)
                green_pos = all_targets.get("Green") if all_targets else None

                now = time.time()
                if green_pos is not None and not paused and \
                   (now - last_point_time) >= self.track_interval:
                    gx, gy = green_pos
                    try:
                        pan, tilt = self.point_to(gx, gy)
                        print(f"[t={now:.1f}s] Green=({gx},{gy}) → "
                              f"Motor=({pan:+.2f}°, {tilt:+.2f}°)")
                    except Exception as e:
                        print(f"  指向失败: {e}")
                    last_point_time = now

                # 可视化
                display = tr_result
                if laser_spot is not None:
                    lx, ly, ld, lq = laser_spot
                    cv2.circle(display, (lx, ly), 15, (0, 0, 255), 2)
                    cv2.circle(display, (lx, ly), 3, (0, 0, 255), -1)
                    cv2.putText(display, f"Laser({lx},{ly}) d={ld:.0f}mm",
                                (lx + 20, ly - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if green_pos is not None:
                    gx, gy = green_pos
                    cv2.putText(display, f"Green Target({gx},{gy})",
                                (gx + 15, gy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if laser_spot is not None and green_pos is not None:
                    lx, ly, _, _ = laser_spot
                    cv2.line(display, (lx, ly), (gx, gy), (255, 255, 0), 1, cv2.LINE_AA)

                elapsed = now - last_point_time
                remain = max(0, self.track_interval - elapsed)
                cv2.putText(display,
                            f"Open-Loop V2 | Next: {remain:.1f}s | "
                            f"Interval: {self.track_interval}s",
                            (10, display.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                if paused:
                    cv2.putText(display, "PAUSED",
                                (display.shape[1] // 2 - 50, display.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                binary_masked = np.zeros_like(binary)
                cv2.circle(binary_masked, self.tr.ref_mask_center,
                           self.tr.ref_mask_radius, 255, -1)
                binary_masked = cv2.bitwise_and(binary, binary_masked)

                cv2.imshow("Open-Loop Track", display)
                cv2.imshow("Tracking Binary (Masked)", binary_masked)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n退出追踪...")
                    return
                elif key == ord('r'):
                    print("复位电机...")
                    try:
                        self.mc.reset()
                    except Exception as e:
                        print(f"复位失败: {e}")
                    last_point_time = 0.0
                elif key == ord(' '):
                    paused = not paused
                    print("已暂停" if paused else "继续")

            cv2.destroyWindow("Tracking Binary (Masked)")

    # ==========================================
    # 便捷: 获取当前帧中的绿色靶标坐标
    # ==========================================
    def detect_green_target(self):
        """检测当前帧中的绿色靶标坐标

        Returns:
            (gx, gy) 或 None
        """
        frame = self.cam.read_rgb_frame()
        if frame is None:
            return None
        dpt = self.cam.get_depth_map()
        self.tr.depth_map = dpt

        _, binary = self.tr.preprocess(frame)
        track_ok, all_targets, _, _, _ = self.tr.track_frame(frame, binary)

        if track_ok and all_targets:
            return all_targets.get("Green")
        return None

    # ==========================================
    # 属性
    # ==========================================
    @property
    def is_calibrated(self):
        return self.model is not None

    @property
    def motor_ok(self):
        return self._motor_ok


# ==========================================
# 主函数 (兼容直接运行)
# ==========================================
def main():
    ctrl = OpenLoopController(
        serial_port='COM7',
        pan_range=(-8.0, 8.0),
        tilt_range=(-3.0, 11.0),
        num_calib_points=16,
        track_interval=5.0,
    )

    if not ctrl.setup():
        print("初始化失败, 退出。")
        return

    time.sleep(1.0)

    try:
        ctrl.calibrate()

        print("\n标定数据汇总:")
        print(f"{'摄像头坐标':<16s} {'电机角度':<22s}")
        print("-" * 38)
        for cx, cy, pan, tilt in ctrl.calib_data:
            print(f"({cx:4d},{cy:4d})        ({pan:+7.2f}°, {tilt:+7.2f}°)")

        print("\n进入开环追踪阶段...")
        print(f"按 q 退出 | 按 r 复位 | 按 空格 暂停\n")
        ctrl.start_tracking()
    finally:
        ctrl.shutdown()


if __name__ == "__main__":
    main()
