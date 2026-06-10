"""
开环激光指向控制系统
==============================
Phase 1: 均匀网格采样校准 - 开环指向采点, 收集 (摄像头坐标, 电机角度) 对
Phase 2: 训练神经网络 - 摄像头坐标 → 电机开环坐标
Phase 3: 开环追踪 - 每10秒识别绿色靶标, NN预测角度, 开环指向

使用方法:
    python open_loop_control.py
    按 q 退出 | 按 r 复位电机
"""

import cv2
import numpy as np
import time

from camera_manager import CameraManager
from laser_recognize import LaserRecognizer
from motor_control import MotorController
from target_recognize_v2 import TargetRecognizer


# ==========================================
# 全局可调参数
# ==========================================

# ---- 均匀网格采样范围 (电机角度: pan, tilt) ----
PAN_MIN, PAN_MAX = -5.0, 10.0
TILT_MIN, TILT_MAX = -3.0, 12.0
NUM_CALIB_POINTS = 16          # 校准采样点数

# ---- 电机控制 ----
SETTLE_TIME = 2.0              # 开环指向后电机稳定等待时间(秒)
SERIAL_PORT = 'COM7'
SERIAL_BAUD = 9600

# ---- 激光检测 ----
LASER_THRESH = 240
LASER_BLUR = 3
DEPTH_MAX = 1300

# ---- 开环追踪 ----
TRACK_INTERVAL = 5.0          # 开环指向间隔(秒)

# ---- 神经网络 ----
NN_HIDDEN = (32, 16)           # 隐藏层大小
NN_LR = 0.01                   # 学习率
NN_EPOCHS = 2000               # 训练轮数


# ==========================================
# 均匀网格采样
# ==========================================
def uniform_grid_2d(n, x_range, y_range):
    """在2D范围内均匀网格采点, 返回约 n 个点

    根据采样范围的长宽比自动分配网格行列数, 使点尽量接近方形网格。
    Returns:
        np.ndarray of shape (n, 2): [[x1, y1], ...]
    """
    w = x_range[1] - x_range[0]
    h = y_range[1] - y_range[0]

    # 按长宽比分配行列数
    n_x = max(2, int(np.ceil(np.sqrt(n * w / h))))
    n_y = max(2, int(np.ceil(n / n_x)))

    xs = np.linspace(x_range[0], x_range[1], n_x)
    ys = np.linspace(y_range[0], y_range[1], n_y)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])

    # 超出目标数量的部分从边缘均匀剔除
    if len(pts) > n:
        idx = np.linspace(0, len(pts) - 1, n, dtype=int)
        pts = pts[idx]
    return pts


# ==========================================
# 简单MLP回归器 (numpy实现)
# ==========================================
class SimpleMLP:
    """2层全连接神经网络回归器, 用于学习 摄像头坐标→电机角度 的映射"""

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
            # He初始化
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
                a = np.maximum(0, z)  # ReLU hidden
            else:
                a = z                 # Linear output
            activations.append(a)
        return activations, pre_acts

    def _backward(self, activations, pre_acts, y):
        m = y.shape[0]
        delta = (activations[-1] - y) * (2.0 / m)  # dL/da (MSE)

        grads_w = []
        grads_b = []
        for i in reversed(range(len(self.weights))):
            if i < len(self.weights) - 1:
                delta = delta * (pre_acts[i] > 0)  # ReLU grad
            dw = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            if i > 0:
                delta = delta @ self.weights[i].T
        return grads_w, grads_b

    def fit(self, X, y):
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

            if epoch % 500 == 0:
                loss = np.mean((act[-1] - yn) ** 2)
                print(f"    Epoch {epoch:4d}: loss={loss:.6f}")

        final = np.mean((self._forward(Xn)[0][-1] - yn) ** 2)
        print(f"    训练完成: final_loss={final:.6f}")

    def predict(self, X):
        Xn = (X - self.in_mean) / self.in_std
        yn = self._forward(Xn)[0][-1]
        return yn * self.out_std + self.out_mean


# ==========================================
# Phase 1: 开环校准采样
# ==========================================
def phase1_calibrate(mc, cam, lr):
    """均匀网格采样 → 开环指向 → 摄像头采集激光点位置

    Returns:
        list of (cam_cx, cam_cy, motor_pan, motor_tilt)
    """
    print("\n" + "=" * 60)
    print("  Phase 1: 开环校准采样")
    print("=" * 60)

    print("复位电机...")
    mc.reset()
    time.sleep(10)

    motor_pts = uniform_grid_2d(NUM_CALIB_POINTS,
                                 (PAN_MIN, PAN_MAX),
                                 (TILT_MIN, TILT_MAX))
    print(f"均匀网格生成 {NUM_CALIB_POINTS} 个采样点:")
    for i, (p, t) in enumerate(motor_pts):
        print(f"  [{i:2d}] pan={p:+7.2f}°, tilt={t:+7.2f}°")

    calib_data = []         # [(cx, cy, pan, tilt), ...]
    retry_queue = list(motor_pts)

    cv2.namedWindow("Calibration")

    while len(calib_data) < NUM_CALIB_POINTS:
        if not retry_queue:
            extra = uniform_grid_2d(NUM_CALIB_POINTS - len(calib_data),
                                     (PAN_MIN, PAN_MAX),
                                     (TILT_MIN, TILT_MAX))
            retry_queue = list(extra)
            print(f"队列耗尽, 补充 {len(extra)} 个点")

        pan, tilt = retry_queue.pop(0)

        # 开环指向
        print(f"指向 pan={pan:+6.2f}°, tilt={tilt:+6.2f}° ... ", end="", flush=True)
        mc.set_position(pan, tilt)
        time.sleep(SETTLE_TIME)

        # 多次尝试检测激光点
        spot = None
        for _ in range(15):
            frame = cam.read_rgb_frame()
            if frame is None:
                continue
            dpt = cam.get_depth_map()
            _, _, spot = lr.process_frame(frame, dpt, DEPTH_MAX, LASER_THRESH, LASER_BLUR)

            if spot is not None:
                cx, cy, _ = spot
                calib_data.append((cx, cy, pan, tilt))
                print(f"OK → cam=({cx:4d},{cy:4d})  [{len(calib_data)}/{NUM_CALIB_POINTS}]")

                # 可视化
                disp = frame.copy()
                cv2.circle(disp, (cx, cy), 15, (0, 0, 255), 2)
                cv2.circle(disp, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(disp, f"L({cx},{cy}) -> M({pan:.1f},{tilt:.1f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(disp, f"Progress: {len(calib_data)}/{NUM_CALIB_POINTS}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Calibration", disp)
                cv2.waitKey(1)
                break
            time.sleep(0.1)

        if spot is None:
            print("未检测到激光点 → 重新入队")
            retry_queue.append((pan, tilt))

        cv2.waitKey(1)

    cv2.destroyWindow("Calibration")
    print(f"\n校准完成! 共计 {len(calib_data)} 组有效数据\n")
    return calib_data


# ==========================================
# Phase 2: 训练神经网络
# ==========================================
def phase2_train(calib_data):
    """训练MLP: 摄像头坐标(cx, cy) → 电机开环角度(pan, tilt)

    Returns:
        SimpleMLP: 训练好的模型
    """
    print("=" * 60)
    print("  Phase 2: 训练神经网络")
    print("=" * 60)

    X = np.array([[cx, cy] for cx, cy, _, _ in calib_data], dtype=np.float32)
    y = np.array([[pan, tilt] for _, _, pan, tilt in calib_data], dtype=np.float32)

    print(f"输入 (摄像头坐标):  {X.tolist()}")
    print(f"输出 (电机角度):    {y.tolist()}")
    print(f"\n网络结构: 2 → {NN_HIDDEN[0]} → {NN_HIDDEN[1]} → 2")
    print(f"学习率: {NN_LR}, 迭代: {NN_EPOCHS}\n")

    model = SimpleMLP(hidden_layers=NN_HIDDEN, lr=NN_LR, max_iter=NN_EPOCHS)
    model.fit(X, y)

    # 验证: 打印预测结果对比
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
# Phase 3: 开环追踪绿色靶标
# ==========================================
def phase3_track(model, mc, cam, tr, lr):
    """每 TRACK_INTERVAL 秒识别绿色靶标, NN预测电机角度, 开环指向"""
    print("=" * 60)
    print("  Phase 3: 开环追踪绿色靶标")
    print("=" * 60)

    cv2.namedWindow("Open-Loop Track")
    cv2.namedWindow("Tracking Binary (Masked)")

    last_point_time = 0.0
    paused = False

    # --- 初始化靶标追踪 (Phase 1+2 of TargetRecognizer) ---
    while True:
        print("\n>>> 等待三靶标初始化...")
        ref_frame, confirmed = tr.capture_initial_targets()
        if ref_frame is None:
            break
        print(">>> 构建ORB模板...")
        tr.build_template(ref_frame, confirmed)

        prev_green_pos = None

        # --- 追踪循环 ---
        while True:
            frame = cam.read_rgb_frame()
            if frame is None:
                break
            dpt = cam.get_depth_map()
            tr.depth_map = dpt

            # 靶标追踪
            gray, binary = tr.preprocess(frame)
            track_ok, all_targets, tr_result, rot_angle, _ = \
                tr.track_frame(frame, binary)

            # 追踪丢失 → 重新初始化
            if not track_ok and tr.consecutive_tri_fail >= 7:
                print("\n追踪丢失! 重新初始化...\n")
                break

            # 检测激光点 (仅用于可视化)
            _, _, laser_spot = lr.process_frame(frame, dpt, DEPTH_MAX,
                                                 LASER_THRESH, LASER_BLUR)

            # 提取绿色靶标坐标
            green_pos = all_targets.get("Green") if all_targets else None

            # --- 定时开环指向 ---
            now = time.time()
            if green_pos is not None and not paused and (now - last_point_time) >= TRACK_INTERVAL:
                gx, gy = green_pos

                # NN预测: 摄像头坐标 → 电机角度
                cam_pt = np.array([[gx, gy]], dtype=np.float32)
                pred = model.predict(cam_pt)
                pred_pan, pred_tilt = float(pred[0, 0]), float(pred[0, 1])

                # 限幅
                pred_pan = np.clip(pred_pan, PAN_MIN * 2, PAN_MAX * 2)
                pred_tilt = np.clip(pred_tilt, TILT_MIN * 2, TILT_MAX * 2)

                print(f"[t={now:.1f}s] Green=({gx},{gy}) → "
                      f"Motor=({pred_pan:+.2f}°, {pred_tilt:+.2f}°)")

                try:
                    mc.set_position(pred_pan, pred_tilt)
                except Exception as e:
                    print(f"  电机通信错误: {e}")

                last_point_time = now
                prev_green_pos = green_pos

            # --- 可视化 ---
            display = tr_result

            # 激光点
            if laser_spot is not None:
                lx, ly, ld = laser_spot
                cv2.circle(display, (lx, ly), 15, (0, 0, 255), 2)
                cv2.circle(display, (lx, ly), 3, (0, 0, 255), -1)
                cv2.putText(display, f"Laser({lx},{ly}) d={ld:.0f}mm",
                            (lx + 20, ly - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 预测指向位置标记
            if green_pos is not None:
                gx, gy = green_pos
                cv2.putText(display, f"Green Target({gx},{gy})",
                            (gx + 15, gy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 激光→绿色靶标连线
            if laser_spot is not None and green_pos is not None:
                lx, ly, _ = laser_spot
                gx, gy = green_pos
                cv2.line(display, (lx, ly), (gx, gy), (255, 255, 0), 1, cv2.LINE_AA)

            # 状态信息
            elapsed = now - last_point_time
            remain = max(0, TRACK_INTERVAL - elapsed)
            status = (f"Open-Loop | Next point in: {remain:.1f}s | "
                      f"Interval: {TRACK_INTERVAL}s")
            cv2.putText(display, status, (10, display.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            if paused:
                cv2.putText(display, "PAUSED",
                            (display.shape[1] // 2 - 50, display.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # 追踪二值图
            binary_masked = np.zeros_like(binary)
            cv2.circle(binary_masked, tr.ref_mask_center, tr.ref_mask_radius, 255, -1)
            binary_masked = cv2.bitwise_and(binary, binary_masked)

            cv2.imshow("Open-Loop Track", display)
            cv2.imshow("Tracking Binary (Masked)", binary_masked)

            # --- 键盘 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n退出...")
                return
            elif key == ord('r'):
                print("复位电机...")
                try:
                    mc.reset()
                except Exception as e:
                    print(f"复位失败: {e}")
                last_point_time = 0.0
            elif key == ord(' '):
                paused = not paused
                print("已暂停" if paused else "继续")

        cv2.destroyWindow("Tracking Binary (Masked)")


# ==========================================
# 主函数
# ==========================================
def main():
    print("=" * 60)
    print("  开环激光指向控制系统")
    print("  均匀网格校准 → NN训练 → 开环追踪绿色靶标")
    print("=" * 60)
    print(f"  采样范围: pan=[{PAN_MIN}, {PAN_MAX}], tilt=[{TILT_MIN}, {TILT_MAX}]")
    print(f"  采样点数: {NUM_CALIB_POINTS}")
    print(f"  追踪间隔: {TRACK_INTERVAL}s")
    print(f"  NN结构: 2→{NN_HIDDEN[0]}→{NN_HIDDEN[1]}→2")
    print()

    # ---- 初始化硬件 ----
    cam = CameraManager()
    cam.start()

    lr = LaserRecognizer(depth_max=DEPTH_MAX,
                          thresh_val=LASER_THRESH,
                          blur_size=LASER_BLUR)
    lr.setup(cam)

    tr = TargetRecognizer()
    tr.setup(cam)

    mc = MotorController(port=SERIAL_PORT, baudrate=SERIAL_BAUD)
    motor_ok = True
    try:
        mc.connect()
        print(f"电机已连接: {SERIAL_PORT}\n")
    except Exception as e:
        print(f"警告: 无法连接电机 ({e}) → 仅视觉模式\n")
        motor_ok = False

    if not motor_ok:
        print("电机未连接, 无法执行校准和追踪。退出。")
        cam.release()
        return

    # ---- Phase 1: 校准 ----
    time.sleep(1.0)  # 等待系统稳定
    calib_data = phase1_calibrate(mc, cam, lr)

    print("\n校准数据汇总:")
    print(f"{'摄像头坐标':<16s} {'电机角度':<22s}")
    print("-" * 38)
    for cx, cy, pan, tilt in calib_data:
        print(f"({cx:4d},{cy:4d})        ({pan:+7.2f}°, {tilt:+7.2f}°)")

    # ---- Phase 2: 训练 ----
    model = phase2_train(calib_data)

    # ---- Phase 3: 开环追踪 ----
    print("\n进入开环追踪阶段...")
    print(f"按 q 退出 | 按 r 复位 | 按 空格 暂停\n")
    phase3_track(model, mc, cam, tr, lr)

    # ---- 清理 ----
    print("正在清理...")
    try:
        mc.reset()
        time.sleep(1.0)
        mc.disconnect()
    except Exception:
        pass
    cam.release()
    print("已退出。")


if __name__ == "__main__":
    main()
