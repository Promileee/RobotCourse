"""
mission_open_loop_v3.py
完整流程——开环版本 V3

相比 V2 的改进:
  1. 物料(rect_recognize)在整个程序运行期间不变，仅在电机初始化前识别一次
  2. 靶标(target_recognize_v2)可能发生变化，采用持续追踪模式：
     指向靶标前先持续追踪一段时间，让卡尔曼滤波器收敛，再用稳定坐标指向

使用方法:
    python mission_open_loop_v3.py
    按 'q' 可在各阶段退出
"""

import cv2
import time
import numpy as np

from camera_manager import CameraManager
from QR_code_recognize_v1 import QRCodeRecognizer
import rect_recognize as rr
from target_recognize_v2 import TargetRecognizerV2
from laser_recognize_v3 import LaserRecognizerV3
from motor_control import MotorController
from open_loop_control_v2 import OpenLoopController

# ==========================================
# 全局参数
# ==========================================
SERIAL_PORT = 'COM7'
SERIAL_BAUD = 9600
SETTLE_TIME = 2.0
QR_TIMEOUT = 20.0
THRESH_VAL = 230
MATERIAL_TIMEOUT = 15.0
RESET_WAIT = 10.0       # 电机复位后等待时间(秒)
QR_WAIT_AFTER = 5.0     # 二维码识别成功后等待时间(秒)
MATERIAL_TILT_OFFSET = -3.0  # 物料指向tilt补偿(度)，负值=向下

ROI_SCALE = 1.75          # 激光ROI圆盘半径倍数 (1.7/1.8/1.9)
TARGET_TRACK_TIME = 3.0  # 靶标持续追踪时间(秒)，让卡尔曼滤波器收敛

DIGIT_TO_COLOR = {'1': 'Red', '2': 'Green', '3': 'Blue'}
COLOR_CN = {'Red': '红', 'Green': '绿', 'Blue': '蓝'}


# ==========================================
# 二维码识别
# ==========================================
def read_qr_code(cam):
    """识别二维码，读取颜色顺序。识别成功自动确认。

    Returns:
        sequence: ['Red', 'Green', 'Blue'] 或 None
        raw_data: 原始二维码字符串
    """
    qr = QRCodeRecognizer()
    print("\n" + "=" * 60)
    print("  QR码识别 (QR_code_recognize_v1)")
    print("=" * 60)
    print("请将二维码对准摄像头，识别成功后将自动确认...")
    print("按 'q' 跳过\n")

    cv2.namedWindow("QR Scanner")
    sequence = None
    raw_data = None
    start_time = time.time()

    while time.time() - start_time < QR_TIMEOUT:
        frame = cam.read_rgb_frame()
        if frame is None:
            continue

        data, bbox, _ = qr.detect(frame)

        if data:
            sequence = [DIGIT_TO_COLOR.get(ch, '?') for ch in data]
            raw_data = data

            qr.draw_result(frame, data, bbox)
            label = f"QR: {data} -> {' -> '.join(COLOR_CN.get(c, c) for c in sequence)}"
            cv2.putText(frame, label, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Detected! Waiting {QR_WAIT_AFTER:.0f}s...",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("QR Scanner", frame)
            cv2.waitKey(1)
            break

        cv2.putText(frame, "q: skip", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("QR Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if sequence:
        print(f"识别结果: {raw_data} -> {' -> '.join(COLOR_CN.get(c, c) for c in sequence)}")
        print(f"等待 {QR_WAIT_AFTER:.0f} 秒，请撤出二维码及人手...")
        t0 = time.time()
        while time.time() - t0 < QR_WAIT_AFTER:
            frame = cam.read_rgb_frame()
            if frame is not None:
                elapsed = time.time() - t0
                remain = max(0, QR_WAIT_AFTER - elapsed)
                cv2.putText(frame, f"Waiting {remain:.1f}s... remove QR & hands",
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
                cv2.imshow("QR Scanner", frame)
            cv2.waitKey(100)
    else:
        print("未识别到二维码")

    cv2.destroyWindow("QR Scanner")
    return sequence, raw_data


# ==========================================
# 圆柱物料识别 (仅执行一次，重复5次取trim-mean)
# ==========================================
def detect_materials(cam):
    """识别圆柱物料(矩形靶标)，重复5次取稳定检测结果。

    对每种颜色收集5组中心坐标，剔除最大最小值后取剩余3组的均值，
    降低单次误识别的影响。

    指向坐标: 水平=中心x, 垂直=矩形底部(y+h)

    Returns:
        materials: {'Red': (point_cx, point_cy_bottom), ...} 或 {}
    """
    N_ROUNDS = 5

    print("\n" + "=" * 60)
    print(f"  圆柱物料识别 (rect_recognize) —— 重复{N_ROUNDS}次取均值")
    print("=" * 60)
    print("请确保红/绿/蓝三种圆柱物料在摄像头视野内...")

    cv2.namedWindow("Material Detection")

    # collected: {'Red': [(point_cx, point_cy_bottom), ...], ...}
    collected = {color: [] for color in ['Red', 'Green', 'Blue']}

    for round_idx in range(N_ROUNDS):
        print(f"\n--- 第 {round_idx + 1}/{N_ROUNDS} 次检测 ---")

        stab = rr.init_stability_state(skip_frames=5, stable_frames=3)
        round_start = time.time()
        detected = {}

        while time.time() - round_start < MATERIAL_TIMEOUT:
            frame = cam.read_rgb_frame()
            if frame is None:
                continue

            dpt = cam.get_depth_map()
            result, frame_rects = rr.detect_frame(frame, dpt)

            is_stable, corrected = rr.update_stability(stab, frame_rects)
            if is_stable:
                for color, rc in corrected.items():
                    if rc is not None:
                        x, y, _, h, cx, cy, d_val = rc
                        point_cx = cx
                        point_cy = y + h
                        detected[color] = (point_cx, point_cy)
                        print(f"  {COLOR_CN.get(color, color)}: "
                              f"中心({cx},{cy}) 指向({point_cx},{point_cy}) "
                              f"d={d_val:.0f}mm")
                break

            for name, rc in frame_rects.items():
                if rc is not None:
                    x, y, _, h, cx, cy, d_val = rc
                    cv2.putText(result, f"{name}:({cx},{cy})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.circle(result, (cx, y + h), 5, (0, 255, 255), -1)

            cv2.putText(result, f"Round {round_idx + 1}/{N_ROUNDS}",
                        (10, result.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Material Detection", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for color in ['Red', 'Green', 'Blue']:
            if color in detected:
                collected[color].append(detected[color])
            else:
                print(f"  ! {COLOR_CN.get(color, color)}未检测到")

    cv2.destroyWindow("Material Detection")

    # trim-mean: 每种颜色对x和y分别排序，去掉一个最大一个最小，取中间3个均值
    materials = {}
    for color in ['Red', 'Green', 'Blue']:
        pts = collected[color]
        if len(pts) < 3:
            print(f"警告: {COLOR_CN.get(color, color)}有效检测次数不足 "
                  f"({len(pts)}/{N_ROUNDS})")
            if len(pts) > 0:
                avg_cx = int(round(sum(p[0] for p in pts) / len(pts)))
                avg_cy = int(round(sum(p[1] for p in pts) / len(pts)))
                materials[color] = (avg_cx, avg_cy)
            continue

        xs = sorted([p[0] for p in pts])
        ys = sorted([p[1] for p in pts])

        trim_xs = xs[1:-1]
        trim_ys = ys[1:-1]

        avg_cx = int(round(sum(trim_xs) / len(trim_xs)))
        avg_cy = int(round(sum(trim_ys) / len(trim_ys)))

        materials[color] = (avg_cx, avg_cy)
        print(f"\n  {COLOR_CN.get(color, color)}最终: ({avg_cx}, {avg_cy})")
        print(f"    原始x: {xs} -> trim: {trim_xs}")
        print(f"    原始y: {ys} -> trim: {trim_ys}")

    if not materials:
        print("警告: 物料检测未完成")
    return materials


# ==========================================
# 靶标持续追踪
# ==========================================
def track_target_continuously(ctrl, tr, target_color, duration=TARGET_TRACK_TIME):
    """持续追踪指定颜色的靶标，让卡尔曼滤波器收敛。

    在 duration 秒内持续调用 track_frame，使 V2 的中值+卡尔曼滤波器
    充分收敛到稳定位置。追踪结束后返回滤波后的坐标。

    Args:
        ctrl: OpenLoopController
        tr: TargetRecognizerV2
        target_color: 'Red' / 'Green' / 'Blue'
        duration: 追踪时长(秒)

    Returns:
        (tgt_cx, tgt_cy) 或 None
    """
    cn = COLOR_CN.get(target_color, target_color)
    print(f"  持续追踪{cn}靶标 ({duration:.0f}s)...")

    cv2.namedWindow("Target Tracking")
    best_pos = None
    start_time = time.time()

    while time.time() - start_time < duration:
        frame = ctrl.cam.read_rgb_frame()
        if frame is None:
            continue

        dpt = ctrl.cam.get_depth_map()
        tr.depth_map = dpt
        _, binary = tr.preprocess(frame)
        track_ok, all_targets, tr_result, _, _ = tr.track_frame(frame, binary)

        # 更新当前最佳位置（来自滤波后的结果）
        if track_ok and all_targets and target_color in all_targets:
            best_pos = all_targets[target_color]

        # 显示追踪状态
        if tr_result is not None:
            elapsed = time.time() - start_time
            remain = max(0, duration - elapsed)
            cv2.putText(tr_result, f"Tracking {cn}: {remain:.1f}s",
                        (10, tr_result.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Target Tracking", tr_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("Target Tracking")

    if best_pos is not None:
        print(f"  {cn}靶标追踪完成: ({best_pos[0]}, {best_pos[1]})")
    else:
        print(f"  {cn}靶标追踪失败: 未捕获到有效位置")

    return best_pos


# ==========================================
# 执行任务序列
# ==========================================
def execute_sequence(ctrl, tr, sequence, materials):
    """按QR码顺序执行指向：每个颜色先指物料(底部)再指靶标

    靶标采用持续追踪模式，先追踪一段时间让滤波器收敛，再指向。

    Args:
        ctrl: OpenLoopController (已标定)
        tr: TargetRecognizerV2 (已初始化)
        sequence: ['Red', 'Green', 'Blue'] 颜色顺序
        materials: {'Red': (point_cx, point_cy), ...} 物料底部指向坐标
    """
    print("\n" + "=" * 60)
    print("  执行任务序列")
    print("=" * 60)

    color_order = " -> ".join(f"{COLOR_CN.get(c, c)}物料→{COLOR_CN.get(c, c)}靶标"
                              for c in sequence)
    print(f"执行顺序: {color_order}\n")

    for i, color in enumerate(sequence):
        step = i + 1
        cn = COLOR_CN.get(color, color)
        print(f"--- 第 {step}/{len(sequence)} 步: {cn} ---")

        # ==========================================
        # 指向物料 (水平=中心x, 垂直=底部y+h)
        # ==========================================
        if materials and color in materials:
            mat_cx, mat_cy = materials[color]
            print(f"  指向{cn}物料 ({mat_cx}, {mat_cy})")
            try:
                pan, tilt = ctrl.predict(mat_cx, mat_cy)
                tilt += MATERIAL_TILT_OFFSET
                print(f"  -> NN预测: pan={pan:+.2f}°, tilt={tilt:+.2f}° "
                      f"(补偿{MATERIAL_TILT_OFFSET:+.1f}°)")
                ctrl.mc.set_position(pan, tilt)
            except Exception as e:
                print(f"  ! 指向物料失败: {e}")
        else:
            print(f"  ! 跳过: {cn}物料坐标未知")

        print(f"  等待{cn}物料稳定...")
        time.sleep(SETTLE_TIME)

        # ==========================================
        # 指向靶标 —— 持续追踪模式
        # ==========================================
        tgt_hit = False

        # 先持续追踪，让滤波器收敛
        tracked_pos = track_target_continuously(ctrl, tr, color)

        if tracked_pos is not None:
            tgt_cx, tgt_cy = tracked_pos
            print(f"  指向{cn}靶标 ({tgt_cx}, {tgt_cy})")
            try:
                pan, tilt = ctrl.point_to(tgt_cx, tgt_cy)
                print(f"  -> 电机: pan={pan:+.2f}°, tilt={tilt:+.2f}°")
                tgt_hit = True
            except Exception as e:
                print(f"  ! 指向靶标失败: {e}")

        # 追踪失败时回退到参考位置
        if not tgt_hit and tr.ref_targets_centers and color in tr.ref_targets_centers:
            tgt_cx, tgt_cy = tr.ref_targets_centers[color]
            print(f"  指向{cn}靶标(参考) ({tgt_cx}, {tgt_cy})")
            try:
                pan, tilt = ctrl.point_to(tgt_cx, tgt_cy)
                print(f"  -> 电机: pan={pan:+.2f}°, tilt={tilt:+.2f}°")
            except Exception as e:
                print(f"  ! 指向靶标失败: {e}")

        if not tgt_hit and not (tr.ref_targets_centers and color in tr.ref_targets_centers):
            print(f"  ! 跳过: {cn}靶标未找到且无参考位置")

        print(f"  等待{cn}靶标稳定...")
        time.sleep(SETTLE_TIME)

    print("\n" + "=" * 60)
    print("  任务序列执行完毕!")
    print("=" * 60)


# ==========================================
# 靶标 & 激光初始化 (仅首次执行)
# ==========================================
def init_targets_and_laser(ctrl):
    """用 target_recognize_v2 + laser_recognize_v3 替换控制器默认的 V1"""
    print("=" * 60)
    print("  靶标 & 激光识别初始化")
    print("  (target_recognize_v2 + laser_recognize_v3)")
    print("=" * 60)

    tr_v2 = TargetRecognizerV2()
    tr_v2.setup(ctrl.cam)

    tr_v2.ref_targets_centers = dict(ctrl.tr.ref_targets_centers)
    tr_v2.ref_mask_center = ctrl.tr.ref_mask_center
    tr_v2.ref_mask_radius = ctrl.tr.ref_mask_radius

    frame = ctrl.cam.read_rgb_frame()
    if frame is not None:
        gray, binary = tr_v2.preprocess(frame)
        tr_v2.ref_binary = binary

        confirmed = {}
        for color, (cx, cy) in tr_v2.ref_targets_centers.items():
            confirmed[color] = (0, 0, 1, 1, cx, cy)
        tr_v2.build_template(frame, confirmed)

    lr_v3 = LaserRecognizerV3(
        thresh_val=THRESH_VAL, blur_size=3, depth_max=1300,
        roi_scale=ROI_SCALE,
    )
    lr_v3.setup(ctrl.cam)
    dpt = ctrl.cam.get_depth_map()
    lr_v3.init_from_target_data(ctrl.tr.ref_targets_centers, dpt)

    ctrl.tr = tr_v2
    ctrl.lr = lr_v3

    print(f"激光ROI: 圆心({lr_v3.roi_center[0]}, {lr_v3.roi_center[1]}), "
          f"半径={lr_v3.roi_radius}px")
    for color, (cx, cy) in tr_v2.ref_targets_centers.items():
        print(f"  {COLOR_CN.get(color, color)}靶标参考: ({cx}, {cy})")

    return tr_v2


# ==========================================
# 主函数
# ==========================================
def main():
    print("=" * 60)
    print("  完整流程 —— 开环版本 V3 (循环执行)")
    print("=" * 60)
    print(f"  电机串口: {SERIAL_PORT}")
    print(f"  稳定等待: {SETTLE_TIME}s")
    print(f"  复位等待: {RESET_WAIT}s")
    print(f"  靶标追踪: {TARGET_TRACK_TIME}s")
    print()

    # ==========================================
    # Step 1: 物料识别 (仅一次，在所有初始化之前)
    # ==========================================
    cam = CameraManager()
    cam.start()
    time.sleep(1.0)

    materials = detect_materials(cam)

    if not materials:
        print("物料识别失败，使用空物料列表继续...")

    # ==========================================
    # Step 2: 电机 & 控制器初始化
    # ==========================================
    ctrl = OpenLoopController(
        serial_port=SERIAL_PORT,
        serial_baud=SERIAL_BAUD,
        pan_range=(-8.0, 8.0),
        tilt_range=(-3.0, 11.0),
        num_calib_points=16,
        settle_time=SETTLE_TIME,
    )

    if not ctrl.setup():
        print("控制器初始化失败, 退出。")
        cam.release()
        return

    time.sleep(1.0)

    try:
        # ---- 靶标 & 激光 V2/V3 替换 (仅一次) ----
        tr_v2 = init_targets_and_laser(ctrl)

        # ---- 开环校准 (仅一次) ----
        print("\n" + "=" * 60)
        print("  开环校准 & 神经网络训练 (仅首次)")
        print("=" * 60)
        print("将进行均匀网格采样，请确保激光点可见...\n")

        ctrl.calibrate()

        if ctrl.calib_data:
            print("\n校准数据汇总:")
            print(f"{'摄像头坐标':<16s} {'电机角度':<22s}")
            print("-" * 38)
            for cx, cy, pan, tilt in ctrl.calib_data:
                print(f"({cx:4d},{cy:4d})        ({pan:+7.2f}°, {tilt:+7.2f}°)")

        # ==========================================
        # 循环执行任务
        # ==========================================
        round_num = 0
        while True:
            round_num += 1
            print("\n" + "#" * 60)
            print(f"#  第 {round_num} 轮任务")
            print("#" * 60)

            # ---- QR码识别 ----
            sequence, qr_data = read_qr_code(ctrl.cam)
            if not sequence:
                print("使用默认顺序: 红 -> 绿 -> 蓝")
                sequence = ['Red', 'Green', 'Blue']

            # ---- 执行序列 (物料使用预先识别的一次性结果) ----
            execute_sequence(ctrl, tr_v2, sequence, materials)

            # ---- 电机复位 (防丢步) ----
            print(f"\n>>> 电机复位，等待 {RESET_WAIT}s...")
            try:
                ctrl.mc.reset()
            except Exception as e:
                print(f"复位失败: {e}")
            time.sleep(RESET_WAIT)

            print("\n准备进入下一轮，请放置新二维码...")
            print("(按 Ctrl+C 可终止程序)\n")

    except KeyboardInterrupt:
        print("\n\n用户中断，正在退出...")
    finally:
        ctrl.shutdown()


if __name__ == "__main__":
    main()
