"""
mission_open_loop.py
完整流程——开环版本

基于 mission.md，串联所有已有模块完成整个作业流程:
  1. QR码识别 (QR_code_recognize_v1) → 读取颜色顺序
  2. 圆柱物料识别 (rect_recognize) → 获取各颜色物料中心坐标
  3. 靶标 & 激光识别 (target_recognize_v2 + laser_recognize_v3)
  4. 初始化开环控制，采样并训练神经网络
  5. 按QR码顺序执行指向：物料1→靶标1→物料2→靶标2→物料3→靶标3

使用方法:
    python mission_open_loop.py
    按提示操作，按 q 可随时退出
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
QR_TIMEOUT = 20.0      # QR码识别超时(秒)
MATERIAL_TIMEOUT = 15.0  # 物料检测超时(秒)

# 颜色映射: 二维码数字 → 颜色名
DIGIT_TO_COLOR = {'1': 'Red', '2': 'Green', '3': 'Blue'}
COLOR_CN = {'Red': '红', 'Green': '绿', 'Blue': '蓝'}


# ==========================================
# Phase 1: QR码识别
# ==========================================
def phase1_read_qr(cam):
    """识别二维码，读取颜色顺序

    Returns:
        sequence: ['Red', 'Green', 'Blue'] 或 None
        raw_data: 原始二维码字符串
    """
    QR_WAIT_AFTER = 5.0   # 识别成功后等待，确保二维码及人手撤出视野

    qr = QRCodeRecognizer()
    print("\n" + "=" * 60)
    print("  Phase 1: QR码识别 (QR_code_recognize_v1)")
    print("  QR码数字→颜色: 1=红, 2=绿, 3=蓝")
    print("=" * 60)
    print("请将二维码对准摄像头，识别成功后将自动确认...")
    print(f"按 'q' 跳过\n")

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
            colors = [DIGIT_TO_COLOR.get(ch, '?') for ch in data]
            label = f"QR: {data} -> {' -> '.join(COLOR_CN.get(c, c) for c in colors)}"
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
        # 保持窗口显示倒计时
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
# Phase 2: 圆柱物料识别
# ==========================================
def phase2_detect_materials(cam):
    """识别圆柱物料(矩形靶标)，返回稳定检测后的各颜色中心坐标

    Returns:
        materials: {'Red': (cx, cy), 'Green': (cx, cy), 'Blue': (cx, cy)} 或 {}
    """
    print("\n" + "=" * 60)
    print("  Phase 2: 圆柱物料识别 (rect_recognize)")
    print("=" * 60)
    print("请确保红/绿/蓝三种圆柱物料在摄像头视野内...")

    stab = rr.init_stability_state(skip_frames=5, stable_frames=3)
    cv2.namedWindow("Material Detection")

    materials = {}
    start_time = time.time()

    while time.time() - start_time < MATERIAL_TIMEOUT:
        frame = cam.read_rgb_frame()
        if frame is None:
            continue

        dpt = cam.get_depth_map()
        result, frame_rects = rr.detect_frame(frame, dpt)

        is_stable, corrected = rr.update_stability(stab, frame_rects)
        if is_stable:
            for color, rc in corrected.items():
                if rc is not None:
                    x, y, w, h, cx, cy, d_val = rc
                    materials[color] = (cx, cy)
                    print(f"  {COLOR_CN.get(color, color)}物料: "
                          f"中心({cx}, {cy}), 深度={d_val:.0f}mm, 尺寸={w}x{h}")
            print("物料检测稳定!\n")
            break

        # 在图上显示当前检测状态
        for name, rc in frame_rects.items():
            if rc is not None:
                x, y, w, h, cx, cy, d_val = rc
                cv2.putText(result, f"{name}:({cx},{cy})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(result, "Waiting for stable detection...",
                    (10, result.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Material Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("Material Detection")

    if not materials:
        print("警告: 物料检测未完成")
    return materials


# ==========================================
# Phase 5: 执行任务序列
# ==========================================
def phase5_execute_sequence(ctrl, tr, sequence, materials):
    """按QR码顺序执行指向：每个颜色先指物料再指靶标

    Args:
        ctrl: OpenLoopController (已标定)
        tr: TargetRecognizerV2 (已初始化)
        sequence: ['Red', 'Green', 'Blue'] 颜色顺序
        materials: {'Red': (cx, cy), ...} 物料坐标
    """
    print("\n" + "=" * 60)
    print("  Phase 5: 执行任务序列")
    print("=" * 60)

    color_order = " -> ".join(f"{COLOR_CN.get(c, c)}物料→{COLOR_CN.get(c, c)}靶标"
                              for c in sequence)
    print(f"执行顺序: {color_order}\n")

    cv2.namedWindow("Mission Execution")

    for i, color in enumerate(sequence):
        step = i + 1
        cn = COLOR_CN.get(color, color)
        print(f"--- 第 {step}/{len(sequence)} 步: {cn} ---")

        # ==========================================
        # 5a. 指向该颜色的物料
        # ==========================================
        if materials and color in materials:
            mat_cx, mat_cy = materials[color]
            print(f"  指向{cn}物料 ({mat_cx}, {mat_cy})")
            try:
                pan, tilt = ctrl.point_to(mat_cx, mat_cy)
                print(f"  -> 电机: pan={pan:+.2f}°, tilt={tilt:+.2f}°")
            except Exception as e:
                print(f"  ! 指向物料失败: {e}")
        else:
            print(f"  ! 跳过: {cn}物料坐标未知")

        print(f"  等待{cn}物料稳定...")
        time.sleep(SETTLE_TIME)

        # ==========================================
        # 5b. 指向该颜色的靶标
        # ==========================================
        frame = ctrl.cam.read_rgb_frame()
        tgt_hit = False

        if frame is not None:
            dpt = ctrl.cam.get_depth_map()
            tr.depth_map = dpt
            _, binary = tr.preprocess(frame)
            track_ok, all_targets, tr_result, _, _ = tr.track_frame(frame, binary)

            if track_ok and all_targets and color in all_targets:
                tgt_cx, tgt_cy = all_targets[color]
                print(f"  指向{cn}靶标 ({tgt_cx}, {tgt_cy})")
                try:
                    pan, tilt = ctrl.point_to(tgt_cx, tgt_cy)
                    print(f"  -> 电机: pan={pan:+.2f}°, tilt={tilt:+.2f}°")
                    tgt_hit = True
                except Exception as e:
                    print(f"  ! 指向靶标失败: {e}")

                cv2.imshow("Mission Execution", tr_result)

            # 回退：使用参考靶标位置
            if not tgt_hit and tr.ref_targets_centers and color in tr.ref_targets_centers:
                tgt_cx, tgt_cy = tr.ref_targets_centers[color]
                print(f"  指向{cn}靶标(参考) ({tgt_cx}, {tgt_cy})")
                try:
                    pan, tilt = ctrl.point_to(tgt_cx, tgt_cy)
                    print(f"  -> 电机: pan={pan:+.2f}°, tilt={tilt:+.2f}°")
                except Exception as e:
                    print(f"  ! 指向靶标失败: {e}")

                # 简单可视化
                disp = frame.copy()
                cv2.circle(disp, (tgt_cx, tgt_cy), 15, (0, 255, 255), 2)
                cv2.putText(disp, f"Target {cn} (ref)", (tgt_cx + 20, tgt_cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow("Mission Execution", disp)

            if not tgt_hit and not (tr.ref_targets_centers and color in tr.ref_targets_centers):
                print(f"  ! 跳过: {cn}靶标未找到且无参考位置")
        else:
            print(f"  ! 摄像头帧读取失败")

        print(f"  等待{cn}靶标稳定...")
        time.sleep(SETTLE_TIME)
        cv2.waitKey(500)

    cv2.destroyWindow("Mission Execution")
    print("\n" + "=" * 60)
    print("  任务序列执行完毕!")
    print("=" * 60)


# ==========================================
# 主函数
# ==========================================
def main():
    print("=" * 60)
    print("  完整流程 —— 开环版本")
    print("  基于 mission.md")
    print("=" * 60)
    print(f"  电机串口: {SERIAL_PORT}")
    print(f"  稳定等待: {SETTLE_TIME}s")
    print()

    # ==========================================
    # 初始化开环控制器
    # ==========================================
    # OpenLoopController.setup() 内部会:
    #   1. 启动相机
    #   2. 连接电机
    #   3. 初始化靶标识别 (target_recognize_v1)
    #   4. 初始化激光识别 (laser_recognize_v3)
    ctrl = OpenLoopController(
        serial_port=SERIAL_PORT,
        serial_baud=SERIAL_BAUD,
        pan_range=(1.0, 18.0),
        tilt_range=(-3.0, 11.0),
        num_calib_points=16,
        settle_time=SETTLE_TIME,
    )

    if not ctrl.setup():
        print("控制器初始化失败, 退出。")
        return

    time.sleep(1.0)

    try:
        # ==========================================
        # Phase 1: QR码识别 → 颜色顺序
        # ==========================================
        sequence, qr_data = phase1_read_qr(ctrl.cam)
        if not sequence:
            print("\n使用默认顺序: 红 -> 绿 -> 蓝")
            sequence = ['Red', 'Green', 'Blue']

        # ==========================================
        # Phase 2: 圆柱物料识别
        # ==========================================
        materials = phase2_detect_materials(ctrl.cam)

        # ==========================================
        # Phase 3: 靶标 & 激光识别
        # ==========================================
        # 使用 target_recognize_v2 替换控制器中默认的 V1
        print("=" * 60)
        print("  Phase 3: 靶标 & 激光识别")
        print("  (target_recognize_v2 + laser_recognize_v3)")
        print("=" * 60)

        tr_v2 = TargetRecognizerV2()
        tr_v2.setup(ctrl.cam)

        # 将 V1 初始化得到的参考靶标数据传递给 V2
        tr_v2.ref_targets_centers = dict(ctrl.tr.ref_targets_centers)
        tr_v2.ref_mask_center = ctrl.tr.ref_mask_center
        tr_v2.ref_mask_radius = ctrl.tr.ref_mask_radius

        # 用当前帧重建 V2 的参考二值图
        frame = ctrl.cam.read_rgb_frame()
        if frame is not None:
            gray, binary = tr_v2.preprocess(frame)
            tr_v2.ref_binary = binary

            # 用参考靶标位置构建ORB模板
            # build_template 需要 (x, y, w, h, cx, cy) 格式，实际只用后两个
            confirmed = {}
            for color, (cx, cy) in tr_v2.ref_targets_centers.items():
                confirmed[color] = (0, 0, 1, 1, cx, cy)
            tr_v2.build_template(frame, confirmed)

        # V3 激光识别器 (替换控制器中默认的 V3 实例以确保参数一致)
        lr_v3 = LaserRecognizerV3(
            thresh_val=225, blur_size=3, depth_max=1300,
        )
        lr_v3.setup(ctrl.cam)
        dpt = ctrl.cam.get_depth_map()
        lr_v3.init_from_target_data(ctrl.tr.ref_targets_centers, dpt)

        # 将控制器的识别器替换为 V2/V3
        ctrl.tr = tr_v2
        ctrl.lr = lr_v3

        print(f"激光ROI: 圆心({lr_v3.roi_center[0]}, {lr_v3.roi_center[1]}), "
              f"半径={lr_v3.roi_radius}px")
        for color, (cx, cy) in tr_v2.ref_targets_centers.items():
            print(f"  {COLOR_CN.get(color, color)}靶标参考: ({cx}, {cy})")

        # ==========================================
        # Phase 4: 开环校准 & NN训练
        # ==========================================
        print("\n" + "=" * 60)
        print("  Phase 4: 开环校准 & 神经网络训练")
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
        # Phase 5: 执行任务序列
        # ==========================================
        phase5_execute_sequence(ctrl, tr_v2, sequence, materials)

    finally:
        ctrl.shutdown()


if __name__ == "__main__":
    main()
