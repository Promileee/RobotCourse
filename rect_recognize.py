"""
矩形靶标识别模块（函数化版本）
基于HSV颜色空间的矩形靶标检测，结合深度信息验证检测结果。
支持识别红色、蓝色、绿色矩形靶标。

使用方法:
    from camera_manager import CameraManager
    import rect_recognize as rr

    cam = CameraManager()
    cam.start()
    frame = cam.read_rgb_frame()
    dpt = cam.get_depth_map()
    result, frame_rects = rr.detect_frame(frame, dpt)
"""

import cv2
import numpy as np
from camera_manager import CameraManager


# HSV 颜色范围
LOWER_RED1 = np.array([0, 100, 50])
UPPER_RED1 = np.array([15, 255, 255])
LOWER_RED2 = np.array([160, 100, 50])
UPPER_RED2 = np.array([180, 255, 255])

LOWER_BLUE = np.array([100, 120, 70])
UPPER_BLUE = np.array([130, 255, 255])

LOWER_GREEN = np.array([40, 70, 70])
UPPER_GREEN = np.array([80, 255, 255])

COLORS = {
    "Red": ((0, 0, 255), (LOWER_RED1, UPPER_RED1), (LOWER_RED2, UPPER_RED2)),
    "Blue": ((255, 0, 0), (LOWER_BLUE, UPPER_BLUE), None),
    "Green": ((0, 255, 0), (LOWER_GREEN, UPPER_GREEN), None),
}


# ==========================================
# 深度处理函数
# ==========================================

def create_depth_mask(dpt, depth_max=1300):
    """创建深度遮罩"""
    mask = np.where((dpt > 0) & (dpt < depth_max), 255, 0).astype(np.uint8)
    return mask


def get_depth_at(dpt, cx, cy):
    """获取指定像素位置的深度值"""
    if 0 <= cy < dpt.shape[0] and 0 <= cx < dpt.shape[1]:
        return dpt[cy, cx]
    return -1


def visualize_depth(dpt, depth_max=1300):
    """生成深度可视化图像"""
    vis = cv2.normalize(np.clip(dpt, 0, depth_max), None, 0, 255,
                        cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


# ==========================================
# 颜色识别函数
# ==========================================

def create_color_mask(hsv, color_name):
    """为指定颜色创建HSV遮罩"""
    _, range1, range2 = COLORS[color_name]
    mask = cv2.inRange(hsv, range1[0], range1[1])
    if range2 is not None:
        mask2 = cv2.inRange(hsv, range2[0], range2[1])
        mask = cv2.bitwise_or(mask, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def find_best_rect(color_mask, depth_mask, depth_ratio=0.2, min_area=500):
    """从颜色遮罩中找到满足深度验证的最佳矩形
    返回 (x, y, w, h, cx, cy) 或 None
    """
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if not (0.2 < aspect_ratio < 5.0):
            continue

        # 深度覆盖率验证
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, color_mask.shape[1]), min(y + h, color_mask.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue

        roi_color = color_mask[y1:y2, x1:x2]
        roi_depth = depth_mask[y1:y2, x1:x2]
        color_pixels = cv2.countNonZero(roi_color)
        if color_pixels == 0:
            continue
        in_range_pixels = cv2.countNonZero(cv2.bitwise_and(roi_color, roi_depth))
        if in_range_pixels / color_pixels < depth_ratio:
            continue

        if area > best_area:
            best_area = area
            cx, cy = x + w // 2, y + h // 2
            best_rect = (x, y, w, h, cx, cy)

    return best_rect


# ==========================================
# 主检测函数
# ==========================================

def detect_frame(frame, dpt, depth_max=1300, depth_ratio=0.2, min_area=500):
    """检测一帧中的所有颜色矩形靶标
    返回 (result_img, frame_rects)
    frame_rects = {"Red": (x,y,w,h,cx,cy,d_val) or None, ...}
    """
    depth_mask = create_depth_mask(dpt, depth_max)
    if depth_mask.shape[:2] != frame.shape[:2]:
        depth_mask = cv2.resize(depth_mask, (frame.shape[1], frame.shape[0]))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = frame.copy()
    frame_rects = {}

    for name, (color_bgr, range1, range2) in COLORS.items():
        color_mask = create_color_mask(hsv, name)

        best_rect = find_best_rect(color_mask, depth_mask, depth_ratio, min_area)

        if best_rect is not None:
            x, y, w, h, cx, cy = best_rect
            d_val = get_depth_at(dpt, cx, cy)

            cv2.rectangle(result, (x, y), (x + w, y + h), color_bgr, 2)
            label = f"{name} ({cx},{cy}) d={d_val:.0f}mm"
            cv2.putText(result, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

            frame_rects[name] = (x, y, w, h, cx, cy, d_val)
        else:
            frame_rects[name] = None

    return result, frame_rects


# ==========================================
# 稳定性检测函数
# ==========================================

def init_stability_state(stable_frames=3, skip_frames=5, position_tolerance=5):
    """初始化稳定性检测状态"""
    return {
        "stable_frames": stable_frames,
        "skip_frames": skip_frames,
        "position_tolerance": position_tolerance,
        "history": [],
        "frame_count": 0,
    }


def reset_stability(state):
    """重置稳定性检测状态"""
    state["history"].clear()
    state["frame_count"] = 0


def update_stability(state, frame_rects):
    """更新一帧数据，返回 (is_stable, corrected_rects_or_None)"""
    stable_frames = state["stable_frames"]
    skip_frames = state["skip_frames"]
    position_tolerance = state["position_tolerance"]

    state["frame_count"] += 1

    if state["frame_count"] <= skip_frames:
        return False, None

    # 所有颜色都检测到才算有效
    all_detected = all(v is not None for v in frame_rects.values())
    if not all_detected:
        state["history"].clear()
        return False, None

    state["history"].append(frame_rects)
    if len(state["history"]) > stable_frames:
        state["history"].pop(0)

    if len(state["history"]) < stable_frames:
        return False, None

    # 检查稳定性
    for i in range(1, stable_frames):
        prev = state["history"][i - 1]
        curr = state["history"][i]
        for name in frame_rects:
            for j in range(4):  # x, y, w, h
                if abs(prev[name][j] - curr[name][j]) > position_tolerance:
                    return False, None

    # 稳定：统一矩形尺寸为中位数
    final = state["history"][-1]
    ws = [rc[2] for rc in final.values()]
    hs = [rc[3] for rc in final.values()]
    med_w = int(np.median(ws))
    med_h = int(np.median(hs))

    corrected = {}
    for name, (x, y, w, h, cx, cy, d_val) in final.items():
        nx = cx - med_w // 2
        ny = cy - med_h // 2
        corrected[name] = (nx, ny, med_w, med_h, cx, cy, d_val)

    return True, corrected


# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    cam = CameraManager()
    cam.start()

    stab_state = init_stability_state(skip_frames=5, stable_frames=3)
    final_rects = None
    last_frame = None

    cv2.namedWindow("Result")
    cv2.createTrackbar("Min Area", "Result", 500, 5000, lambda _: None)

    while True:
        frame = cam.read_rgb_frame()
        if frame is None:
            print("Cannot read RGB frame.")
            break

        last_frame = frame.copy()
        dpt = cam.get_depth_map()
        min_area = cv2.getTrackbarPos("Min Area", "Result")

        result, frame_rects = detect_frame(frame, dpt, min_area=min_area)
        depth_vis = visualize_depth(dpt)

        cv2.imshow("Result", result)
        cv2.imshow("Depth", depth_vis)

        is_stable, corrected = update_stability(stab_state, frame_rects)
        if is_stable:
            final_rects = corrected
            print("\nStable detection achieved:")
            for name, rc in final_rects.items():
                print(f"  {name}: x={rc[0]}, y={rc[1]}, w={rc[2]}, h={rc[3]}")
            break

        if cv2.waitKey(1) == ord("q"):
            break

    cam.release()

    # 显示最终结果
    if final_rects is not None and last_frame is not None:
        import matplotlib.pyplot as plt

        fill_colors = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}
        vis = last_frame.copy()

        for name, rc in final_rects.items():
            x, y, w, h, cx, cy, d_val = rc
            cv2.rectangle(vis, (x, y), (x + w, y + h), fill_colors[name], -1)
            cv2.putText(vis, f"{name} ({x},{y}) {w}x{h}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, fill_colors[name], 2)

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(vis_rgb)
        plt.title("Detected Rectangles")
        plt.axis("off")
        plt.tight_layout()
        plt.show()