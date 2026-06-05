import cv2
import numpy as np
from openni import openni2

# ==========================================
# 1. 初始化与预热 (跳过前5帧)
# ==========================================
print("正在初始化相机...")
openni2.initialize()
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开RGB摄像头")
    exit()

print("正在跳过前5帧以稳定传感器...")
for _ in range(5):
    cap.read()
    depth_stream.read_frame()
    cv2.waitKey(100)

# 读取第 6 帧
ret, frame = cap.read()
depth_frame = depth_stream.read_frame()
print("获取第6帧成功，开始同心几何特征分析...")

# 解析深度图
dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
depth_map = (np.asarray(dframe_data[:, :, 0], dtype="float32") + 
             np.asarray(dframe_data[:, :, 1], dtype="float32") * 255)[:, ::-1]

# ==========================================
# 2. 边缘提取与“同心聚类”寻靶
# ==========================================
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(blurred, 40, 120)

# 获取所有轮廓（无需层级关系）
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

bboxes = []
# 第一步：把所有有意义的轮廓变成矩形框，并计算它们的中心点
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    # 宽泛过滤：忽略极小的噪点碎屑和占满全屏的超大轮廓
    if 100 < area < 40000:
        cx = x + w // 2
        cy = y + h // 2
        bboxes.append((cx, cy, x, y, w, h))

# 第二步：按中心点位置进行聚类（如果一堆框的中心点都在一起，说明遇到了同心图案）
clusters = []
DIST_THRESH = 15  # 中心点偏差允许值（像素），应对形状断裂导致的中心偏移

for box in bboxes:
    cx, cy = box[0], box[1]
    added = False
    for cluster in clusters:
        ref_cx, ref_cy = cluster[0][0], cluster[0][1]
        # 判断当前框的中心是否与该聚类的中心重合
        if abs(cx - ref_cx) < DIST_THRESH and abs(cy - ref_cy) < DIST_THRESH:
            cluster.append(box)
            added = True
            break
    if not added:
        clusters.append([box])

target_candidates = []
# 第三步：从聚类中筛选出真正的靶标
for cluster in clusters:
    # 核心特征 1：复杂的同心组合！靶标至少会产生 4-5 个以上的同心轮廓（方框+多圈圆弧+十字）
    if len(cluster) >= 4:
        # 计算这个同心聚类的总外围边界 (囊括最外层的正方形)
        min_x = min(b[2] for b in cluster)
        min_y = min(b[3] for b in cluster)
        max_r = max(b[2] + b[4] for b in cluster)
        max_b = max(b[3] + b[5] for b in cluster)
        
        tw, th = max_r - min_x, max_b - min_y
        aspect_ratio = float(tw) / th if th > 0 else 0
        
        # 核心特征 2：整体长宽比接近正方形
        if 0.75 < aspect_ratio < 1.3:
            tarea = tw * th
            # 核心特征 3：根据你说的“占比比较小”，锁定真实面积范围
            if 2000 < tarea < 25000:
                tcx, tcy = min_x + tw // 2, min_y + th // 2
                target_candidates.append((min_x, min_y, tw, th, tcx, tcy))

# 简单去重：防止重叠区域被多次计算
final_targets = []
target_candidates.sort(key=lambda b: b[2]*b[3], reverse=True)
for candidate in target_candidates:
    tcx, tcy = candidate[4], candidate[5]
    if not any(ft[0] < tcx < ft[0]+ft[2] and ft[1] < tcy < ft[1]+ft[3] for ft in final_targets):
        final_targets.append(candidate)

print(f"同心特征聚类完成，找到 {len(final_targets)} 个符合几何特征的目标。")

# ==========================================
# 3. 深度过滤与颜色匹配
# ==========================================
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
result_img = frame.copy()
valid_targets_count = 0

for x, y, w, h, cx, cy in final_targets:
    # --- 增加你要求的深度约束：< 1000mm ---
    depth_val = depth_map[cy, cx] if (0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]) else 0
    if not (0 < depth_val < 1000):
        print(f"丢弃目标 ({cx}, {cy})：深度 {depth_val}mm 超出 1000mm 范围。")
        continue

    # --- 颜色投票匹配 ---
    roi_hsv = hsv[y:y+h, x:x+w]
    mask_r1 = cv2.inRange(roi_hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    mask_r2 = cv2.inRange(roi_hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    mask_r = cv2.bitwise_or(mask_r1, mask_r2)
    mask_g = cv2.inRange(roi_hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
    mask_b = cv2.inRange(roi_hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    
    counts = {
        "Red Target": cv2.countNonZero(mask_r),
        "Green Target": cv2.countNonZero(mask_g),
        "Blue Target": cv2.countNonZero(mask_b)
    }
    best_color_name = max(counts, key=counts.get)
    max_pixels = counts[best_color_name]
    
    # 必须包含足够的彩色像素，排除误抓取的黑白图案
    if max_pixels > 150:
        valid_targets_count += 1
        color_bgr = (0, 0, 255) if "Red" in best_color_name else (0, 255, 0) if "Green" in best_color_name else (255, 0, 0)
        
        cv2.rectangle(result_img, (x, y), (x+w, y+h), color_bgr, 3)
        cv2.circle(result_img, (cx, cy), 5, (0, 255, 255), -1)
        label = f"{best_color_name} {depth_val:.0f}mm"
        cv2.putText(result_img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
        print(f"✅ 锁定靶标: {best_color_name}, 中心 ({cx}, {cy}), 深度 {depth_val}mm")

# ==========================================
# 4. 显示结果
# ==========================================
cap.release()
depth_stream.stop()
dev.close()

# 将原始图像和识别结果拼接对比
display_img = np.hstack([frame, result_img])
display_img = cv2.resize(display_img, (display_img.shape[1]//2, display_img.shape[0]//2))

print(f"\n全部处理完毕，最终确认 {valid_targets_count} 个靶标。按任意键关闭窗口。")
cv2.imshow("Original (Left) vs Detected (Right)", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()