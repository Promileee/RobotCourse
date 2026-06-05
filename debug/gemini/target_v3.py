import cv2
import numpy as np
import math
from openni import openni2

# ==========================================
# 严格的几何约束阈值 (像素)
# 建议初始测试设定在 10-15 左右，你可以修改为 2 尝试挑战极限
# ==========================================
TOL_SQUARE = 12       # 靶标长宽差最大允许值
TOL_SIZE = 12         # 三个靶标最大和最小尺寸差值
TOL_TRIANGLE = 15     # 等边三角形三边最大差值

def get_distance(p1, p2):
    """计算两点之间的欧式距离"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ==========================================
# 1. 初始化相机
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

print("\n开始执行严苛的初始帧捕获逻辑...")
print(f"当前阈值设定: 正方形容差≤{TOL_SQUARE}px, 尺寸一致性容差≤{TOL_SIZE}px, 等边三角形容差≤{TOL_TRIANGLE}px")

confirmed_targets = None
frame_count = 0

# ==========================================
# 2. 严苛的 While 循环
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    depth_frame = depth_stream.read_frame()
    
    # --- 1. 同心聚类特征提取 (与之前一致) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 100 < w * h < 40000:
            bboxes.append((x + w // 2, y + h // 2, x, y, w, h))

    clusters = []
    DIST_THRESH = 15
    for box in bboxes:
        added = False
        for cluster in clusters:
            if abs(box[0] - cluster[0][0]) < DIST_THRESH and abs(box[1] - cluster[0][1]) < DIST_THRESH:
                cluster.append(box)
                added = True
                break
        if not added:
            clusters.append([box])

    target_candidates = []
    for cluster in clusters:
        if len(cluster) >= 4:
            min_x = min(b[2] for b in cluster)
            min_y = min(b[3] for b in cluster)
            max_r = max(b[2] + b[4] for b in cluster)
            max_b = max(b[3] + b[5] for b in cluster)
            tw, th = max_r - min_x, max_b - min_y
            # 这里的长宽比只做初步过滤，严格检查在后面
            if 0.7 < (float(tw)/th if th>0 else 0) < 1.4 and 2000 < tw * th < 25000:
                tcx, tcy = min_x + tw // 2, min_y + th // 2
                target_candidates.append((min_x, min_y, tw, th, tcx, tcy))

    # 去重
    final_targets = []
    target_candidates.sort(key=lambda b: b[2]*b[3], reverse=True)
    for cand in target_candidates:
        tcx, tcy = cand[4], cand[5]
        if not any(ft[0] < tcx < ft[0]+ft[2] and ft[1] < tcy < ft[1]+ft[3] for ft in final_targets):
            final_targets.append(cand)

    # --- 2. 颜色身份验证 ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    identified = {}
    
    for x, y, w, h, cx, cy in final_targets:
        roi_hsv = hsv[y:y+h, x:x+w]
        mask_r = cv2.bitwise_or(
            cv2.inRange(roi_hsv, np.array([0, 50, 50]), np.array([15, 255, 255])),
            cv2.inRange(roi_hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
        )
        mask_g = cv2.inRange(roi_hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
        mask_b = cv2.inRange(roi_hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        
        counts = {"Red": cv2.countNonZero(mask_r), "Green": cv2.countNonZero(mask_g), "Blue": cv2.countNonZero(mask_b)}
        best_color = max(counts, key=counts.get)
        
        if counts[best_color] > 100:
            identified[best_color] = (x, y, w, h, cx, cy)

    # 显示实时搜寻画面
    preview_img = frame.copy()
    cv2.putText(preview_img, f"Searching... Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Initialization Preview", preview_img)
    cv2.waitKey(1) # 必须有，否则画面不刷新

    # --- 3. 核心：严苛的条件校验 ---
    
    # 条件一：红蓝绿必须全部凑齐
    if len(identified) != 3 or not all(c in identified for c in ["Red", "Green", "Blue"]):
        continue # 凑不齐，直接看下一帧

    # 提取数据
    tr = identified["Red"]
    tg = identified["Green"]
    tb = identified["Blue"]
    
    # 打印找到了3个目标，开始校验
    print(f"\n[帧 {frame_count}] 提取到 R,G,B 三个目标，开始几何校验:")

    # 条件二：三个靶标都接近正方形
    r_sq_diff = abs(tr[2] - tr[3])
    g_sq_diff = abs(tg[2] - tg[3])
    b_sq_diff = abs(tb[2] - tb[3])
    max_sq_diff = max(r_sq_diff, g_sq_diff, b_sq_diff)
    print(f"  -> 正方形校验: 最大长宽差 {max_sq_diff}px")
    if max_sq_diff > TOL_SQUARE:
        print("  ❌ 失败：未满足正方形条件。")
        continue

    # 条件三：三个靶标大小相近 (宽和高的最大极差)
    ws = [tr[2], tg[2], tb[2]]
    hs = [tr[3], tg[3], tb[3]]
    max_w_diff = max(ws) - min(ws)
    max_h_diff = max(hs) - min(hs)
    max_size_diff = max(max_w_diff, max_h_diff)
    print(f"  -> 尺寸一致性校验: 最大尺寸极差 {max_size_diff}px")
    if max_size_diff > TOL_SIZE:
        print("  ❌ 失败：未满足大小一致条件。")
        continue

    # 条件四：等边三角形校验
    d_rg = get_distance((tr[4], tr[5]), (tg[4], tg[5]))
    d_gb = get_distance((tg[4], tg[5]), (tb[4], tb[5]))
    d_br = get_distance((tb[4], tb[5]), (tr[4], tr[5]))
    
    tri_edges = [d_rg, d_gb, d_br]
    max_tri_diff = max(tri_edges) - min(tri_edges)
    print(f"  -> 等边三角形校验: 三边长度 {d_rg:.1f}, {d_gb:.1f}, {d_br:.1f}")
    print(f"  -> 边长最大极差 {max_tri_diff:.1f}px")
    if max_tri_diff > TOL_TRIANGLE:
        print("  ❌ 失败：未满足等边三角形条件。")
        continue

    # ==========================================
    # 4. 全部条件满足，跳出循环！
    # ==========================================
    print("\n✅✅✅ 所有严苛几何条件均已满足！初始化捕获成功！ ✅✅✅")
    confirmed_targets = identified
    
    # 记录该帧的深度信息以便后续使用
    dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
    final_depth_map = (np.asarray(dframe_data[:, :, 0], dtype="float32") + 
                       np.asarray(dframe_data[:, :, 1], dtype="float32") * 255)[:, ::-1]
    final_frame = frame.copy()
    break

# ==========================================
# 5. 释放并展示完美捕获的结果
# ==========================================
cap.release()
depth_stream.stop()
dev.close()
cv2.destroyAllWindows()

# 在这完美的一帧上绘制结果
colors_bgr = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}
pts = []

for color, data in confirmed_targets.items():
    x, y, w, h, cx, cy = data
    bgr = colors_bgr[color]
    pts.append((cx, cy))
    
    # 画框和中心点
    cv2.rectangle(final_frame, (x, y), (x+w, y+h), bgr, 3)
    cv2.circle(final_frame, (cx, cy), 5, (255, 255, 255), -1)
    
    d_val = final_depth_map[cy, cx] if (0 <= cy < final_depth_map.shape[0] and 0 <= cx < final_depth_map.shape[1]) else 0
    cv2.putText(final_frame, f"{color} {w}x{h}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

# 画出那个等边三角形
cv2.line(final_frame, pts[0], pts[1], (0, 255, 255), 2)
cv2.line(final_frame, pts[1], pts[2], (0, 255, 255), 2)
cv2.line(final_frame, pts[2], pts[0], (0, 255, 255), 2)

cv2.putText(final_frame, "PERFECT INIT FRAME", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

print("按任意键退出展示...")
cv2.imshow("Perfect Initialization", final_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()