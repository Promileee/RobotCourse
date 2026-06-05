import cv2
import numpy as np
import math
from openni import openni2

# ==========================================
# 严格的几何约束阈值 (像素)
# ==========================================
TOL_SQUARE = 15       # 靶标长宽差最大允许值
TOL_SIZE = 15         # 三个靶标最大和最小尺寸差值
TOL_TRIANGLE = 20     # 等边三角形三边最大差值
MASK_SCALE = 1.25     # 外接圆放缩比例 (用于包住整个靶标以提取特征)

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
confirmed_targets = None
ref_frame = None
frame_count = 0

# ==========================================
# PHASE 1: 严苛的初始化捕获
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    depth_frame = depth_stream.read_frame()
    
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
            if 0.7 < (float(tw)/th if th>0 else 0) < 1.4 and 2000 < tw * th < 25000:
                tcx, tcy = min_x + tw // 2, min_y + th // 2
                target_candidates.append((min_x, min_y, tw, th, tcx, tcy))

    final_targets = []
    target_candidates.sort(key=lambda b: b[2]*b[3], reverse=True)
    for cand in target_candidates:
        tcx, tcy = cand[4], cand[5]
        if not any(ft[0] < tcx < ft[0]+ft[2] and ft[1] < tcy < ft[1]+ft[3] for ft in final_targets):
            final_targets.append(cand)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    identified = {}
    
    for x, y, w, h, cx, cy in final_targets:
        # 防止越界
        y1, y2 = max(0, y), min(frame.shape[0], y+h)
        x1, x2 = max(0, x), min(frame.shape[1], x+w)
        roi_hsv = hsv[y1:y2, x1:x2]
        if roi_hsv.size == 0: continue

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

    preview_img = frame.copy()
    cv2.putText(preview_img, f"Searching... Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Initialization", preview_img)
    cv2.waitKey(1) 

    # --- 严苛校验 ---
    if len(identified) != 3 or not all(c in identified for c in ["Red", "Green", "Blue"]):
        continue 

    tr, tg, tb = identified["Red"], identified["Green"], identified["Blue"]
    
    max_sq_diff = max(abs(tr[2]-tr[3]), abs(tg[2]-tg[3]), abs(tb[2]-tb[3]))
    if max_sq_diff > TOL_SQUARE: continue

    ws, hs = [tr[2], tg[2], tb[2]], [tr[3], tg[3], tb[3]]
    max_size_diff = max(max(ws)-min(ws), max(hs)-min(hs))
    if max_size_diff > TOL_SIZE: continue

    d_rg = get_distance((tr[4], tr[5]), (tg[4], tg[5]))
    d_gb = get_distance((tg[4], tg[5]), (tb[4], tb[5]))
    d_br = get_distance((tb[4], tb[5]), (tr[4], tr[5]))
    max_tri_diff = max(d_rg, d_gb, d_br) - min(d_rg, d_gb, d_br)
    if max_tri_diff > TOL_TRIANGLE: continue

    print("\n✅✅✅ 所有严苛几何条件均已满足！初始化捕获成功！ ✅✅✅")
    confirmed_targets = identified
    ref_frame = frame.copy()
    break  # 跳出初始化循环

cv2.destroyWindow("Initialization")

# ==========================================
# PHASE 2: 创建 ORB 星座模板
# ==========================================
print("正在构建星座模板与特征库...")
orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 提取三个靶标中心点，计算最小外接圆
pts = np.array([[data[4], data[5]] for _, data in confirmed_targets.items()], dtype=np.float32)
(center_x, center_y), strict_radius = cv2.minEnclosingCircle(pts)
ref_mask_center = (int(center_x), int(center_y))
ref_mask_radius = int(strict_radius * MASK_SCALE)

# 生成模板遮罩
mask = np.zeros(ref_frame.shape[:2], dtype=np.uint8)
cv2.circle(mask, ref_mask_center, ref_mask_radius, 255, -1)

# 提取参考特征
ref_kp, ref_des = orb.detectAndCompute(ref_frame, mask)
ref_targets_centers = {color: (data[4], data[5]) for color, data in confirmed_targets.items()}
print(f"模板构建完毕，共提取 {len(ref_kp)} 个特征点。进入实时追踪！\n")

# ==========================================
# PHASE 3: ORB 实时刚体追踪
# ==========================================
colors_bgr = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}
current_targets = ref_targets_centers.copy()
current_mask_center = ref_mask_center

while True:
    ret, frame = cap.read()
    if not ret: break
    
    depth_frame = depth_stream.read_frame()
    dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
    depth_map = (np.asarray(dframe_data[:, :, 0], dtype="float32") + 
                 np.asarray(dframe_data[:, :, 1], dtype="float32") * 255)[:, ::-1]
                 
    result_img = frame.copy()
    
    # 提取当前帧特征
    kp, des = orb.detectAndCompute(frame, None)
    
    tracking_success = False
    if des is not None and len(des) > 10:
        matches = matcher.match(ref_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:max(10, int(len(matches) * 0.2))]
        
        if len(good_matches) >= 10:
            src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # 计算包含旋转、平移的仿射矩阵
            M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC)
            
            if M is not None:
                tracking_success = True
                # 用算出的矩阵更新三个坐标
                for color, (orig_cx, orig_cy) in ref_targets_centers.items():
                    new_x = M[0, 0] * orig_cx + M[0, 1] * orig_cy + M[0, 2]
                    new_y = M[1, 0] * orig_cx + M[1, 1] * orig_cy + M[1, 2]
                    current_targets[color] = (int(new_x), int(new_y))
                
                # 更新星座外接大圆的中心
                new_mc_x = M[0, 0] * ref_mask_center[0] + M[0, 1] * ref_mask_center[1] + M[0, 2]
                new_mc_y = M[1, 0] * ref_mask_center[0] + M[1, 1] * ref_mask_center[1] + M[1, 2]
                current_mask_center = (int(new_mc_x), int(new_mc_y))

    if tracking_success:
        cv2.putText(result_img, "TRACKING: ORB Constellation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        pts = []
        for color, (cx, cy) in current_targets.items():
            bgr = colors_bgr[color]
            pts.append((cx, cy))
            cv2.circle(result_img, (cx, cy), 15, bgr, 3)
            cv2.circle(result_img, (cx, cy), 4, (255, 255, 255), -1)
            
            d_val = depth_map[cy, cx] if (0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]) else 0
            cv2.putText(result_img, f"{color} {d_val:.0f}mm", (cx-30, cy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
            
        # 绘制整个星座组的外围界限
        cv2.circle(result_img, current_mask_center, ref_mask_radius, (0, 255, 255), 1, cv2.LINE_AA)
        
        # 连线构成三角形
        if len(pts) == 3:
            cv2.polylines(result_img, [np.array(pts)], isClosed=True, color=(255, 255, 255), thickness=1)
    else:
        cv2.putText(result_img, "TRACKING LOST! (Keep still or return to frame)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # 如果跟丢了，画面上的框会停在最后一帧的位置

    cv2.imshow("Constellation Tracking", result_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
depth_stream.stop()
dev.close()
cv2.destroyAllWindows()