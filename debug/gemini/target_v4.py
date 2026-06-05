import cv2
import numpy as np
import math
from openni import openni2

# ==========================================
# 放宽后的约束阈值
# ==========================================
MIN_CIRCULARITY = 0.65  # 圆形度阈值 (完美圆=1.0，0.65容忍了一定程度的椭圆和遮挡)
MAX_TARGET_DIST = 400   # 三个靶标之间的最大允许距离(像素)，仅用于防止把背景里的杂物误认
MASK_SCALE = 1.25       # 外接圆放缩比例 (用于包住整个靶标以提取特征)

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

print("\n开始执行基于圆形特征的初始帧捕获逻辑...")
confirmed_targets = None
ref_frame = None
frame_count = 0

# ==========================================
# PHASE 1: 放宽的圆形初始化捕获
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    depth_frame = depth_stream.read_frame()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 稍微加大模糊，有助于平滑边缘
    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    target_candidates = []
    
    # 1. 遍历轮廓，寻找“圆形”特征
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 放宽面积限制，剔除太小或太大的噪点
        if 200 < area < 40000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            
            # 计算圆形度 (Circularity)
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            
            if circularity > MIN_CIRCULARITY:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                target_candidates.append((int(x), int(y), int(w), int(h), int(cx), int(cy), int(radius), area))

    # 2. 候选目标去重 (防止同一个圆产生多个相近的轮廓)
    final_targets = []
    target_candidates.sort(key=lambda b: b[6], reverse=True) # 按半径降序
    for cand in target_candidates:
        tcx, tcy = cand[4], cand[5]
        # 如果当前圆心附近没有其他已记录的圆，则加入
        if not any(get_distance((tcx, tcy), (ft[4], ft[5])) < 20 for ft in final_targets):
            final_targets.append(cand)

    # 3. 颜色识别
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    identified = {}
    
    for x, y, w, h, cx, cy, r, area in final_targets:
        # 提取外接矩形ROI，防止越界
        y1, y2 = max(0, y), min(frame.shape[0], y+h)
        x1, x2 = max(0, x), min(frame.shape[1], x+w)
        roi_hsv = hsv[y1:y2, x1:x2]
        if roi_hsv.size == 0: continue

        # 颜色阈值
        mask_r = cv2.bitwise_or(
            cv2.inRange(roi_hsv, np.array([0, 50, 50]), np.array([15, 255, 255])),
            cv2.inRange(roi_hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
        )
        mask_g = cv2.inRange(roi_hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
        mask_b = cv2.inRange(roi_hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        
        counts = {"Red": cv2.countNonZero(mask_r), "Green": cv2.countNonZero(mask_g), "Blue": cv2.countNonZero(mask_b)}
        best_color = max(counts, key=counts.get)
        
        # 只要该颜色的像素占据了整个轮廓面积的一小部分（例如20%），就认为颜色匹配
        if counts[best_color] > (area * 0.2):
            identified[best_color] = (x, y, w, h, cx, cy)

    # 预览界面绘制
    preview_img = frame.copy()
    cv2.putText(preview_img, f"Searching Circles... Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 画出找到的候选圆（用于调试和确认）
    for cand in final_targets:
        cv2.circle(preview_img, (cand[4], cand[5]), cand[6], (255, 0, 255), 2)

    cv2.imshow("Initialization", preview_img)
    cv2.waitKey(1) 

    # 4. 宽松的最终校验：集齐三色，且距离不过远
    if len(identified) == 3 and all(c in identified for c in ["Red", "Green", "Blue"]):
        tr, tg, tb = identified["Red"], identified["Green"], identified["Blue"]
        
        d_rg = get_distance((tr[4], tr[5]), (tg[4], tg[5]))
        d_gb = get_distance((tg[4], tg[5]), (tb[4], tb[5]))
        d_br = get_distance((tb[4], tb[5]), (tr[4], tr[5]))
        
        # 只要三个圆没有分散在画面的天涯海角（判定为同一个靶标板）
        if max(d_rg, d_gb, d_br) < MAX_TARGET_DIST:
            print("\n✅✅✅ 发现红绿蓝圆形靶标！初始化捕获成功！ ✅✅✅")
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
# PHASE 3: ORB 实时刚体追踪 (保持不变)
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

    cv2.imshow("Constellation Tracking", result_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
depth_stream.stop()
dev.close()
cv2.destroyAllWindows()