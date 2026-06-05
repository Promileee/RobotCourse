import cv2
import numpy as np
import math
from openni import openni2

# ==========================================
# 几何与容错参数
# ==========================================
TOL_ISOSCELES = 30     # 等腰三角形“两条腰”允许的最大长度差（像素）
MIN_UNIQUE_DIFF = 15   # “底”和“腰”必须有明显的长度区分（确保是“品”字而不是等边三角形）
PATIENCE = 15          # 容错帧数：跟丢后保持多少帧才退回初始化

def get_constellation_roles(pts):
    """
    【核心几何引擎】：输入3个点，无视旋转，利用“品”字结构识别出它们的物理身份
    返回字典：{'Apex': 点坐标, 'Base_Pos': 点坐标, 'Base_Neg': 点坐标}
    """
    p0, p1, p2 = pts
    d01 = math.hypot(p0[0]-p1[0], p0[1]-p1[1])
    d12 = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
    d20 = math.hypot(p2[0]-p0[0], p2[1]-p0[1])

    # 记录边长，以及它“对面的点”（顶点），和组成它的两个点
    edges = [
        (d12, p0, p1, p2), 
        (d20, p1, p0, p2),
        (d01, p2, p0, p1)
    ]

    # 找出长度最接近的两条边（等腰三角形的两条腰）
    diff01 = abs(edges[0][0] - edges[1][0])
    diff12 = abs(edges[1][0] - edges[2][0])
    diff20 = abs(edges[2][0] - edges[0][0])

    min_diff = min(diff01, diff12, diff20)

    if min_diff == diff01:
        eq1, eq2, unique = edges[0], edges[1], edges[2]
    elif min_diff == diff12:
        eq1, eq2, unique = edges[1], edges[2], edges[0]
    else:
        eq1, eq2, unique = edges[2], edges[0], edges[1]

    # 校验 1：这两条“腰”的长度必须差不多
    if abs(eq1[0] - eq2[0]) > TOL_ISOSCELES:
        return None
        
    # 校验 2：“底”的长度必须明显不同于“腰”（品字结构确认）
    if abs(unique[0] - eq1[0]) < MIN_UNIQUE_DIFF:
        return None

    # “独特边”对面的点，就是“品”字的头 (Apex)
    apex = unique[1] 
    base1 = unique[2]
    base2 = unique[3]

    # 使用向量叉乘，区分左腿和右腿 (保证旋转时的左右一致性)
    axis_x = (base1[0] + base2[0])/2.0 - apex[0]
    axis_y = (base1[1] + base2[1])/2.0 - apex[1]
    v1_x = base1[0] - apex[0]
    v1_y = base1[1] - apex[1]
    cross = axis_x * v1_y - axis_y * v1_x

    if cross > 0:
        base_pos, base_neg = base1, base2
    else:
        base_pos, base_neg = base2, base1

    return {'Apex': apex, 'Base_Pos': base_pos, 'Base_Neg': base_neg}

class BinaryTopologyTracker:
    def __init__(self):
        self.state = "INIT"
        self.role_to_color = {} # 记忆映射：例如 {'Apex': 'Red', 'Base_Pos': 'Blue', ...}
        self.current_targets = {}
        self.missed_frames = 0

    def get_binary_centers(self, frame):
        """提取二值图并获取同心特征中心"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用自适应阈值，生成和你截图一样完美的黑白二值图
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 50 < w * h < 40000:
                bboxes.append((x + w // 2, y + h // 2))
                
        clusters = []
        for cx, cy in bboxes:
            added = False
            for cluster in clusters:
                if abs(cx - cluster[0][0]) < 15 and abs(cy - cluster[0][1]) < 15:
                    cluster.append((cx, cy))
                    added = True; break
            if not added: clusters.append([(cx, cy)])

        centers = []
        for cluster in clusters:
            if len(cluster) >= 3: # 找到至少嵌套了 3 层的同心结构
                avg_cx = int(np.mean([p[0] for p in cluster]))
                avg_cy = int(np.mean([p[1] for p in cluster]))
                if not any(math.hypot(avg_cx - f[0], avg_cy - f[1]) < 30 for f in centers):
                    centers.append((avg_cx, avg_cy))
                    
        return centers, binary

    def process_frame(self, frame, hsv_frame):
        centers, binary_img = self.get_binary_centers(frame)

        if self.state == "INIT":
            if len(centers) == 3:
                roles = get_constellation_roles(centers)
                if roles:
                    identified = {}
                    # 仅在此时看一眼颜色
                    for role_name, (cx, cy) in roles.items():
                        roi = hsv_frame[max(0, cy-15):min(frame.shape[0], cy+15), 
                                        max(0, cx-15):min(frame.shape[1], cx+15)]
                        if roi.size == 0: continue
                        
                        mask_r = cv2.bitwise_or(cv2.inRange(roi, np.array([0, 50, 50]), np.array([15, 255, 255])),
                                                cv2.inRange(roi, np.array([160, 50, 50]), np.array([180, 255, 255])))
                        mask_g = cv2.inRange(roi, np.array([40, 50, 50]), np.array([90, 255, 255]))
                        mask_b = cv2.inRange(roi, np.array([100, 50, 50]), np.array([130, 255, 255]))
                        
                        counts = {"Red": cv2.countNonZero(mask_r), "Green": cv2.countNonZero(mask_g), "Blue": cv2.countNonZero(mask_b)}
                        best_color = max(counts, key=counts.get)
                        
                        if counts[best_color] > 20:
                            identified[role_name] = best_color

                    # 确认红、绿、蓝全齐
                    if len(set(identified.values())) == 3:
                        self.role_to_color = identified
                        self.state = "TRACKING"
                        self.missed_frames = 0
                        print(f"✅ 拓扑结构锁定！物理映射关系: {self.role_to_color}")
            
        elif self.state == "TRACKING":
            if len(centers) >= 3:
                # 寻找最可能的 3 个点（通过距离聚类过滤噪点，这里简化直接取前3个，如果环境干净）
                roles = get_constellation_roles(centers[:3]) 
                if roles:
                    self.current_targets = {}
                    # 【核心魔法】：直接靠位置分配颜色，完全不用 RGB！
                    for role_name, pt in roles.items():
                        color = self.role_to_color[role_name]
                        self.current_targets[color] = pt
                    self.missed_frames = 0
                else:
                    self.missed_frames += 1
            else:
                self.missed_frames += 1

            if self.missed_frames > PATIENCE:
                print("⚠️ 拓扑变形或丢失过久，退回重新验证...")
                self.state = "INIT"
                self.current_targets.clear()

        return binary_img

# ==========================================
# 主程序运行
# ==========================================
if __name__ == "__main__":
    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    cap = cv2.VideoCapture(0)
    tracker = BinaryTopologyTracker()
    colors_bgr = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}

    print("启动相机，寻找“品”字拓扑结构...")
    for _ in range(5): cap.read(); depth_stream.read_frame(); cv2.waitKey(30)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        depth_frame = depth_stream.read_frame()
        dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        depth_map = (np.asarray(dframe_data[:, :, 0], dtype="float32") + 
                     np.asarray(dframe_data[:, :, 1], dtype="float32") * 255)[:, ::-1]

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result_img = frame.copy()

        # 核心处理
        binary_img = tracker.process_frame(frame, hsv_frame)

        # 绘制结果
        if tracker.state == "INIT":
            cv2.putText(result_img, "STATUS: INIT (Waiting for pure Topology)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        elif tracker.state == "TRACKING":
            cv2.putText(result_img, "STATUS: TRACKING (Zero-RGB Topology)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            pts = []
            for color, (cx, cy) in tracker.current_targets.items():
                bgr = colors_bgr[color]
                pts.append((cx, cy))
                
                # 绘制定位圆
                cv2.circle(result_img, (cx, cy), 20, bgr, 3)
                cv2.circle(result_img, (cx, cy), 4, (255, 255, 255), -1)
                
                d_val = depth_map[cy, cx] if (0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]) else 0
                cv2.putText(result_img, f"{color} {d_val:.0f}mm", (cx+25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
                
            # 画出品字形骨架，直观展示等腰三角形拓扑
            if len(pts) == 3:
                cv2.polylines(result_img, [np.array(pts)], isClosed=True, color=(255, 255, 255), thickness=2)

        # 把你提供的完美二值图也显示在右边！
        binary_bgr = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        display_img = np.hstack([result_img, binary_bgr])
        display_img = cv2.resize(display_img, (display_img.shape[1]//2, display_img.shape[0]//2))

        cv2.imshow("Smart Topology Tracking", display_img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    depth_stream.stop()
    dev.close()
    cv2.destroyAllWindows()