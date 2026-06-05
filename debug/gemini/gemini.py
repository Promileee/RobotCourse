import cv2
import numpy as np
from openni import openni2
import matplotlib.pyplot as plt

# ==============================================================================
# --- 1. 配置参数与初始值 ---
# ==============================================================================

# 增强的 HSV 范围，包括红色的两个色块
# 这里的范围比圆柱稍宽，以便同时捕获靶标图案和圆柱
LOWER_RED1 = np.array([0, 100, 50])
UPPER_RED1 = np.array([15, 255, 255])
LOWER_RED2 = np.array([160, 100, 50])
UPPER_RED2 = np.array([180, 255, 255])

LOWER_BLUE = np.array([100, 100, 70])
UPPER_BLUE = np.array([130, 255, 255])

LOWER_GREEN = np.array([40, 70, 70])
UPPER_GREEN = np.array([90, 255, 255])

# 深度限制 (毫米)
DEPTH_MAX_ABS = 3000   # 绝对深度上限 (过滤远处背景)
DEPTH_MIN_CYLINDER = 0 # 圆柱体的深度下限 (用于靶标检测的相对深度过滤)
DEPTH_TARGET_OFFSET = 50 # 靶标相对于圆柱体的最小深度偏移

# 稳定性逻辑
SKIP_FRAMES = 10     # 忽略前10帧以稳定传感器
STABLE_FRAMES = 5   # 需要连续5帧稳定检测
HISTORY_MAX = 10     # 历史记录的最大帧数

# 形态学核大小
MORPH_KERNEL_CYL = (7, 7)
MORPH_KERNEL_TAR = (15, 15) # 靶标需要更大的核来连接图案

# 窗口和 Trackbar 名称
WIN_RESULT = "Detection Result"
WIN_PREVIEW = "Image Previews"

# ==============================================================================
# --- 2. 目标检测器类 ---
# ==============================================================================
class TargetDetector:
    def __init__(self):
        # --- 初始化硬件 ---
        # 1. RGB 相机
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Cannot open RGB camera.")
            exit()

        # 2. 深度相机
        openni2.initialize()
        self.dev = openni2.Device.open_any()
        print(f"✅ Depth device initialized: {self.dev.get_device_info().name}")
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.start()

        # CLAHE 对象用于预处理
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 稳定性状态
        self.frame_count = 0
        self.history = [] # 历史记录列表：{"Laser": (x,y,d), "Cylinders": {"Red": (rc, d), ...}, "Targets": {...}}
        self.final_stable_rects = None
        self.last_frame = None

        # --- 初始化 Trackbars ---
        self._init_trackbars()

    def _init_trackbars(self):
        cv2.namedWindow(WIN_RESULT)
        # 形态学核大小
        cv2.createTrackbar("Morph Cyl (Odd)", WIN_RESULT, MORPH_KERNEL_CYL[0], 21, lambda _: None)
        cv2.createTrackbar("Morph Tar (Odd)", WIN_RESULT, MORPH_KERNEL_TAR[0], 31, lambda _: None)
        # 深度上限
        cv2.createTrackbar("Depth Max Abs (mm)", WIN_RESULT, DEPTH_MAX_ABS, 5000, lambda _: None)
        cv2.createTrackbar("Target Depth Offset (mm)", WIN_RESULT, DEPTH_TARGET_OFFSET, 500, lambda _: None)
        # 圆度阈值 (针对靶标)
        cv2.createTrackbar("Target Roundness (0.1x)", WIN_RESULT, 7, 10, lambda _: None) # 0.7
        # 长宽比范围
        cv2.createTrackbar("Cyl Aspect Rng (x100)", WIN_RESULT, 50, 100, lambda _: None) # 0.5 - 2.0
        cv2.createTrackbar("Tar Aspect Rng (x100)", WIN_RESULT, 70, 100, lambda _: None) # 0.7 - 1.4
        # 深度过滤的覆盖率 (圆柱)
        cv2.createTrackbar("Cyl Depth Ratio (x100)", WIN_RESULT, 30, 100, lambda _: None) # 0.3

    def get_frame_and_depth(self):
        """读取 RGB 和毫米深度帧"""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None

        depth_frame = self.depth_stream.read_frame()
        dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([depth_frame.height, depth_frame.width, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype="float32")
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype="float32")
        dpt2 *= 255
        depth = dpt1 + dpt2
        # --- 对齐 (手动翻转深度) ---
        # 深度相机通常需要沿X轴翻转以与 RGB 对齐，这里提供一个开关
        # 您可能需要根据实际物理位置调整：False or True
        if True:
            depth = depth[:, ::-1]

        self.frame_count += 1
        self.last_frame = frame.copy()
        return frame, depth

    def preprocess_image(self, frame):
        """CLAHE 预处理，增强对比度"""
        # 转为 HSV -> 分离 V 通道 -> 应用 CLAHE -> 合并 -> 转回 BGR
        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_full)
        v_enhanced = self.clahe.apply(v)
        hsv_enhanced = cv2.merge([h, s, v_enhanced])
        return hsv_enhanced

    def get_trackbar_values(self):
        """获取 Trackbars 的最新值"""
        max_abs_depth = cv2.getTrackbarPos("Depth Max Abs (mm)", WIN_RESULT)
        target_offset = cv2.getTrackbarPos("Target Depth Offset (mm)", WIN_RESULT)
        roundness_tar = cv2.getTrackbarPos("Target Roundness (0.1x)", WIN_RESULT) / 10.0
        aspect_cyl = cv2.getTrackbarPos("Cyl Aspect Rng (x100)", WIN_RESULT) / 100.0
        aspect_tar = cv2.getTrackbarPos("Tar Aspect Rng (x100)", WIN_RESULT) / 100.0
        ratio_cyl = cv2.getTrackbarPos("Cyl Depth Ratio (x100)", WIN_RESULT) / 100.0
        morph_cyl = cv2.getTrackbarPos("Morph Cyl (Odd)", WIN_RESULT)
        if morph_cyl % 2 == 0: morph_cyl += 1
        morph_tar = cv2.getTrackbarPos("Morph Tar (Odd)", WIN_RESULT)
        if morph_tar % 2 == 0: morph_tar += 1

        return {
            "max_abs_depth": max_abs_depth,
            "target_offset": target_offset,
            "roundness_tar": roundness_tar,
            "aspect_cyl_rng": (1.0 - aspect_cyl, 1.0 + aspect_cyl), # (0.5, 1.5)
            "aspect_tar_rng": (1.0 - aspect_tar, 1.0 + aspect_tar), # (0.7, 1.3)
            "ratio_cyl": ratio_cyl,
            "morph_cyl_kernel": (morph_cyl, morph_cyl),
            "morph_tar_kernel": (morph_tar, morph_tar)
        }

    def detect_laser(self, frame, depth_mask, hsv_enhanced):
        """激光检测：动态阈值 + 深度过滤"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # OTSU 动态阈值，并取 V 通道作为参考
        _, bright_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 将 V 通道的高亮度部分合并，以处理极亮的情况
        # 将 V 通道的高亮度部分合并，以处理极亮的情况
        _, _, hsv_enhanced_v = cv2.split(hsv_enhanced)
        _, hsv_enhanced_thresh = cv2.threshold(hsv_enhanced_v, 240, 255, cv2.THRESH_BINARY)
        combined_bright = cv2.bitwise_or(bright_thresh, hsv_enhanced_thresh)

        # 结合深度过滤和形态学
        combined = cv2.bitwise_and(combined_bright, depth_mask)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 5:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy) # 激光在深度图上的深度，后续处理中获取
        return None

    def detect_cylinders(self, hsv_enhanced, depth_mask, params, depth):
        """圆柱体检测：基于颜色 + 形状 + 深度比"""
        frame_rects = {}
        detected_min_depth = 9999
        results_list = []

        for color_name, (lower, upper, lower2) in [("Red", (LOWER_RED1, UPPER_RED1, LOWER_RED2, UPPER_RED2)),
                                                 ("Blue", (LOWER_BLUE, UPPER_BLUE, None, None)),
                                                 ("Green", (LOWER_GREEN, UPPER_GREEN, None, None))]:
            color_mask = cv2.inRange(hsv_enhanced, lower[0], lower[1])
            if lower2 is not None:
                mask2 = cv2.inRange(hsv_enhanced, lower2[0], lower2[1])
                color_mask = cv2.bitwise_or(color_mask, mask2)
            
            # 形态学处理，连接和去除噪声
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones(params["morph_cyl_kernel"], np.uint8))
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones(params["morph_cyl_kernel"], np.uint8))
            
            # 找轮廓
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_rect = None
            best_area = 0
            best_depth = -1

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500: continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if not (params["aspect_cyl_rng"][0] < aspect_ratio < params["aspect_cyl_rng"][1]):
                    continue

                # 深度覆盖率检查：颜色轮廓中有多少像素在有效深度范围内
                roi_color = color_mask[y:y+h, x:x+w]
                roi_depth = depth_mask[y:y+h, x:x+w]
                in_range_pixels = cv2.countNonZero(cv2.bitwise_and(roi_color, roi_depth))
                depth_ratio = in_range_pixels / (float(w*h) + 1)
                
                if depth_ratio < params["ratio_cyl"]:
                    continue

                # 计算圆柱体的深度 (ROI 区域的深度均值/中位数，比中心点更稳定)
                roi_depth_raw = depth[y:y+h, x:x+w]
                # 过滤掉 0 值和 DEPTH_MAX_ABS 之外的值
                valid_depths = roi_depth_raw[(roi_depth_raw > 0) & (roi_depth_raw < params["max_abs_depth"])]
                if len(valid_depths) > 0:
                    best_depth = np.median(valid_depths)
                    if best_depth < detected_min_depth:
                        detected_min_depth = best_depth
                else:
                    best_depth = -1

                # 检查凸包的圆度，以确保不是复杂的背景
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                hull_perimeter = cv2.arcLength(hull, True)
                roundness = (4 * np.pi * hull_area) / (hull_perimeter * hull_perimeter + 1)
                if roundness < 0.2: # 圆柱体纵向看也是接近圆，这里做一个基础过滤
                    continue
                
                if area > best_area:
                    best_area = area
                    best_rect = (x, y, w, h)

            if best_rect is not None:
                frame_rects[color_name] = (best_rect, best_depth)
                results_list.append((color_name, best_rect, best_depth, 1)) # Type 1 for Cylinder

        return frame_rects, detected_min_depth, results_list

    def detect_targets(self, hsv_enhanced, depth_mask, params, depth, cylinder_min_depth):
        """靶标检测：颜色掩码 + **凸包** + **圆度** + **相对深度** (靶标 > 圆柱)"""
        frame_rects = {}
        results_list = []

        # 即使圆柱体未检测到，也使用一个较大的深度偏移来排除背景误报
        if cylinder_min_depth == 9999:
            base_min_depth = params["max_abs_depth"] / 2.0
        else:
            base_min_depth = cylinder_min_depth + params["target_offset"]

        # 靶标相对于圆柱体的最小深度过滤 (靶标应在圆柱后面)
        depth_offset_mask = np.where((depth > base_min_depth) & (depth < params["max_abs_depth"]), 255, 0).astype(np.uint8)

        for color_name, (lower, upper, lower2) in [("Red", (LOWER_RED1, UPPER_RED1, LOWER_RED2, UPPER_RED2)),
                                                 ("Blue", (LOWER_BLUE, UPPER_BLUE, None, None)),
                                                 ("Green", (LOWER_GREEN, UPPER_GREEN, None, None))]:
            # **针对靶标调宽 HSV 饱和度/亮度范围**，以捕获同心圆环
            # 这是一个关键步骤：宽范围用于将同心圆环从白色圆盘上分离
            color_mask = cv2.inRange(hsv_enhanced, lower[0], lower[1])
            if lower2 is not None:
                mask2 = cv2.inRange(hsv_enhanced, lower2[0], lower2[1])
                color_mask = cv2.bitwise_or(color_mask, mask2)
            
            # **大核形态学闭运算**，将同心圆环连接成一个整体
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones(params["morph_tar_kernel"], np.uint8))
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) # 去除微小噪声

            # 找轮廓
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_rect = None
            best_area = 0
            best_depth = -1

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500: continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if not (params["aspect_tar_rng"][0] < aspect_ratio < params["aspect_tar_rng"][1]):
                    continue

                # **关键：计算凸包 (Convex Hull)**，使同心圆图案看起来更像圆盘
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                hull_perimeter = cv2.arcLength(hull, True)
                # **关键：检查凸包的圆度 (Roundness)**
                roundness = (4 * np.pi * hull_area) / (hull_perimeter * hull_perimeter + 1)
                
                if roundness < params["roundness_tar"]: # 凸包圆度应接近 1
                    continue

                # 计算靶标的深度 (ROI 区域的深度均值/中位数)
                roi_depth_raw = depth[y:y+h, x:x+w]
                valid_depths = roi_depth_raw[(roi_depth_raw > 0) & (roi_depth_raw < params["max_abs_depth"])]
                if len(valid_depths) > 0:
                    best_depth = np.median(valid_depths)
                else:
                    best_depth = -1
                
                # **关键：深度过滤**：确保靶标比圆柱远，并且比远处背景近
                if best_depth < base_min_depth: # 排除比圆柱更近的具有相似颜色的物体
                    continue
                
                if area > best_area:
                    best_area = area
                    best_rect = (x, y, w, h)

            if best_rect is not None:
                frame_rects[color_name] = (best_rect, best_depth)
                results_list.append((color_name, best_rect, best_depth, 2)) # Type 2 for Target

        return frame_rects, results_list

    def update_stability(self, frame_laser, frame_cylinders, frame_targets):
        """
        更新稳定性检查逻辑。
        只有所有三个目标类别（激光、3圆柱、3靶标）都同时检测到，才更新稳定性。
        这是一个简化版本，确保全场目标稳定。
        """
        all_cylinders = all(v[0] is not None for v in frame_cylinders.values()) and len(frame_cylinders) == 3
        all_targets = all(v[0] is not None for v in frame_targets.values()) and len(frame_targets) == 3
        
        frame_all_detected = (frame_laser is not None) and all_cylinders and all_targets

        if not frame_all_detected:
            self.history.clear()
            return

        # 保存当前帧的简短状态
        current_frame_data = {
            "Laser": frame_laser,
            "Cylinders": {k: v[0] for k, v in frame_cylinders.items()}, # 只要 rect 坐标
            "Targets": {k: v[0] for k, v in frame_targets.items()}
        }
        self.history.append(current_frame_data)
        if len(self.history) > STABLE_FRAMES:
            self.history.pop(0)

        if len(self.history) >= STABLE_FRAMES:
            stable = True
            # 通用矩形移动检查 (x, y, w, h 偏移量不超过 5)
            # 1. 激光中心点
            if True:
                prev_cx, prev_cy = self.history[-2]["Laser"]
                curr_cx, curr_cy = self.history[-1]["Laser"]
                if abs(prev_cx - curr_cx) > 8 or abs(prev_cy - curr_cy) > 8: stable = False
            
            # 2. 圆柱体和靶标
            if stable:
                for target_type in ["Cylinders", "Targets"]:
                    for name in ["Red", "Green", "Blue"]:
                        prev_rc = self.history[-2][target_type][name]
                        curr_rc = self.history[-1][target_type][name]
                        for i in range(4): # x, y, w, h
                            if abs(prev_rc[i] - curr_rc[i]) > 8:
                                stable = False
                                break
                        if not stable: break
                    if not stable: break

            if stable:
                # 稳定性达成：保存最后的 raw 结果
                self.final_stable_rects = current_frame_data
                print(f"\n✅ Stable detection achieved at frame {self.frame_count}. converging...")
                
                # --- Correction: Unify Rect sizes (using median, but applied to specific groups) ---
                for target_type in ["Cylinders", "Targets"]:
                    ws = [self.final_stable_rects[target_type][name][2] for name in ["Red", "Green", "Blue"]]
                    hs = [self.final_stable_rects[target_type][name][3] for name in ["Red", "Green", "Blue"]]
                    med_w = int(np.median(ws))
                    med_h = int(np.median(hs))
                    corrected = {}
                    for name in ["Red", "Green", "Blue"]:
                        x, y, w, h = self.final_stable_rects[target_type][name]
                        cx = x + w // 2
                        cy = y + h // 2
                        nx = cx - med_w // 2
                        ny = cy - med_h // 2
                        corrected[name] = (nx, ny, med_w, med_h)
                    self.final_stable_rects[target_type] = corrected

                print("Corrected coordinates:")
                for tt in ["Cylinders", "Targets"]:
                    for name in ["Red", "Green", "Blue"]:
                        rc = self.final_stable_rects[tt][name]
                        print(f"  {tt} {name}: {rc}")

    def draw_results(self, frame, laser, cylinders, targets, params, depth):
        vis = frame.copy()
        
        # 深度图用于可视化 (剪裁并归一化)
        depth_vis = np.clip(depth, 0, params["max_abs_depth"])
        depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 激光点
        if laser:
            lx, ly = laser
            ld_val = -1
            if 0 <= ly < depth.shape[0] and 0 <= lx < depth.shape[1]:
                ld_val = depth[ly, lx]
            cv2.circle(vis, (lx, ly), 10, (0, 255, 0), -1) # 绿色激光
            cv2.putText(vis, f"Laser ({lx},{ly}) d={ld_val:.0f}mm", (lx+15, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 目标绘制列表
        fill_colors = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}
        text_colors = {"Red": (255, 255, 255), "Green": (0, 0, 0), "Blue": (255, 255, 255)}

        for type_list in [cylinders, targets]:
            for color_name, (best_rect, best_depth) in type_list.items():
                if best_rect is not None:
                    x, y, w, h = best_rect
                    cv2.rectangle(vis, (x, y), (x + w, y + h), fill_colors[color_name], 2)
                    label_type = "Cyl" if type_list == cylinders else "Tar"
                    label = f"{label_type} {color_name} d={best_depth:.0f}mm"
                    cv2.putText(vis, label, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, fill_colors[color_name], 1)
                    # 同时绘制在深度图上以验证对齐
                    cv2.rectangle(depth_vis, (x, y), (x + w, y + h), (255, 255, 255), 2)

        return vis, depth_vis

    def run(self):
        while True:
            frame, depth = self.get_frame_and_depth()
            if frame is None: break

            # 1. 前处理
            hsv_enhanced = self.preprocess_image(frame)
            params = self.get_trackbar_values()

            # 2. 深度过滤掩码 (绝对深度限制)
            depth_mask = np.where((depth > 0) & (depth < params["max_abs_depth"]), 255, 0).astype(np.uint8)

            # 3. 目标检测
            if self.frame_count > SKIP_FRAMES:
                # A. 激光检测
                frame_laser = self.detect_laser(frame, depth_mask, hsv_enhanced)

                # B. 圆柱体检测 (并保存最小检测深度)
                frame_cylinders, cylinder_min_depth, results_list_cyl = self.detect_cylinders(hsv_enhanced, depth_mask, params, depth)

                # C. 靶标检测 (使用相对深度过滤：靶标 > 圆柱)
                frame_targets, results_list_tar = self.detect_targets(hsv_enhanced, depth_mask, params, depth, cylinder_min_depth)

                # 4. 稳定性处理
                self.update_stability(frame_laser, frame_cylinders, frame_targets)

                # 绘制结果
                vis, depth_vis = self.draw_results(frame, frame_laser, frame_cylinders, frame_targets, params, depth)
                cv2.imshow(WIN_RESULT, vis)
                cv2.imshow(WIN_PREVIEW, depth_vis)

            else:
                # 忽略前几帧
                cv2.imshow(WIN_RESULT, frame)

            # 稳定性达成后的可视化
            if self.final_stable_rects is not None:
                # 在窗口右上角显示一个指示符
                cv2.rectangle(vis, (frame.shape[1]-100, 10), (frame.shape[1]-10, 50), (0, 255, 0), -1)
                cv2.putText(vis, "STABLE", (frame.shape[1]-90, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(WIN_RESULT, vis)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            if key == ord("r"): # 'r' 重置稳定性检查
                self.history.clear()
                self.final_stable_rects = None
                print("\nStability reset manually.")

        # ==============================================================================
        # --- 3. 最终输出与可视化 (Matplotlib) ---
        # ==============================================================================
        self.cleanup()
        
        if self.final_stable_rects is not None:
            self._final_visualize_plot()
        else:
            print("\n❌ Stability logic did not converge. Using last available frame...")
            self._final_visualize_plot_last()

    def cleanup(self):
        self.cap.release()
        self.depth_stream.stop()
        self.dev.close()
        cv2.destroyAllWindows()

    def _final_visualize_plot(self):
        """ Matplotlib 可视化稳定性达成后的结果"""
        print("\n✅ Stable detection results:")
        vis = self.last_frame.copy()
        fill_colors = {"Red": (0, 0, 255), "Green": (0, 255, 0), "Blue": (255, 0, 0)}
        text_colors = {"Red": (255, 255, 255), "Green": (0, 0, 0), "Blue": (255, 255, 255)}

        # 通用填充
        for target_type in ["Cylinders", "Targets"]:
            for name, rc in self.final_stable_rects[target_type].items():
                x, y, w, h = rc
                # 在图像上填充区域
                vis[y:y+h, x:x+w] = fill_colors[name] # 直接覆盖像素颜色
                # 绘制文字
                label_type = "Cyl" if target_type == "Cylinders" else "Tar"
                label = f"{label_type} {name} {rc}"
                cv2.putText(vis, label, (x+5, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_colors[name], 1)
        
        # 激光点
        lx, ly = self.final_stable_rects["Laser"]
        cv2.circle(vis, (lx, ly), 12, (255, 255, 255), -1) # 白色大圆
        cv2.circle(vis, (lx, ly), 6, (0, 255, 0), -1) # 绿色中心

        # 最终显示
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(vis_rgb)
        plt.title("Stable Detection (Corrected & Filled Coordinates)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def _final_visualize_plot_last(self):
        """ 可视化最后一帧"""
        vis_rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(vis_rgb)
        plt.title("Final Frame (Stability Not Reached)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# ==============================================================================
# --- 4. 主程序运行 ---
# ==============================================================================
if __name__ == "__main__":
    detector = TargetDetector()
    detector.run()