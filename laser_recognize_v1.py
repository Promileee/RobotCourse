"""
laser_recognize_v2.py
激光点识别模块 V2 (结合靶标 ROI、深度切片与卡尔曼时域滤波)
"""

import cv2
import numpy as np
from camera_manager import CameraManager
from target_recognize_v2 import TargetRecognizerV2, KalmanFilter2D

class LaserRecognizerV2:
    def __init__(self, depth_tolerance=40, v_thresh_min=220, kalman_q=0.1, kalman_r=5.0):
        self.depth_tolerance = depth_tolerance
        self.v_thresh_min = v_thresh_min
        self.kf = KalmanFilter2D(dt=1.0, q_val=kalman_q, r_val=kalman_r)
        self.lost_frames = 0
        self.max_lost = 5

    def _create_roi_mask(self, shape, target_info):
        mask = np.zeros(shape, dtype=np.uint8)
        if target_info and "center" in target_info and "radius" in target_info:
            cx, cy = target_info["center"]
            r = int(target_info["radius"] * 1.2)
            cv2.circle(mask, (int(cx), int(cy)), r, 255, -1)
        else:
            mask.fill(255)
        return mask

    def _create_depth_slice_mask(self, dpt, target_info):
        if not target_info or "mean_depth" not in target_info:
            return np.where((dpt > 100) & (dpt < 2000), 255, 0).astype(np.uint8)
        
        mean_depth = target_info["mean_depth"]
        lower = mean_depth - self.depth_tolerance
        upper = mean_depth + self.depth_tolerance
        return np.where((dpt > lower) & (dpt < upper), 255, 0).astype(np.uint8)

    def detect(self, frame, dpt, target_info=None, base_img=None):
        h, w = frame.shape[:2]
        result = base_img.copy() if base_img is not None else frame.copy()
        
        roi_mask = self._create_roi_mask((h, w), target_info)
        
        if dpt.shape[:2] != (h, w):
            dpt_resized = cv2.resize(dpt, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            dpt_resized = dpt

        depth_mask = self._create_depth_slice_mask(dpt_resized, target_info)
        valid_space_mask = cv2.bitwise_and(roi_mask, depth_mask)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        masked_v = cv2.bitwise_and(v_channel, v_channel, mask=valid_space_mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(masked_v)

        spot = None
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        if max_val >= self.v_thresh_min:
            dynamic_thresh = max(self.v_thresh_min, int(max_val) - 20)
            _, bright_mask = cv2.threshold(v_channel, dynamic_thresh, 255, cv2.THRESH_BINARY)
            
            lower_red1 = np.array([0, 30, dynamic_thresh])
            upper_red1 = np.array([20, 255, 255])
            lower_red2 = np.array([160, 30, dynamic_thresh])
            upper_red2 = np.array([180, 255, 255])
            
            r_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            r_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(r_mask1, r_mask2)
            
            _, core_mask = cv2.threshold(v_channel, 250, 255, cv2.THRESH_BINARY)
            laser_color_mask = cv2.bitwise_or(red_mask, core_mask)

            combined_mask = cv2.bitwise_and(bright_mask, laser_color_mask)
            combined_mask = cv2.bitwise_and(combined_mask, valid_space_mask)
            
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                valid_contours = [c for c in contours if 2 <= cv2.contourArea(c) <= 400] 
                if valid_contours:
                    best_cnt = max(valid_contours, key=cv2.contourArea)
                    M = cv2.moments(best_cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        spot = (cx, cy)

        if spot is not None:
            self.lost_frames = 0
            kf_pos = self.kf.update(spot[0], spot[1])
            final_cx, final_cy = int(round(kf_pos[0])), int(round(kf_pos[1]))
            
            d_val = 0
            if 0 <= final_cy < h and 0 <= final_cx < w:
                d_val = dpt_resized[final_cy, final_cx]
            
            spot_info = (final_cx, final_cy, d_val)
            
            cv2.circle(result, (final_cx, final_cy), 18, (0, 255, 255), 2)  
            cv2.circle(result, (final_cx, final_cy), 3, (0, 0, 255), -1)    
            
            label = f"Laser: ({final_cx},{final_cy}) D:{d_val:.0f}mm"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (final_cx + 15, final_cy - lh - 15), 
                          (final_cx + 15 + lw, final_cy - 5), (0, 0, 0), -1)
            cv2.putText(result, label, (final_cx + 15, final_cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost:
                self.kf.reset()
            spot_info = None

        return result, combined_mask, spot_info


# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    print("正在初始化相机与识别模块...")
    cam = CameraManager()
    cam.start()

    # 初始化靶标识别（圆盘上的）
    tr = TargetRecognizerV2()
    tr.setup(cam)
    
    # 初始化激光识别
    lr = LaserRecognizerV2()

    window_name = "Laser Recognition V2 (Debug)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    cv2.createTrackbar("V_Min_Thresh", window_name, 220, 255, lambda _: None)
    cv2.createTrackbar("Depth_Tol_mm", window_name, 40, 200, lambda _: None)

    print("系统就绪。按 'q' 键退出。")

    while True:
        frame = cam.read_rgb_frame()
        if frame is None: continue
        dpt = cam.get_depth_map()
        if dpt is None: continue

        lr.v_thresh_min = cv2.getTrackbarPos("V_Min_Thresh", window_name)
        lr.depth_tolerance = cv2.getTrackbarPos("Depth_Tol_mm", window_name)

        target_info = None
        base_img = frame.copy()

        # 【重点修复】：安全调用真正的靶标识别！
        try:
            tracking_success, targets, tr_img, rot_angle, affine_angle = tr.track_frame(frame, dpt)
            
            if tracking_success and hasattr(tr, 'ref_mask_center') and tr.ref_mask_center is not None:
                depths = []
                for color, (cx, cy) in targets.items():
                    if 0 <= cy < dpt.shape[0] and 0 <= cx < dpt.shape[1]:
                        d = dpt[cy, cx]
                        if d > 0: depths.append(d)
                
                mean_d = sum(depths) / len(depths) if depths else 1000.0

                target_info = {
                    "center": tr.ref_mask_center, # 真正圆盘的中心！
                    "radius": tr.ref_mask_radius, # 真正圆盘的半径！
                    "mean_depth": mean_d
                }
                base_img = tr_img
        except TypeError:
            # 捕捉到解包报错，说明靶标类还没找到目标、还没初始化好 ref_mask_center。
            # 这里什么都不做，安全跳过，防止程序崩溃！
            pass
        except Exception as e:
            # 捕捉其他可能的错误
            print(f"Target recognizer exception ignored: {e}")
            pass

        # 运行激光识别（哪怕靶标没找到，target_info 是 None 也能跑，找到了就会加上圆盘约束）
        final_img, laser_mask, spot = lr.detect(frame, dpt, target_info, base_img=base_img)

        cv2.imshow(window_name, final_img)
        cv2.imshow("Laser Mask Debug", laser_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()