import cv2
from camera_manager import CameraManager

# 颜色映射字典
color_map = {
    '1': '红',
    '2': '绿',
    '3': '蓝'
}

def main():
    cam = CameraManager()
    cam.start()
    detector = cv2.QRCodeDetector()

    print("开始识别二维码，按 'q' 键退出。")
    while True:
        frame = cam.read_rgb_frame()
        if frame is None:
            print("获取图像失败。")
            break

        data, bbox, _ = detector.detectAndDecode(frame)
        if data:
            # 绘制边界
            if bbox is not None:
                pts = bbox.astype(int)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(frame, data, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 按顺序输出每个数字对应的颜色
            print(f"识别到二维码内容: {data}")
            for ch in data:
                color = color_map.get(ch, '未知')
                print(f"  -> 数字 {ch} : {color}")

        cv2.imshow("QR Code Scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
