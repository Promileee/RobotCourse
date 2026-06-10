import cv2
import numpy as np
from camera_manager import CameraManager


class QRCodeRecognizer:
    """二维码识别器，支持颜色映射输出"""

    COLOR_MAP = {
        '1': '红',
        '2': '绿',
        '3': '蓝',
    }

    def __init__(self):
        self._detector = cv2.QRCodeDetector()

    def detect(self, frame: np.ndarray):
        """
        检测单帧图像中的二维码。

        Returns:
            (data, bbox, colors) 元组。
            - data: 解码后的字符串，未检测到时为 None
            - bbox: 二维码角点坐标，未检测到时为 None
            - colors: 与 data 每个字符对应的颜色名称列表
        """
        data, bbox, _ = self._detector.detectAndDecode(frame)
        if not data:
            return None, None, []

        colors = [self.COLOR_MAP.get(ch, '未知') for ch in data]
        return data, bbox, colors

    @staticmethod
    def draw_result(frame: np.ndarray, data: str, bbox):
        """在图像上绘制二维码边界和解码内容"""
        if bbox is not None:
            pts = bbox.astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cv2.putText(frame, data, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def run(self, cam: CameraManager = None):
        """
        启动交互式识别循环。按 'q' 退出。

        Args:
            cam: 可选，已初始化的 CameraManager 实例。若为 None 则自动创建。
        """
        own_cam = cam is None
        if own_cam:
            cam = CameraManager()
            cam.start()

        print("开始识别二维码，按 'q' 键退出。")
        try:
            while True:
                frame = cam.read_rgb_frame()
                if frame is None:
                    print("获取图像失败。")
                    break

                data, bbox, colors = self.detect(frame)
                if data:
                    self.draw_result(frame, data, bbox)
                    print(f"识别到二维码内容: {data}")
                    for ch, color in zip(data, colors):
                        print(f"  -> 数字 {ch} : {color}")

                cv2.imshow("QR Code Scanner", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            if own_cam:
                cam.release()
            cv2.destroyAllWindows()


def main():
    recognizer = QRCodeRecognizer()
    recognizer.run()


if __name__ == "__main__":
    main()
