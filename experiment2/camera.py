import cv2
import numpy as np
from process import binary_manual, gray_level_manual, gamma_transform, histogram_equalization_manual, make_panel, build_dashboard

def main():
	cap = cv2.VideoCapture(0)

	if not cap.isOpened():
		raise RuntimeError("无法打开摄像头")

	window_name = "Camera Dashboard"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(window_name, 1280, 480)

	print("按 q 键退出。")
	print("二值化使用手写阈值判断；分级灰度使用手写量化；Gamma<1 会偏亮，直方图均衡会提升整体对比度。")

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				continue

			mirrored_frame = np.fliplr(frame)
			gray_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)

			binary_frame = binary_manual(gray_frame, threshold=128)
			gray_4 = gray_level_manual(gray_frame, 4)
			gray_8 = gray_level_manual(gray_frame, 8)
			gray_16 = gray_level_manual(gray_frame, 16)
			gray_32 = gray_level_manual(gray_frame, 32)
			gamma_frame = gamma_transform(gray_frame, gamma_value=0.6)
			histogram_frame = histogram_equalization_manual(gray_frame)

			panels = [
				make_panel(mirrored_frame, "Original"),
				make_panel(binary_frame, "Binary", is_gray=True),
				make_panel(gray_4, "Gray 4", is_gray=True),
				make_panel(gray_8, "Gray 8", is_gray=True),
				make_panel(gray_16, "Gray 16", is_gray=True),
				make_panel(gray_32, "Gray 32", is_gray=True),
				make_panel(gamma_frame, "Gamma", is_gray=True),
				make_panel(histogram_frame, "Histogram Equalization", is_gray=True),
			]

			dashboard = build_dashboard(panels)
			cv2.imshow(window_name, dashboard)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
