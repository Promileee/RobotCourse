import cv2
import numpy as np
from process2 import (
	binary_manual,
	gray_level_manual,
	gamma_transform,
	histogram_equalization_manual,
	log_transform,
	binary_otsu,
	clahe_manual,
	histogram_counts,
	make_panel,
)


def hist_to_image(hist, height=200, width=256):
	h = np.array(hist, dtype=np.float32)
	if h.max() > 0:
		h = h / h.max()
	else:
		h = h
	img = np.ones((height, width), dtype=np.uint8) * 255
	for x in range(min(width, h.size)):
		val = int(h[x] * (height - 1))
		cv2.line(img, (x, height - 1), (x, height - 1 - val), 0, 1)
	return img


def build_grid(panels, cols=5):
	rows = []
	for i in range(0, len(panels), cols):
		row = cv2.hconcat(panels[i:i+cols])
		rows.append(row)
	return cv2.vconcat(rows)


def main():
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("无法打开摄像头")

	window_name = "Camera Dashboard"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	panel_size = (256, 192)  # (width, height)
	canvas_w = panel_size[0] * 5
	canvas_h = panel_size[1] * 4
	cv2.resizeWindow(window_name, canvas_w, canvas_h)

	gamma_vals = [0.2, 0.4, 0.6, 0.8]

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				continue

			frame = np.fliplr(frame).copy()
			original = frame
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			bin64 = binary_manual(gray, threshold=64)
			bin128 = binary_manual(gray, threshold=128)
			bin192 = binary_manual(gray, threshold=192)
			bin_otsu = binary_otsu(gray)

			g4 = gray_level_manual(gray, 4)
			g8 = gray_level_manual(gray, 8)
			g16 = gray_level_manual(gray, 16)
			g32 = gray_level_manual(gray, 32)

			gamma_imgs = [gamma_transform(gray, g) for g in gamma_vals]

			log_img = log_transform(gray)
			he = histogram_equalization_manual(gray)
			clahe = clahe_manual(gray, tile_grid_size=(8, 8), clip_limit=2.0)

			hist_original = histogram_counts(gray)
			hist_he = histogram_counts(he)
			hist_clahe = histogram_counts(clahe)

			hist_img_orig = hist_to_image(hist_original, height=panel_size[1], width=256)
			hist_img_he = hist_to_image(hist_he, height=panel_size[1], width=256)
			hist_img_clahe = hist_to_image(hist_clahe, height=panel_size[1], width=256)

			panels = [
				make_panel(original, "Original", size=panel_size, is_gray=False),
				make_panel(gray, "Gray", size=panel_size, is_gray=True),
				make_panel(bin64, "Binary 64", size=panel_size, is_gray=True),
				make_panel(bin128, "Binary 128", size=panel_size, is_gray=True),
				make_panel(bin192, "Binary 192", size=panel_size, is_gray=True),
				make_panel(bin_otsu, "Binary OTSU", size=panel_size, is_gray=True),
				make_panel(g4, "Quant 4", size=panel_size, is_gray=True),
				make_panel(g8, "Quant 8", size=panel_size, is_gray=True),
				make_panel(g16, "Quant 16", size=panel_size, is_gray=True),
				make_panel(g32, "Quant 32", size=panel_size, is_gray=True),
				make_panel(gamma_imgs[0], f"Gamma {gamma_vals[0]}", size=panel_size, is_gray=True),
				make_panel(gamma_imgs[1], f"Gamma {gamma_vals[1]}", size=panel_size, is_gray=True),
				make_panel(gamma_imgs[2], f"Gamma {gamma_vals[2]}", size=panel_size, is_gray=True),
				make_panel(gamma_imgs[3], f"Gamma {gamma_vals[3]}", size=panel_size, is_gray=True),
				make_panel(log_img, "Log Transform", size=panel_size, is_gray=True),
				make_panel(he, "HE", size=panel_size, is_gray=True),
				make_panel(clahe, "CLAHE", size=panel_size, is_gray=True),
				make_panel(hist_img_orig, "Hist Original", size=panel_size, is_gray=True),
				make_panel(hist_img_he, "Hist HE", size=panel_size, is_gray=True),
				make_panel(hist_img_clahe, "Hist CLAHE", size=panel_size, is_gray=True),
			]

			canvas = build_grid(panels, cols=5)
			cv2.imshow(window_name, canvas)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
