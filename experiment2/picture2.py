import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from process2 import (
	binary_manual,
	gray_level_manual,
	gamma_transform,
	histogram_equalization_manual,
	log_transform,
	otsu_threshold,
	binary_otsu,
	clahe_manual,
    histogram_counts,
)


def main():
	image_path = r'C:\Users\20230\Desktop\RobotCourse\experiment2\cT9mqPVWS6.jpg'
	frame = cv2.imread(str(image_path))

	if frame is None:
		raise FileNotFoundError(f"无法读取图片: {image_path}")

	# Original and gray
	original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Binary with thresholds 64,128,192 and OTSU
	bin64 = binary_manual(gray, threshold=64)
	bin128 = binary_manual(gray, threshold=128)
	bin192 = binary_manual(gray, threshold=192)
	bin_otsu = binary_otsu(gray)

	# Gray level quantization (4 variants)
	g4 = gray_level_manual(gray, 4)
	g8 = gray_level_manual(gray, 8)
	g16 = gray_level_manual(gray, 16)
	g32 = gray_level_manual(gray, 32)

	# Gamma transforms (four values)
	gamma_vals = [0.2, 0.4, 0.6, 0.8]
	gamma_imgs = [gamma_transform(gray, g) for g in gamma_vals]

	# Log transform
	log_img = log_transform(gray)

	# Histogram equalization (manual) and CLAHE
	he = histogram_equalization_manual(gray)
	clahe = clahe_manual(gray, tile_grid_size=(8, 8), clip_limit=2.0)

	# arrange 16 images in 4x4
	# prepare titles and image/plot items for 4x5 grid (20 items)
	titles = [
		'Original', 'Gray',
		'Binary 64', 'Binary 128',
		'Binary 192', 'Binary OTSU',
		'Quant 4', 'Quant 8',
		'Quant 16', 'Quant 32',
		f'Gamma {gamma_vals[0]}', f'Gamma {gamma_vals[1]}',
		f'Gamma {gamma_vals[2]}', f'Gamma {gamma_vals[3]}',
		'Log Transform', 'HE', 'CLAHE',
		'Hist Original', 'Hist HE', 'Hist CLAHE'
	]

	imgs = [
		original, gray,
		bin64, bin128,
		bin192, bin_otsu,
		g4, g8,
		g16, g32,
		gamma_imgs[0], gamma_imgs[1], gamma_imgs[2], gamma_imgs[3],
		log_img, he, clahe,
		None, None, None  # placeholders for histograms
	]

	# compute histograms (use process function)
	hist_original = histogram_counts(gray)
	hist_he = histogram_counts(he)
	hist_clahe = histogram_counts(clahe)

	fig, axes = plt.subplots(4, 5, figsize=(20, 9))
	axes = axes.flatten()
	for idx, ax in enumerate(axes):
		title = titles[idx]
		item = imgs[idx]
		if 'Hist' in title:
			# select histogram
			if title == 'Hist Original':
				h = hist_original
			elif title == 'Hist HE':
				h = hist_he
			else:
				h = hist_clahe
			ax.bar(np.arange(256), h, width=1.0, color='black')
			ax.set_xlim([0, 255])
			ax.set_title(title)
			ax.set_xticks([])
			ax.set_yticks([])
		else:
			if isinstance(item, np.ndarray):
				if item.ndim == 3:
					ax.imshow(item)
				else:
					ax.imshow(item, cmap='gray', vmin=0, vmax=255)
			else:
				ax.text(0.5, 0.5, title, ha='center', va='center')
			ax.set_title(title)
			ax.axis('off')

	plt.tight_layout()
	plt.savefig('experiment2/results.pdf')
	plt.show()	



if __name__ == '__main__':
	main()