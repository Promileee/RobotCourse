import numpy as np
import cv2

def binary_manual(gray_image, threshold=128):
	binary_image = np.zeros_like(gray_image)
	binary_image[gray_image >= threshold] = 255
	return binary_image


def gray_level_manual(gray_image, levels):
	step = 256 // levels
	quantized_image = (gray_image // step) * step
	return quantized_image.astype(np.uint8)


def gamma_transform(gray_image, gamma_value=0.6):
	normalized = gray_image.astype(np.float32) / 255.0
	transformed = 255.0 * np.power(normalized, gamma_value)
	return np.clip(transformed, 0, 255).astype(np.uint8)


def log_transform(gray_image):
	img = gray_image.astype(np.float32)
	c = 255.0 / np.log(1.0 + 255.0)
	transformed = c * np.log(1.0 + img)
	return np.clip(transformed, 0, 255).astype(np.uint8)


def histogram_equalization_manual(gray_image):
	histogram = np.bincount(gray_image.ravel(), minlength=256)
	cdf = histogram.cumsum()
	nonzero_cdf = cdf[cdf > 0]

	if nonzero_cdf.size == 0:
		return gray_image.copy()

	cdf_min = nonzero_cdf[0]
	cdf_max = nonzero_cdf[-1]

	if cdf_max == cdf_min:
		return gray_image.copy()

	lut = (cdf - cdf_min) * 255.0 / (cdf_max - cdf_min)
	lut = np.clip(lut, 0, 255).astype(np.uint8)
	return lut[gray_image]


def otsu_threshold(gray_image):
	hist = np.bincount(gray_image.ravel(), minlength=256).astype(np.float32)
	total = hist.sum()
	if total == 0:
		return 0

	sum_total = np.dot(np.arange(256), hist)

	sumB = 0.0
	wB = 0.0
	max_var = 0.0
	threshold = 0

	for t in range(256):
		wB += hist[t]
		if wB == 0:
			continue
		wF = total - wB
		if wF == 0:
			break
		sumB += t * hist[t]
		mB = sumB / wB
		mF = (sum_total - sumB) / wF
		var_between = wB * wF * (mB - mF) * (mB - mF)
		if var_between > max_var:
			max_var = var_between
			threshold = t

	return int(threshold)


def binary_otsu(gray_image):
	t = otsu_threshold(gray_image)
	out = np.zeros_like(gray_image)
	out[gray_image >= t] = 255
	return out


def clahe_manual(gray_image, tile_grid_size=(8, 8), clip_limit=2.0):
	h, w = gray_image.shape
	nr, nc = tile_grid_size
	tile_h = int(np.ceil(h / nr))
	tile_w = int(np.ceil(w / nc))

	# Precompute LUTs for each tile
	luts = [[None for _ in range(nc)] for __ in range(nr)]
	for r in range(nr):
		for c in range(nc):
			y0 = r * tile_h
			x0 = c * tile_w
			y1 = min(y0 + tile_h, h)
			x1 = min(x0 + tile_w, w)
			tile = gray_image[y0:y1, x0:x1]
			hist = np.bincount(tile.ravel(), minlength=256).astype(np.float32)

			# clip limit in pixels
			max_clip = max(1.0, clip_limit * (tile.size) / 256.0)
			excess = hist - max_clip
			excess[excess < 0] = 0
			clipped = hist.copy()
			clipped[clipped > max_clip] = max_clip
			redistribute = excess.sum()
			if redistribute > 0:
				clipped += redistribute / 256.0

			cdf = clipped.cumsum()
			cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0.0
			denom = (cdf[-1] - cdf_min) if (cdf[-1] - cdf_min) > 0 else 1.0
			lut = ((cdf - cdf_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
			luts[r][c] = lut

	# Interpolate mapped values for each pixel
	out = np.zeros_like(gray_image, dtype=np.uint8)
	for y in range(h):
		for x in range(w):
			# tile indices
			ty = min(nr - 1, y // tile_h)
			tx = min(nc - 1, x // tile_w)

			# relative position inside tile for interpolation
			y0 = ty * tile_h
			x0 = tx * tile_w
			dy = (y - y0) / tile_h if tile_h > 0 else 0
			dx = (x - x0) / tile_w if tile_w > 0 else 0

			# neighbors for bilinear
			ty1 = min(nr - 1, ty + 1)
			tx1 = min(nc - 1, tx + 1)

			val = int(gray_image[y, x])
			v00 = luts[ty][tx][val]
			v10 = luts[ty1][tx][val]
			v01 = luts[ty][tx1][val]
			v11 = luts[ty1][tx1][val]

			# bilinear interpolation
			w00 = (1 - dy) * (1 - dx)
			w10 = dy * (1 - dx)
			w01 = (1 - dy) * dx
			w11 = dy * dx

			out[y, x] = np.clip(v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11, 0, 255)

	return out


def histogram_counts(gray_image):
	"""Return histogram counts for 0..255 as a 1D numpy array."""
	hist = np.bincount(gray_image.ravel(), minlength=256).astype(np.int32)
	return hist



def make_panel(image, title, size=(320, 240), is_gray=False):
	if is_gray:
		panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	else:
		panel = image.copy()

	panel = cv2.resize(panel, size)
	cv2.putText(panel, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
	return panel


def build_dashboard(panels):
	row1 = cv2.hconcat(panels[:4])
	row2 = cv2.hconcat(panels[4:])
	return cv2.vconcat([row1, row2])