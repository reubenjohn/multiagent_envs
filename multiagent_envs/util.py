import cv2
import numpy as np


def mag(a: np.ndarray):
	return np.sum(np.sqrt(a ** 2))


def cos(a: np.ndarray, b: np.ndarray):
	return np.dot(a, b) / mag(a) / mag(b)


def sin(a: np.ndarray, b: np.ndarray):
	return np.sqrt(1 - cos(a, b) ** 2)


class Point(np.ndarray):
	def __new__(cls, x_or_point, y=None):
		obj = super().__new__(cls, (2,), np.float32)
		if y is not None:
			x_or_point = np.array([x_or_point, y], np.float32, copy=True)
		np.copyto(obj, x_or_point)
		return obj

	def __hash__(self) -> int:
		return hash(self.tobytes())

	def __eq__(self, other):
		return np.all(np.equal(self, other))

	def __nq__(self, other):
		return not self.__eq__(other)


def expand_matrix(mat):
	size_1 = len(mat)
	size_0 = len(mat[0])
	mat = np.insert(mat, size_1, np.zeros([size_1 + 1], np.float16), axis=1)
	mat = np.insert(mat, size_0, np.zeros([size_0 + 1], np.float16), axis=0)
	return mat


def overlay_transparent(bg_img, img_to_overlay_t, x, y):
	"""
	@brief      Overlays a transparant PNG onto another image using CV2
	"""
	# Extract the alpha mask of the RGBA image, convert to RGB
	b, g, r, a = cv2.split(img_to_overlay_t)
	overlay_color = cv2.merge((b, g, r))

	a3 = np.repeat(np.reshape(a / 255, a.shape + (1,)), 3, 2)

	h, w, _ = overlay_color.shape
	if x < 0 or y < 0 or x > w or y > h:
		return
	roi = bg_img[y:y + h, x:x + w]
	if a3.shape != roi.shape:
		return

	roi *= (1 - a3)
	roi += overlay_color * a3