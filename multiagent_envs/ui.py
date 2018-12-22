import cv2
import numpy as np

w, a, s, d = 119, 97, 115, 100
plus, minus = 43, 95
faster, slower = 93, 91
space = 32
enter = 13
escape = 27


class Window:
	def __init__(self, w=1280, h=720):
		self.w = w
		self.h = h
		self.img = self.reset()
		self.frame_hud_count = 0

	def reset(self):
		self.img = np.ones([self.h, self.w, 3])
		return self.img

	def display(self):
		cv2.imshow('image', self.img)
		self.frame_hud_count = 0

	def hud(self, text: str):
		self.frame_hud_count += 1
		cv2.putText(self.img, text, (0, self.frame_hud_count * 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
