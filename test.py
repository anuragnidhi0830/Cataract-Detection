from skimage.io import imread
from skimage.filter import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2
import cmath

'''
def uniformity(arr):
	length_of_array = len(arr)
	for i in range(length_of_array):

'''

img = cv2.imread('normal1.jpg', 0)


cataract_image = imread("cataract.jpg", as_grey=True)
normal_image = imread("normal1.jpg",as_grey=True)


gray_normal_image = normal_image * 255
gray_cataract_image = cataract_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(gray_cataract_image, cmap="gray")
threshold_value = threshold_otsu(gray_cataract_image)
binary_cataract_image = gray_cataract_image > threshold_value
ax2.imshow(binary_cataract_image, cmap="gray")

plt.show()

histr = cv2.calcHist([img],[0],None,[256],[0,256])
#uniformity_image = uniformity(histr)
#plt.plot(histr)
#plt.xlim([0,256])
plt.hist(img.ravel(),256,[0,256])
plt.show()

