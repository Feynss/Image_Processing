import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, mean_squared_error
import copy


# Зашумить изображение при помощи шума гаусса, постоянного шума.
# Протестировать медианный фильтр, фильтр гаусса, билатериальный фильтр,
# фильтр нелокальных средних с различными параметрами.
# Выяснить, какой фильтр показал лучший результат фильтрации шума.


# 1
image = cv2.imread('img.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray, cmap="gray")
plt.show()

mean = 0
stddev = 100
noise_gauss = np.zeros(image_gray.shape, np.uint8)
cv2.randn(noise_gauss, mean, stddev)
plt.imshow(noise_gauss, cmap="gray")
plt.show()

image_noise_gauss = cv2.add(image_gray,noise_gauss)
plt.imshow(image_noise_gauss, cmap="gray")
plt.show()

noise =  np.random.randint(0, 101, size = (image_gray.shape[0], image_gray.shape[1]), dtype=int)
zeros_pixel = np.where(noise == 0)
ones_pixel = np.where(noise == 100)
bg_image = np.ones(image_gray.shape, np.uint8) * 128
bg_image[zeros_pixel] = 0
bg_image[ones_pixel] = 255
plt.imshow(bg_image, cmap="gray")
plt.show()

image_noise_constant = cv2.add(image_gray,bg_image)
plt.imshow(image_noise_constant, cmap="gray")
plt.show()


# 2
image_sp = copy.deepcopy(image_gray)

image_sp[zeros_pixel] = 0
image_sp[ones_pixel] = 255

mse_sp = mean_squared_error(image_gray, image_sp)
(ssim_sp, diff) = structural_similarity(image_gray, image_sp, full=True)
print("Изначальный: ", mse_sp, ssim_sp)

plt.imshow(image_sp, cmap="gray")
plt.show()

image_sp_median = cv2.medianBlur(image_sp, 3)

mse_sp_median = mean_squared_error(image_gray, image_sp_median)
(ssim_sp_median, diff) = structural_similarity(image_gray, image_sp_median, full=True)
print("Медианный: ", mse_sp_median, ssim_sp_median)

plt.imshow(image_sp_median, cmap="gray")
plt.show()


image_gauss_gauss = cv2.GaussianBlur(image_noise_gauss,(5,5),0)
image_gauss_bilat = cv2.bilateralFilter(image_noise_gauss,9,75,75)
image_gauss_nlm = cv2.fastNlMeansDenoising(image_noise_gauss, h = 20)

mse_geom = mean_squared_error(image_gray, image_gauss_gauss)
(ssim_geom, diff) = structural_similarity(image_gray, image_gauss_gauss, full=True)
print("Гаусс: ", mse_geom, ssim_geom)
plt.imshow(image_gauss_gauss, cmap="gray")
plt.show()

mse_geom = mean_squared_error(image_gray, image_gauss_bilat)
(ssim_geom, diff) = structural_similarity(image_gray, image_gauss_bilat, full=True)
print("Билатериальный: ", mse_geom, ssim_geom)
plt.imshow(image_gauss_bilat, cmap="gray")
plt.show()

mse_geom = mean_squared_error(image_gray, image_gauss_nlm)
(ssim_geom, diff) = structural_similarity(image_gray, image_gauss_nlm, full=True)
print("Среднее: ", mse_geom, ssim_geom)
plt.imshow(image_gauss_nlm, cmap="gray")
plt.show()


# 2 (другие параметры)
image_sp = copy.deepcopy(image_gray)

image_sp[zeros_pixel] = 0
image_sp[ones_pixel] = 255

mse_sp = mean_squared_error(image_gray, image_sp)
(ssim_sp, diff) = structural_similarity(image_gray, image_sp, full=True)
print("Изначальный: ", mse_sp, ssim_sp)

plt.imshow(image_sp, cmap="gray")
plt.show()

image_sp_median = cv2.medianBlur(image_sp, 9)

mse_sp_median = mean_squared_error(image_gray, image_sp_median)
(ssim_sp_median, diff) = structural_similarity(image_gray, image_sp_median, full=True)
print("Медианный: ", mse_sp_median, ssim_sp_median)

plt.imshow(image_sp_median, cmap="gray")
plt.show()


image_gauss_gauss = cv2.GaussianBlur(image_noise_gauss,(3,3),0)
image_gauss_bilat = cv2.bilateralFilter(image_noise_gauss,12,150,150)
image_gauss_nlm = cv2.fastNlMeansDenoising(image_noise_gauss, h = 80)

mse_geom = mean_squared_error(image_gray, image_gauss_gauss)
(ssim_geom, diff) = structural_similarity(image_gray, image_gauss_gauss, full=True)
print("Гаусс: ", mse_geom, ssim_geom)
plt.imshow(image_gauss_gauss, cmap="gray")
plt.show()

mse_geom = mean_squared_error(image_gray, image_gauss_bilat)
(ssim_geom, diff) = structural_similarity(image_gray, image_gauss_bilat, full=True)
print("Билатериальный: ", mse_geom, ssim_geom)
plt.imshow(image_gauss_bilat, cmap="gray")
plt.show()

mse_geom = mean_squared_error(image_gray, image_gauss_nlm)
(ssim_geom, diff) = structural_similarity(image_gray, image_gauss_nlm, full=True)
print("Среднее: ", mse_geom, ssim_geom)
plt.imshow(image_gauss_nlm, cmap="gray")
plt.show()

# При kernel size = 3, лучше всех справляется медианный фильтр,
# когда остальные недостаточно убирают шум или
# чересчур смазывают изображение.
