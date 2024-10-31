import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_image(image, text='Image'):
    if image is not None:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(text)
        plt.show()
    else:
        print(f'Error')


# Завантаження та зміна розміру зображення
image_filename = 'hearts1.png'
img_bgr = cv2.imread(image_filename)
img_bgr = cv2.resize(img_bgr, (img_bgr.shape[1] // 2, img_bgr.shape[0] // 2))

# градації сірого
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
plot_image(img_gray, text='Initial Image (Grayscale)')

# адапт гаус інверс
img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 55, 5)

# Збереження
output_filename = 'cleaned_hearts1.png'
cv2.imwrite(output_filename, img_thresh)
print(f'Cleaned image saved as {output_filename}')
plot_image(cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR), text='Adaptive Gaussian Threshold')

# друге зображення
image_filename = 'hearts2.png'
img_bgr = cv2.imread(image_filename)
img_bgr = cv2.resize(img_bgr, (img_bgr.shape[1] // 2, img_bgr.shape[0] // 2))

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
plot_image(img_gray, text='Grayscale')

img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 55, 5)

output_filename = 'cleaned_hearts2.png'
cv2.imwrite(output_filename, img_thresh)
plot_image(cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR), text='Invert Adaptive Gaussian Threshold')

# третє зображення
image_filename = 'hearts3.png'
img_init = cv2.imread(image_filename)

green_channel = img_init[:, :, 1]  # Green channel

# використання Gaussian blur
smooth_green_channel = cv2.GaussianBlur(green_channel, (3, 3), 0)

inverted_hearts = cv2.bitwise_not(smooth_green_channel)  # інверт

# binary threshold
_, binary_hearts = cv2.threshold(inverted_hearts, 128, 255, cv2.THRESH_BINARY)

plot_image(binary_hearts, text='Binary Inverted')
output_filename = 'cleaned_hearts3.png'
cv2.imwrite(output_filename, binary_hearts)

# 4 зображення
image_filename = 'hearts4.png'
img_init = cv2.imread(image_filename)

img_hsv = cv2.cvtColor(img_init, cv2.COLOR_BGR2HSV) # Convert to HSV
channel_image = img_hsv[:, :, 0]

# адаптивне thresholding
_, binary_image = cv2.threshold(channel_image, 120, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # morphologic
cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_image, connectivity=8)
output = np.zeros_like(cleaned_image)

min_area = 100
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        output[labels == i] = 255

plot_image(output, text='Cleaned hearts 4')
cv2.imwrite('cleaned_hearts4.png', output)

# 5 зображення
image_filename = 'hearts5.png'
original_image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
plot_image(original_image, text='Initial Image')

# histogram
hist_values = cv2.calcHist([original_image], [0], None, [256], [0, 256])

cumulative_hist = np.cumsum(hist_values)  # cumulative histogram

# histogram equalization
equalized_image = cv2.equalizeHist(original_image)
plot_image(equalized_image, text='Equalized Image')

output_filename = 'cleaned_hearts5.png'
cv2.imwrite(output_filename, equalized_image)
