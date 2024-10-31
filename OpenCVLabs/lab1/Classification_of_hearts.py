import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def plot_image(image, text='Image'):
    if image is not None:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(text)
        plt.axis('off')
        plt.show()


image_filename = 'cleaned_hearts3_copy.png'
image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)

_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# знайти контур
contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

pixelated_hearts_contours = []
non_pixelated_contours = []
for cnt in contours:
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.015 * perimeter, True)
    pixelated_score = len(approx)

    # Threshold для піксельних сердець
    if pixelated_score > 12:
        pixelated_hearts_contours.append(cnt)
    else:
        non_pixelated_contours.append(cnt)

# піксельні
output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
for cnt in pixelated_hearts_contours:
    cv2.drawContours(output_image, [cnt], -1, (0, 0, 255), -1)  # Red color for pixelated hearts

# не-піксельних
areas = []
aspect_ratios = []
perimeters = []
compactness_values = []

for cnt in non_pixelated_contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h
    perimeter = cv2.arcLength(cnt, True)
    compactness = (perimeter ** 2) / area

    areas.append(area)
    aspect_ratios.append(aspect_ratio)
    perimeters.append(perimeter)
    compactness_values.append(compactness)

features = np.array([areas, aspect_ratios, perimeters, compactness_values]).T

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(features)

colors = [
    (0, 255, 0),  # Regular-bulky
    (255, 255, 0),  # Regular-narrow
    (255, 0, 255),  # Regular-bulky Curvy
    (0, 255, 255),  # Regular-narrow Curvy
    (0, 128, 0)
]

for i, cnt in enumerate(non_pixelated_contours):
    color = colors[labels[i]]
    cv2.drawContours(output_image, [cnt], -1, color, -1)

plot_image(output_image, text='Categorized Hearts')

heart_counts = {'Pixelated': len(pixelated_hearts_contours)}
unique, counts = np.unique(labels, return_counts=True)
for i, count in enumerate(counts):
    heart_counts[f'Heart type {i + 1}'] = count

for heart_type, count in heart_counts.items():
    print(f"{heart_type}: {count} hearts")

output_filename = 'categorized_hearts_final2.png'
cv2.imwrite(output_filename, output_image)
