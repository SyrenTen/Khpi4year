# histogram method

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances


def color_histogram(image, bins=32):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # перетворення в HSV
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def load_images_and_extract_features(dataset_path, feature_extractor):
    images = []
    features = []
    labels = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            if file_name.endswith('.jpg'):
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                    features.append(feature_extractor(image))
                    labels.append(label)
                else:
                    print(f'no image {file_path}')
    return images, features, labels


# найближчі збіги
def find_top_k_matches(input_feature, dataset_features, k=10):
    distances = euclidean_distances([input_feature], dataset_features)[0]
    sorted_indices = np.argsort(distances)
    return sorted_indices[:k], distances


def visualize_results(input_image, matches, match_images):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(matches) + 1, 1)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')

    for i, idx in enumerate(matches):
        plt.subplot(1, len(matches) + 1, i + 2)
        plt.imshow(cv2.cvtColor(match_images[idx], cv2.COLOR_BGR2RGB))
        plt.title(f'Match {i + 1}')
        plt.axis('off')
    plt.show()


def main(input_image_path, dataset_path):
    print('Loading dataset')
    try:
        images, features, labels = load_images_and_extract_features(dataset_path, color_histogram)
    except ValueError as e:
        print(e)
        return

    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f'check error')
        return
    input_feature = color_histogram(input_image)

    try:
        top_matches, distances = find_top_k_matches(input_feature, features)
    except ValueError as e:
        print(e)
        return

    input_label = os.path.basename(os.path.dirname(input_image_path))

    # середня відстань та точність
    avg_distance = np.mean([distances[idx] for idx in top_matches])
    same_class_matches = sum(1 for idx in top_matches if labels[idx] == input_label)
    accuracy = same_class_matches / len(top_matches)

    print(f'Input Class: {input_label}')
    print(f'Average Distance: {avg_distance:.4f}')
    print(f'Class Accuracy: {accuracy:.2%}')

    visualize_results(input_image, top_matches, images)


dataset_path = r'D:\Universitys\Khpi\OpenCVLab\Labopencv\Lab3\flower_photos_selected'
input_image_path = r'D:\Universitys\Khpi\OpenCVLab\Labopencv\Lab3\flower_photos_selected\sunflowers\24459548_27a783feda.jpg'

main(input_image_path, dataset_path)
