import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern


def plot_images(query_image, matched_images):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(matched_images) + 1, 1)
    plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')

    for i, img in enumerate(matched_images):
        plt.subplot(1, len(matched_images) + 1, i + 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Match {i + 1}')
        plt.axis('off')

    plt.show()


def extract_advanced_features(image, kernels=None, hog_weight=0.3, color_weight=0.5, lbp_weight=0.2):
    if kernels is None:
        kernels = []
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10, frequency, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)

    # Gabor
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor_features = []
    for kernel in kernels:
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        gabor_features.append(filtered.mean())
    gabor_features = np.array(gabor_features) * hog_weight

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [32, 32, 32], [0, 180, 0, 256, 0, 256])
    hist_features = cv2.normalize(hist, hist).flatten() * color_weight

    # LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_features = lbp_hist / lbp_hist.sum()
    lbp_features = lbp_features * lbp_weight

    combined_features = np.hstack((gabor_features, hist_features, lbp_features))
    return combined_features


def prepare_dataset_with_labels(dataset_path, feature_extractor):
    images = []
    labels = []
    features = []
    kernels = []  # Gabor кернел
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10, frequency, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)

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
                    labels.append(label)
                    features.append(feature_extractor(image, kernels))
    return images, np.array(features), labels


def main():
    dataset_path = r'D:\Universitys\Khpi\OpenCVLab\Labopencv\Lab3\flower_photos_selected'
    dataset_images, dataset_features, dataset_labels = prepare_dataset_with_labels(dataset_path, extract_advanced_features)

    scaler = StandardScaler()
    dataset_features = scaler.fit_transform(dataset_features)

    knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
    knn.fit(dataset_features)

    query_image_path = r'D:\Universitys\Khpi\OpenCVLab\Labopencv\Lab3\flower_photos_selected\dandelion\142813254_20a7fd5fb6_n.jpg'
    query_image = cv2.imread(query_image_path)
    if query_image is None:
        raise FileNotFoundError(f'error no image')

    query_features = extract_advanced_features(query_image)
    query_features = scaler.transform([query_features])

    distances, indices = knn.kneighbors(query_features)
    matched_images = [dataset_images[i] for i in indices[0]]
    matched_labels = [dataset_labels[i] for i in indices[0]]

    input_label = os.path.basename(os.path.dirname(query_image_path))
    avg_distance = np.mean(distances[0])
    same_class_matches = sum(1 for label in matched_labels if label == input_label)
    accuracy = same_class_matches / len(matched_labels)

    print(f'Input Class: {input_label}')
    print(f'Average Distance: {avg_distance:.4f}')
    print(f'Class Accuracy: {accuracy:.2%}')

    plot_images(query_image, matched_images)


if __name__ == "__main__":
    main()
