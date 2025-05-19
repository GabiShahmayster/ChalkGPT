import numpy as np
import cv2
from fontTools.unicodedata import block
from sklearn.cluster import KMeans, estimate_bandwidth, MeanShift
import matplotlib.pyplot as plt


def extract_color_histogram(image, bins=(8, 8, 8)):
    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.median(np.median(image, axis=0),axis=0)

    # hist = cv2.calcHist([image_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # cv2.normalize(hist, hist)
    # return hist.flatten()


def cluster_images(images, bandwidth=None, quantile=0.2, min_bin_freq=2):
    features = []
    for image in images:
        features.append(extract_color_histogram(image))

    features = np.array(features)

    if bandwidth is None:
        bandwidth = estimate_bandwidth(features, quantile=quantile)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=min_bin_freq)
    ms.fit(features)

    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    n_clusters = len(np.unique(labels))

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    return clusters, labels, cluster_centers, n_clusters


def visualize_clusters(images, clusters):
    # n_clusters = len(clusters)
    # fig, axs = plt.subplots(n_clusters, 1, figsize=(10, 3 * n_clusters))

    # Handle the case of a single cluster
    # if n_clusters == 1:
    #     axs = [axs]

    for i, (label, image_indices) in enumerate(clusters.items()):
        for jj,im_idx in enumerate(image_indices):
            plt.figure()
            plt.gca().imshow(cv2.cvtColor(images[im_idx], cv2.COLOR_BGR2RGB))
            plt.title(f'cluster {i},img={jj}/{len(image_indices)}')
        # axs[i].text(0, 0.5, f"Cluster {label} ({len(image_indices)} images)", fontsize=14)
        # axs[i].axis('off')
        #
        # for j, img_idx in enumerate(image_indices[:5]):
        #     if j >= 5:
        #         break
        #
        #     img = cv2.cvtColor(images[img_idx], cv2.COLOR_BGR2RGB)
        #
        #     x_pos = j * 100 + 100
        #     axs[i].imshow(img, extent=[x_pos, x_pos + 80, 10, 90])

    plt.tight_layout()
    plt.show(block=True)

