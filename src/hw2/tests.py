from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque
from typing import NoReturn
from src.hw2.hw2 import DBScan, KMeans, AgglomertiveClustering, read_image, save_image, show_image


def visualize_clasters(X, labels):
    unique_labels = np.unique(labels)
    unique_colors = np.random.random((len(unique_labels), 3))
    colors = [unique_colors[l] for l in labels]
    plt.figure(figsize=(9, 9))
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()


def clusters_statistics(flatten_image, cluster_colors, cluster_labels):
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    for remove_color in range(3):
        axes_pair = axes[remove_color]
        first_color = 0 if remove_color != 0 else 2
        second_color = 1 if remove_color != 1 else 2
        axes_pair[0].scatter([p[first_color] for p in flatten_image], [p[second_color] for p in flatten_image], c=flatten_image, marker='.')
        axes_pair[1].scatter([p[first_color] for p in flatten_image], [p[second_color] for p in flatten_image], c=[cluster_colors[c] for c in cluster_labels], marker='.')
        for a in axes_pair:
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
    plt.show()


# X_1, true_labels = make_blobs(400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]])
# # visualize_clasters(X_1, true_labels)
# X_2, true_labels = make_moons(400, noise=0.075)
# # visualize_clasters(X_2, true_labels)


# def clusterize_image(image, n_clusters=10):
#     km = KMeans(n_clusters=n_clusters, init='k-means++')
#     img_flat = image.reshape(-1, image.shape[-1])
#     km.fit(img_flat)
#     clusters = km.predict(img_flat)
#
#     recolored = img_flat
#     cluster_colors = []
#     for i in range(n_clusters):
#         col = np.mean(img_flat[clusters == i], axis=0)
#         recolored[clusters == i] = col
#         cluster_colors.append(col)
#     recolored = recolored.reshape(image.shape)
#     # clusters_statistics(image.reshape(-1, 3) / 255, np.array(cluster_colors) / 255, clusters)  # Very slow (:
#     return recolored
#
#
# img = read_image('img2.jpg')
# show_image(img)
# img_clustered = clusterize_image(img, n_clusters=40)
# show_image(img_clustered)

