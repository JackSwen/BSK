##This script is mainly contributed by Yan Zhuang
import pandas as pd
import numpy as np


def initialize_centroids(data, k):
    """
    Randomlly init centroids
    """
    np.random.seed(0)
    centroids = []
    centroids.append(np.random.choice(len(data)))
    for _ in range(1, k):
        distances = []
        for i, point in enumerate(data):
            min_distance = min(data[i][c] for c in centroids)
            distances.append(min_distance)
        probabilities = np.square(distances) / np.sum(np.square(distances))
        new_centroid_index = np.random.choice(len(data), p=probabilities)
        centroids.append(new_centroid_index)
    return np.array(centroids)

def assign_to_clusters(data, centroids):
    """
    Assignment to each clusters
    """
    clusters = [[] for _ in range(len(centroids))]
    for i in range(len(data)):
        distances = [data[i][centroid] for centroid in centroids]
        closest_centroid_index = np.argmin(distances)
        clusters[closest_centroid_index].append(i)
    return clusters

def update_centroids(data, clusters):
    """
    Centroids updating
    """
    new_centroids = []
    for cluster in clusters:
        min_distance_sum = float('inf')
        new_centroid = None
        for point_index in cluster:
            distance_sum = np.sum([data[point_index][other_point_index] for other_point_index in cluster if other_point_index != point_index])
            if distance_sum < min_distance_sum:
                min_distance_sum = distance_sum
                new_centroid = point_index
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def k_means(data, k, max_iterations=100):
    """
    modified K-means clustering method
    """
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters)
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, clusters

def average_distance_within_cluster(sample_idx, cluster, data):
    """
    Distance Calculation for points-centroid in each cluster
    """
    sum_distance = 0
    for idx in cluster:
        if idx != sample_idx:
            sum_distance += data[sample_idx][idx]
    return sum_distance / (len(cluster) - 1) if len(cluster) != 1 else sum_distance

def average_distance_to_nearest_cluster(sample_idx, cluster, clusters, data):
    """
    Distance Calculation for points-other centroids for each cluster
    """
    min_distance = float('inf')
    for other_cluster in clusters:
        if other_cluster != cluster:
            distance_to_other_cluster = np.mean([data[sample_idx][other_sample_idx] for other_sample_idx in other_cluster])
            if distance_to_other_cluster < min_distance:
                min_distance = distance_to_other_cluster
    return min_distance

def silhouette_coefficient(sample_idx, cluster, clusters, data):
    """
    Silhouestte calcuation for each points
    """
    a = average_distance_within_cluster(sample_idx, cluster, data)
    b = average_distance_to_nearest_cluster(sample_idx, cluster, clusters, data)
    return (b - a) / max(a, b)

def silhouette_score(data, clusters):
    """
    Silhouette_calculation for clusters
    """
    silhouette_scores = []
    cluster_labels = np.zeros(len(data), dtype=int)
    for idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            silhouette_scores.append(silhouette_coefficient(sample_idx, cluster, clusters[:idx] + clusters[idx+1:], data))
    return np.mean(silhouette_scores)

def find_best_k(data, max_k):
    """
    The best k value for clustering
    """
    silhouette_scores = []
    best_k = 0
    best_score = -1
    for k in range(2, max_k + 1):  # 2 / 3
        centroids, clusters = k_means(data, k)
        score = silhouette_score(data, clusters)
        silhouette_scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, silhouette_scores


if __name__ == "__main__":
    # File reading
    data = np.array(pd.read_excel('file_name.xlsx', header=None))

    max_k = 30  # Max number of clusters
    best_k,silhouette_scores = find_best_k(data, max_k)
    centroids, clusters = k_means(data, best_k)
    print(f'clusters_numbers：{best_k}\n')
    for i, cluster in enumerate(clusters):
        mean_dis = np.sum(data[centroids[i],cluster])/len(cluster)  # average distance
        Var_dis = np.sum((data[centroids[i],clu]-mean_dis)**2 for clu in cluster)/len(cluster)  # variances
        print(f"Cluster_ID{i+1}  ---  Num_points:{len(cluster)}  Centroid_ID:{centroids[i]+1}  Points_ID:{[point+1 for point in cluster]}  Average_distance:{mean_dis}  Varience:{Var_dis}\n")
