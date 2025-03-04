import pandas as pd
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import json
import os


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)
        end_time = time.time()  
        elapsed_time = end_time - start_time  
        #transfer_h_min_sec 
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print(f"Running time: {hours}h {minutes}min {seconds}s")
        return result

    return wrapper


def initialize_centroids(data, k):
    """
    Random initialization
    """
    # np.random.seed(0)
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
    Assignment to clusters
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

def k_means(data, k, max_iterations=500):
    """
    BSK-means clustering 
    """
    centroids = initialize_centroids(data, k)
    for i in range(max_iterations):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters)
        if np.array_equal(new_centroids, centroids):
            # print(f'centroids remain the same, iterations: {i}, k: {k}')
            break
        centroids = new_centroids
    return centroids, clusters

def average_distance_within_cluster(sample_idx, cluster, data):
    """
    Distance within clusters
    """
    sum_distance = 0
    for idx in cluster:
        if idx != sample_idx:
            sum_distance += data[sample_idx][idx]
    return sum_distance / (len(cluster) - 1) if len(cluster) != 1 else sum_distance

def average_distance_to_nearest_cluster(sample_idx, cluster, clusters, data):
    """
    Distance with nearest cluster
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
    Silhouette_calculation
    """
    a = average_distance_within_cluster(sample_idx, cluster, data)
    b = average_distance_to_nearest_cluster(sample_idx, cluster, clusters, data)
    return (b - a) / max(a, b)

def silhouette_score(data, clusters):
    """
    Overall silhouette_score (Best:1.00)
    """
    silhouette_scores = []
    cluster_labels = np.zeros(len(data), dtype=int)
    for idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            silhouette_scores.append(silhouette_coefficient(sample_idx, cluster, clusters[:idx] + clusters[idx+1:], data))
    return np.mean(silhouette_scores)

def find_best_k(data, max_k):
    """
    K value: the number for the best clustering
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


def plot_clusters(data, clusters, centroids):
    """
    clusters results for each clustering
    """
    plt.figure(figsize=(8, 6))

    for i, cluster in enumerate(clusters):
        cluster_points = data[cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')

    # centroids ploting
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')

    plt.title('K-means Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


def read_json(json_path):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
        return json_file


def write_json(json_path, save_file, indent=4):
    with open(json_path, 'w') as f:
        json.dump(save_file, f, indent=indent)


def process_seed(seed, data, max_k):
    np.random.seed(seed)
    best_k, silhouette_scores = find_best_k(data, max_k)
    centroids, clusters = k_means(data, best_k)

    result = {
        'seed': seed,
        'best_k': best_k,
        'clusters': {}
    }

    print(f'num_clusters：{best_k}')
    for i, cluster in enumerate(clusters):
        mean_dis = np.sum(data[centroids[i], cluster]) / len(cluster)  # average_distance
        Var_dis = np.sum((data[centroids[i], clu] - mean_dis) ** 2 for clu in cluster) / len(cluster)  # var
        print(
            f"Cluster_ID{i + 1}  ---  Num_points:{len(cluster)}  Centroids:{centroids[i] + 1}  Points:{[point + 1 for point in cluster]}  Average_distance:{mean_dis}  Var_dis:{Var_dis}")
        result['clusters'][f'group_{i + 1}'] = {
            'num': len(cluster),
            'centroid': int(centroids[i] + 1),
            'mean_dis': float(round(mean_dis, 6)),
            'var_dis': float(round(Var_dis, 6)),
            'point_set': [point + 1 for point in cluster],
        }
    write_json(f'./result/seed_{seed}-max_k_30.json', result)


@measure_time
def main():
    # reading txt files
    data = np.loadtxt('G3BP1_rmsd_matrix.txt')
    if not os.path.exists('result'):
        os.makedirs('result')
    max_k = 10  # max number of clusters

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_seed, seed, data, max_k) for seed in range(0, 500)]
        #wait for all done
        concurrent.futures.wait(futures)



if __name__ == "__main__":
    import time

    main()
