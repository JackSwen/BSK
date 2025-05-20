import pandas as pd
from openpyxl import load_workbook
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import json
import os
import os.path as op


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
    k_list = list(range(2, max_k + 1))
    best_k = 0
    best_score = -1
    for k in k_list:  # 2 / 3
        centroids, clusters = k_means(data, k)
        score = silhouette_score(data, clusters)    # 轮廓系数
        silhouette_scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, k_list, silhouette_scores


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
    best_k, k_list, silhouette_scores = find_best_k(data, max_k)
    centroids, clusters = k_means(data, best_k)

    excel_save_dict = {
        'seed': seed,
        'best_k': best_k,
        '最佳轮廓系数': max(silhouette_scores),
        'k_list': k_list,
        '轮廓系数': silhouette_scores,
        'clusters': {
            '序号': [],
            '数据点数': [],
            '中心点索引': [],
            '距中心点平均距离': [],
            '距中心点距离标准差': [],
            '点集': []
        }
    }
    # best_k_clusters = {}

    print(f'num_clusters：{best_k}')
    for i, cluster in enumerate(clusters):
        mean_dis = np.sum(data[centroids[i], cluster]) / len(cluster)  # average_distance
        Var_dis = np.sum((data[centroids[i], clu] - mean_dis) ** 2 for clu in cluster) / len(cluster)  # var
        print(
            f"Cluster_ID{i + 1}  ---  Num_points:{len(cluster)}  Centroids:{centroids[i] + 1}  Average_distance:{mean_dis}  Var_dis:{Var_dis}  Points:{[point + 1 for point in cluster]}")
        excel_save_dict['clusters']['序号'].append(f'{i+1}')
        excel_save_dict['clusters']['数据点数'].append(len(cluster))
        excel_save_dict['clusters']['中心点索引'].append(int(centroids[i] + 1))
        excel_save_dict['clusters']['距中心点平均距离'].append(round(float(mean_dis), 6))
        excel_save_dict['clusters']['距中心点距离标准差'].append(round(float(Var_dis), 6))
        excel_save_dict['clusters']['点集'].append([point + 1 for point in cluster])
    return excel_save_dict


def save_to_excel(data, excel_path,
                  sheet1_name='整体运行情况',
                  sheet2_name='最佳轮廓系数下的聚类数据',
                  rewrite_all=False,
                  rewrite_sheet1=False,
                  rewrite_sheet2=False
                  ):
    """
    保存数据到Excel，支持追加/覆写模式
    - 默认为追加模式
    """
    # best_k聚类数据
    clusters_data = data.pop('best_clusters') if 'best_clusters' in data else {}
    main_data = data

    # 判断文件是否存在
    file_exists = os.path.exists(excel_path) and os.path.getsize(excel_path) > 0
    mode = 'w' if not file_exists else 'a'
    sheet_exists_mode = 'replace' if mode == 'a' else None

    # 初始化Excel写入器
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode=mode, if_sheet_exists=sheet_exists_mode) as writer:
        # === 处理主数据 (Sheet1) ===
        if rewrite_all or rewrite_sheet1:
            main_df = pd.DataFrame(main_data)
        else:
            main_df = pd.read_excel(excel_path, sheet_name=sheet1_name)
            new_main_df = pd.DataFrame(main_data)
            main_df = pd.concat([main_df, new_main_df], ignore_index=True)
        main_df.to_excel(writer, sheet_name=sheet1_name, index=False)

        # === 处理聚类数据 (Sheet2) ===
        if clusters_data:
            cluster_df = pd.DataFrame(clusters_data)

            if rewrite_all or rewrite_sheet2:
                new_cluster_df = cluster_df
            else:
                try:
                    existing_cluster_df = pd.read_excel(excel_path, sheet_name=sheet2_name)
                    new_cluster_df = pd.concat([existing_cluster_df, cluster_df], ignore_index=True)
                except (FileNotFoundError, ValueError):
                    new_cluster_df = cluster_df

            new_cluster_df.to_excel(writer, sheet_name=sheet2_name, index=False)
    print(f"数据已成功保存至：{excel_path}")


def make_dir(path, **kwargs):
    if not op.exists(path):
        os.makedirs(path, **kwargs)
    return path


@measure_time
def main():
    # 加载数据
    # data = np.loadtxt('G3BP1_rmsd_matrix.txt')        # reading txt files
    data = np.loadtxt('capr1_rmsd_matrix.txt')

    # 自定义保存路径, 指定excel各参数
    save_folder = make_dir('./result_20250520')       # 结果保存的文件夹
    excel_path = op.join(save_folder, 'test-20250520.xlsx')      # excel保存路径
    kwargs = {
        'sheet1_name': '整体运行情况',
        'sheet2_name': '最佳轮廓系数下的聚类数据',
        'rewrite_all': True,
        'rewrite_sheet1': False,
        'rewrite_sheet2': True,
    }
    final_excel_save_dict = {
        'seed': [],
        'best_k': [],
        '最佳轮廓系数': [],
        'k_list': [],
        '轮廓系数': [],
        'best_clusters': {}
    }

    # 程序运行
    best_score = -1
    best_seed = -1
    max_k = 3               # 最大的k值
    min_seed, max_seed = 0, 1  # seed范围[min_seed, max_seed]
    for seed in range(min_seed, max_seed+1):
        excel_save_dict = process_seed(seed, data, max_k)
        clusters = excel_save_dict.pop('clusters')
        score = excel_save_dict['最佳轮廓系数']
        if score > best_score:
            best_score = score
            best_seed = seed
            final_excel_save_dict['best_clusters'] = clusters
        for k, v in excel_save_dict.items():
            final_excel_save_dict[k].append(v)

    kwargs['sheet2_name'] = f'({best_seed}, {best_score:.6f})_最佳轮廓系数下的聚类数据'
    save_to_excel(data=final_excel_save_dict, excel_path=excel_path, **kwargs)


if __name__ == "__main__":
    import time

    main()
