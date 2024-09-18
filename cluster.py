import numpy as np
import pandas as pd

# Kelas KMeansClustering untuk melakukan pengelompokan menggunakan algoritma K-Means
class KMeansClustering:

    # Metode untuk menghitung jarak Euclidean antara dua titik
    @staticmethod
    def jarak_euclidean(x1, x2):
        squared_distances = [(a - b) ** 2 for a, b in zip(x1, x2)]
        euclidean_distance = sum(squared_distances) ** 0.5
        return euclidean_distance

    # Metode untuk menginisialisasi centroid secara acak
    @staticmethod
    def inisialisasi_centroid_acak(k, data):
        indices = np.random.choice(data.shape[0], size=k, replace=False)
        return data.iloc[indices].values

    # Metode untuk mengelompokkan data ke dalam cluster berdasarkan centroid
    @staticmethod
    def kelompokkan_data_ke_cluster(data, centroids):
        clusters = {}
        for i, row in data.iterrows():
            distances = [KMeansClustering.jarak_euclidean(row.values, centroid) for centroid in centroids]
            cluster = np.argmin(distances)
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(i)
        return clusters

    # Metode untuk memperbarui posisi centroid berdasarkan rata-rata data dalam setiap cluster
    @staticmethod
    def perbarui_centroid(data, clusters):
        centroids = []
        for cluster, indices in clusters.items():
            cluster_data = data.iloc[indices]
            centroid = cluster_data.mean().values
            centroids.append(centroid)
        return np.array(centroids)

    # Metode untuk menghitung nilai rata-rata dalam setiap cluster
    @staticmethod
    def calculate_average_values(data, kelompok):
        averages = []
        for cluster, indices in kelompok.items():
            cluster_data = data.iloc[indices]
            averages.append(cluster_data.mean().values)
        return np.array(averages)

    # Metode utama untuk melakukan pengelompokan data menggunakan algoritma K-Means
    def perform_clustering(self, data, k, max_iter=100):
        centroids = KMeansClustering.inisialisasi_centroid_acak(k, data)
        for _ in range(max_iter):
            clusters = KMeansClustering.kelompokkan_data_ke_cluster(data, centroids)
            new_centroids = KMeansClustering.perbarui_centroid(data, clusters)
            if np.array_equal(centroids, new_centroids):
                break
            centroids = new_centroids
        return centroids, clusters

    # Metode untuk membuat dataframe cluster dari data hasil pengelompokan
    @staticmethod
    def create_cluster_dataframe(data, kelompok):
        df = data.copy()
        df['Cluster'] = np.nan
        for cluster, indeks in kelompok.items():
            df.loc[indeks, 'Cluster'] = cluster + 1
        return df[['JPM', 'JP', 'RRP', 'RRLS', 'Cluster']]
