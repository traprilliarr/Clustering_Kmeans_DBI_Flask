import numpy as np
from cluster import KMeansClustering

# Kelas ClusterEvaluation untuk mengevaluasi hasil pengelompokan
class ClusterEvaluation:

    # Metode untuk menghitung indeks Davies-Bouldin
    @staticmethod
    def indeks_davies_bouldin(data, clusters, centroids):
        k = len(centroids)
        distances = []

        # Iterasi melalui setiap pasangan cluster
        for i in range(k):
            cluster_i = data.iloc[clusters[i]]
            
            for j in range(k):
                if i != j:
                    cluster_j = data.iloc[clusters[j]]
                    centroid_i = centroids[i]
                    centroid_j = centroids[j]

                    # Hitung jarak intra-cluster rata-rata
                    intra_cluster_distance_i = np.mean(
                        [KMeansClustering.jarak_euclidean(row.values, centroid_i) for _, row in cluster_i.iterrows()])
                    intra_cluster_distance_j = np.mean(
                        [KMeansClustering.jarak_euclidean(row.values, centroid_j) for _, row in cluster_j.iterrows()])

                    # Hitung jarak Davies-Bouldin antara cluster i dan j
                    distance = (intra_cluster_distance_i + intra_cluster_distance_j) / KMeansClustering.jarak_euclidean(
                        centroid_i, centroid_j)
                    distances.append(distance)

        # Hitung nilai rata-rata Davies-Bouldin Index (DBI)
        dbi = np.mean(distances) / k
        return dbi

    # Metode untuk menghitung nilai Sum of Squared Within (SSW)
    @staticmethod
    def hitung_ssw(data, clusters, centroids):
        ssw = 0

        # Iterasi melalui setiap cluster
        for cluster, indices in clusters.items():
            cluster_data = data.iloc[indices]
            
            # Iterasi melalui setiap data dalam cluster
            for _, row in cluster_data.iterrows():
                # Hitung jarak Euclidean antara data dan centroid cluster
                ssw += KMeansClustering.jarak_euclidean(row.values, centroids[cluster])
        return ssw

    # Metode untuk menghitung nilai Silhouette Score
    @staticmethod
    def silhouette_score(data, clusters, centroids):
        silhouette_values = []

        # Iterasi melalui setiap cluster
        for cluster, indices in clusters.items():
            cluster_data = data.iloc[indices]

            # Iterasi melalui setiap data dalam cluster
            for _, row in cluster_data.iterrows():
                a_i = KMeansClustering.jarak_euclidean(row.values, centroids[cluster])

                # Hitung jarak Euclidean minimum antara data dan centroid cluster lainnya
                b_i = min(
                    [KMeansClustering.jarak_euclidean(row.values, centroids[other_cluster]) for other_cluster in
                     clusters.keys() if other_cluster != cluster])

                # Hitung nilai Silhouette Score untuk data
                silhouette = (b_i - a_i) / max(a_i, b_i)
                silhouette_values.append(silhouette)

        # Hitung nilai rata-rata Silhouette Score
        return np.mean(silhouette_values)

    # Metode untuk menghitung nilai Davies-Bouldin Index (DBI)
    @staticmethod
    def calculate_dbi(data, kelompok, centroid):
        dbi = ClusterEvaluation.indeks_davies_bouldin(data, kelompok, centroid)
        return dbi
