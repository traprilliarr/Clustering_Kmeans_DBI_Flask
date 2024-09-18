# Impor modul yang diperlukan
from flask import Flask, render_template, request, redirect, session
import pandas as pd
import numpy as np
import pickle
from cluster import KMeansClustering
from eval import ClusterEvaluation

# Inisialisasi aplikasi Flask dan tetapkan kunci rahasia untuk manajemen sesi
app = Flask(__name__)
app.secret_key = 'kunci_rahasia_yulya'

# Muat model pengelompokan yang telah dilatih sebelumnya dari file yang disimpan
with open('model.sav', 'rb') as file:
    model_data = pickle.load(file)
    centroid = model_data['centroid']
    k = model_data['k']

# Tentukan kelas pengontrol utama untuk menangani rute dan metode
class AppController:

    # Rute untuk halaman utama (unggah file)
    @app.route('/')
    def halaman_unggah():
        controller = AppController()  # Buat instance dari kelas
        return controller.metode_halaman_unggah()

    # Rute untuk halaman evaluasi
    @app.route('/eval')
    def halaman_evaluasi():
        controller = AppController()  # Buat instance dari kelas
        return controller.metode_halaman_evaluasi()

    # Rute untuk menangani unggahan file dan pengelompokan
    @app.route('/cluster', methods=['POST'])
    def unggah_file():
        controller = AppController()  # Buat instance dari kelas
        return controller.metode_unggah_file()

    # Metode untuk merender halaman utama unggah file
    def metode_halaman_unggah(self):
        return render_template('main.html')

    # Metode untuk merender halaman evaluasi dengan nilai DBI dan informasi kelompok
    def metode_halaman_evaluasi(self):
        # Ambil nilai DBI dan k dari sesi Flask
        dbi = session.get('dbi', None)
        k = session.get('k', None)
        # Hapus nilai DBI dari sesi Flask setelah diambil
        session.pop('k', None)
        session.pop('dbi', None)
        return render_template('eval.html', dbi=dbi, k=k)

    # Metode untuk menangani unggahan file, melakukan pengelompokan, dan menampilkan hasil
    def metode_unggah_file(self):
        # Periksa apakah file disertakan dalam permintaan
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # Periksa apakah nama file kosong
        if file.filename == '':
            return redirect(request.url)

        # Jika file valid ada, baca file dan lakukan pengelompokan
        if file:
            data = pd.read_csv(file)
            num_clusters = int(request.form.get('num_clusters', 3))
            kmeans = KMeansClustering()
            centroid, kelompok = kmeans.perform_clustering(data[['JPM', 'JP', 'RRP', 'RRLS']], num_clusters)
            dbi = ClusterEvaluation.calculate_dbi(data[['JPM', 'JP', 'RRP', 'RRLS']], kelompok, centroid)
            averages = KMeansClustering.calculate_average_values(data[['JPM', 'JP', 'RRP', 'RRLS']], kelompok)

            cluster_data = KMeansClustering.create_cluster_dataframe(data, kelompok)
            result_cluster_data = pd.concat([data['Kabupaten/Kota'], cluster_data], axis=1)

            # Simpan nilai DBI dan k dalam sesi Flask untuk digunakan di halaman evaluasi
            session['dbi'] = dbi
            session['k'] = num_clusters

            # Hitung dan tampilkan statistik kelompok
            cluster_counts = cluster_data['Cluster'].value_counts().sort_index()

            return render_template('cluster.html', data=data, cluster_data=result_cluster_data,
                                   dbi=dbi, averages=averages, k=num_clusters, cluster_counts=cluster_counts)

if __name__ == '__main__':
    app.run(debug=True)
