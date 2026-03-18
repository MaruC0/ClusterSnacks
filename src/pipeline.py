from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np

from features import *
from clustering import *
from utils import *
from constant import PATH_OUTPUT, PATH_DATA

def pipeline():
    print("##### Chargement des données ######")

    images, labels_true, img_paths = load_images(path_data=PATH_DATA)
   
    print("\n\n ##### Extraction de Features ######")
    print("- calcul features hog...")
    descriptors_hog = compute_hog_descriptors(images)
    print("- calcul features Histogram...")
    descriptors_hist = compute_gray_histograms(images)
    print("- calcul features LBP...")
    descriptors_lbp = compute_lbp_descriptors(images)


    print("\n\n ##### Clustering ######")
    number_cluster = 20
    
    kmeans_hog = KMeans(n_clusters=number_cluster, random_state=42)
    kmeans_hist = KMeans(n_clusters=number_cluster, random_state=42)
    kmeans_lbp = KMeans(n_clusters=number_cluster, random_state=42)
    
    spectral_hog = SpectralClustering(
        n_clusters=number_cluster,
        affinity='nearest_neighbors',
        n_neighbors=15,
        assign_labels='cluster_qr',
        pca_components=64,
        random_state=42
    )
    spectral_hist = SpectralClustering(
        n_clusters=number_cluster,
        affinity='nearest_neighbors',
        n_neighbors=20,
        assign_labels='cluster_qr',
        pca_components=32,
        random_state=42
    )
    spectral_lbp = SpectralClustering(
        n_clusters=number_cluster,
        affinity='nearest_neighbors',
        n_neighbors=20,
        assign_labels='cluster_qr',
        pca_components=16,
        random_state=42
    )

    print("- calcul kmeans avec features HOG ...")
    kmeans_hog.fit(np.array(descriptors_hog))
    print("- calcul kmeans avec features Histogram...")
    kmeans_hist.fit(np.array(descriptors_hist))
    print("- calcul kmeans avec features LBP...")
    kmeans_lbp.fit(np.array(descriptors_lbp))
    
    print("- calcul spectral clustering avec features HOG ...")
    spectral_hog.fit(np.array(descriptors_hog))
    print("- calcul spectral clustering avec features Histogram...")
    spectral_hist.fit(np.array(descriptors_hist))
    print("- calcul spectral clustering avec features LBP...")
    spectral_lbp.fit(np.array(descriptors_lbp))

    print("\n=== KMeans ===")
    kmeans_hog_unique, kmeans_hog_counts = np.unique(kmeans_hog.labels_, return_counts=True)
    kmeans_hist_unique, kmeans_hist_counts = np.unique(kmeans_hist.labels_, return_counts=True)
    kmeans_lbp_unique, kmeans_lbp_counts = np.unique(kmeans_lbp.labels_, return_counts=True)
    print(f"- HOG: {len(kmeans_hog_unique)} clusters, distribution={dict(zip(kmeans_hog_unique.tolist(), kmeans_hog_counts.tolist()))}")
    print(f"- HIST: {len(kmeans_hist_unique)} clusters, distribution={dict(zip(kmeans_hist_unique.tolist(), kmeans_hist_counts.tolist()))}")
    print(f"- LBP: {len(kmeans_lbp_unique)} clusters, distribution={dict(zip(kmeans_lbp_unique.tolist(), kmeans_lbp_counts.tolist()))}")
    
    print("\n=== Spectral Clustering ===")
    hog_unique, hog_counts = np.unique(spectral_hog.labels_, return_counts=True)
    hist_unique, hist_counts = np.unique(spectral_hist.labels_, return_counts=True)
    lbp_unique, lbp_counts = np.unique(spectral_lbp.labels_, return_counts=True)
    print(f"- HOG: {len(hog_unique)} clusters distincts, distribution={dict(zip(hog_unique.tolist(), hog_counts.tolist()))}")
    print(f"- HIST: {len(hist_unique)} clusters distincts, distribution={dict(zip(hist_unique.tolist(), hist_counts.tolist()))}")
    print(f"- LBP: {len(lbp_unique)} clusters distincts, distribution={dict(zip(lbp_unique.tolist(), lbp_counts.tolist()))}")
    print(f"- n_neighbors utilisé (HOG): {spectral_hog.used_n_neighbors_}")
    print(f"- n_neighbors utilisé (HIST): {spectral_hist.used_n_neighbors_}")
    print(f"- n_neighbors utilisé (LBP): {spectral_lbp.used_n_neighbors_}")

    print("\n\n##### Résultats KMeans ######")
    metric_kmeans_hist = show_metric(labels_true, kmeans_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", name_model="kmeans", bool_return=True)
    print("\n")
    metric_kmeans_hog = show_metric(labels_true, kmeans_hog.labels_, descriptors_hog, bool_show=True, name_descriptor="HOG", name_model="kmeans", bool_return=True)
    print("\n")
    metric_kmeans_lbp = show_metric(labels_true, kmeans_lbp.labels_, descriptors_lbp, bool_show=True, name_descriptor="LBP", name_model="kmeans", bool_return=True)
    
    print("\n\n##### Résultats Spectral Clustering ######")
    metric_hist = show_metric(labels_true, spectral_hist.labels_, descriptors_hist, bool_show=True, name_descriptor="HISTOGRAM", name_model="spectral", bool_return=True)
    print("\n")
    metric_hog = show_metric(labels_true, spectral_hog.labels_, descriptors_hog, bool_show=True, name_descriptor="HOG", name_model="spectral", bool_return=True)
    print("\n")
    metric_lbp = show_metric(labels_true, spectral_lbp.labels_, descriptors_lbp, bool_show=True, name_descriptor="LBP", name_model="spectral", bool_return=True)


    print("- export des données vers le dashboard")
    # conversion des données vers le format du dashboard
    list_dict = [metric_kmeans_hist, metric_kmeans_hog, metric_kmeans_lbp, metric_hist, metric_hog, metric_lbp]
    df_metric = pd.DataFrame(list_dict)
    
    # Normalisation des données
    scaler = StandardScaler()
    descriptors_hist_norm = scaler.fit_transform(descriptors_hist)
    descriptors_hog_norm = scaler.fit_transform(descriptors_hog)
    descriptors_lbp_norm = scaler.fit_transform(descriptors_lbp)

    #conversion vers un format 3D pour la visualisation
    x_3d_hist = conversion_3d(descriptors_hist_norm)
    x_3d_hog = conversion_3d(descriptors_hog_norm)
    x_3d_lbp = conversion_3d(descriptors_lbp_norm)

    # création des dataframe pour la sauvegarde des données pour la visualisation (KMeans)
    df_hist_kmeans = create_df_to_export(x_3d_hist, labels_true, kmeans_hist.labels_)
    df_hog_kmeans = create_df_to_export(x_3d_hog, labels_true, kmeans_hog.labels_)
    df_lbp_kmeans = create_df_to_export(x_3d_lbp, labels_true, kmeans_lbp.labels_)
    
    # création des dataframe pour la sauvegarde des données pour la visualisation (Spectral)
    df_hist = create_df_to_export(x_3d_hist, labels_true, spectral_hist.labels_)
    df_hog = create_df_to_export(x_3d_hog, labels_true, spectral_hog.labels_)
    df_lbp = create_df_to_export(x_3d_lbp, labels_true, spectral_lbp.labels_)

    # Vérifie si le dossier existe déjà
    if not os.path.exists(PATH_OUTPUT):
        # Crée le dossier
        os.makedirs(PATH_OUTPUT)

    # sauvegarde des données KMeans
    df_hist_kmeans.to_excel(PATH_OUTPUT+"/save_clustering_hist_kmeans.xlsx")
    df_hog_kmeans.to_excel(PATH_OUTPUT+"/save_clustering_hog_kmeans.xlsx")
    df_lbp_kmeans.to_excel(PATH_OUTPUT+"/save_clustering_lbp_kmeans.xlsx")
    
    # sauvegarde des données Spectral
    df_hist.to_excel(PATH_OUTPUT+"/save_clustering_hist_spectral.xlsx")
    df_hog.to_excel(PATH_OUTPUT+"/save_clustering_hog_spectral.xlsx")
    df_lbp.to_excel(PATH_OUTPUT+"/save_clustering_lbp_spectral.xlsx")
    
    df_metric.to_excel(PATH_OUTPUT+"/save_metric.xlsx")
    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")

if __name__ == "__main__":
    pipeline()