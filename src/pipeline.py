from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import argparse
import pandas as pd
import numpy as np

from features import *
from clustering import *
from utils import *
from simclr_model import SimCLRModel
from constant import PATH_OUTPUT, PATH_DATA, MODELS_DIR, PATH_ALL_DATA

# Mapping descripteur → clé fichier
DESC_KEY_MAP = {"HISTOGRAM": "hist", "HOG": "hog", "LBP": "lbp", "SIMCLR": "simclr", "COLOR_HIST": "color"}

AGGLOMERATIVE_CONFIG = {"linkage": "ward"}
# Config SpectralClustering par descripteur
SPECTRAL_CONFIGS = {
    "HISTOGRAM": {"affinity": "nearest_neighbors", "n_neighbors": 40, "gamma": 0.3},
    "HOG":       {"affinity": "nearest_neighbors", "n_neighbors": 30, "gamma": 10.0},
    "LBP":       {"affinity": "nearest_neighbors", "n_neighbors": 30, "gamma": 3.0},
    "SIMCLR":    {"affinity": "nearest_neighbors", "n_neighbors": 25, "gamma": 1.0},
    "COLOR_HIST": {"affinity": "nearest_neighbors", "n_neighbors": 20, "gamma": 0.03},
}
GAMMA_GRID_LOG = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
TEST_RBF_GAMMA_GRID = False
N_NEIGHBORS_GRID = [5, 10, 15, 20, 25, 30, 40]
TEST_N_NEIGHBORS_GRID = False

def _create_models(n_clusters, desc_name):
    """Crée les 4 modèles de clustering pour un descripteur donné."""
    sc_cfg = SPECTRAL_CONFIGS[desc_name]
    spectral_kwargs = {
        "n_clusters": n_clusters,
        "affinity": sc_cfg.get("affinity", "nearest_neighbors"),
        "assign_labels": 'cluster_qr',
        "pca_components": 64,
        "random_state": 42,
    }

    if spectral_kwargs["affinity"] == "nearest_neighbors":
        spectral_kwargs["n_neighbors"] = sc_cfg["n_neighbors"]
    elif spectral_kwargs["affinity"] == "rbf":
        spectral_kwargs["gamma"] = sc_cfg["gamma"]

    return {
        ##"kmeans": KMeans(n_clusters=n_clusters, random_state=42),
        "spectral": SpectralClustering(**spectral_kwargs)
        ##"agglomerative": AgglomerativeClustering(n_clusters=n_clusters, 
            ##linkage=AGGLOMERATIVE_CONFIG["linkage"]),
        ##"diana": DIANA(n_clusters=n_clusters, random_state=42),
    }


def pipeline(path_data=PATH_DATA, path_output=PATH_OUTPUT):
    print("##### Chargement des données ######")
    images, labels_true, img_paths = load_images(path_data=path_data)

    # ── Extraction de Features ────────────────────────────────────
    print("\n\n ##### Extraction de Features ######")

    print("- calcul features HOG...")
    descriptors_hog = compute_hog_descriptors(images)
    print("- calcul features Histogram...")
    descriptors_hist = compute_gray_histograms(images)
    print("- calcul features LBP...")
    descriptors_lbp = compute_lbp_descriptors(images)
    print("- calcul features Color HSV...")
    descriptors_color = compute_color_histograms(img_paths)

    print("- calcul features SimCLR...")
    simclr = SimCLRModel(models_dir=MODELS_DIR)
    if not simclr.is_trained():
        print("[SimCLR] Aucun modèle trouvé — entraînement automatique...")
        all_paths = scan_all_images(PATH_ALL_DATA)
        if len(all_paths) > 0:
            simclr.train(all_paths, epochs=100, batch_size=32)
        else:
            print(f"[WARN] Aucune image trouvée dans {PATH_ALL_DATA} pour l'entraînement SimCLR")
    descriptors_simclr = compute_simclr_descriptors(img_paths, models_dir=MODELS_DIR)

    descriptors = {
        "HISTOGRAM": descriptors_hist,
        "HOG":       descriptors_hog,
        "LBP":       descriptors_lbp,
        "SIMCLR":    descriptors_simclr,
        "COLOR_HIST": descriptors_color,
    }

    # ── Clustering ────────────────────────────────────────────────
    print("\n\n ##### Clustering ######")
    number_cluster = 20
    all_metrics = []
    cluster_results = {}

    for desc_name, desc_data in descriptors.items():
        data = np.array(desc_data)
        desc_key = DESC_KEY_MAP[desc_name]
        sc_cfg = SPECTRAL_CONFIGS[desc_name]

        if TEST_RBF_GAMMA_GRID:
            print(f"- spectral + {desc_name} [RBF gamma grid] ...")

            best_silhouette = -np.inf
            best_model = None
            best_labels = None
            best_gamma = None

            for gamma in GAMMA_GRID_LOG:
                model = SpectralClustering(
                    n_clusters=number_cluster,
                    affinity='rbf',
                    gamma=gamma,
                    assign_labels='cluster_qr',
                    pca_components=64,
                    random_state=42
                )
                model.fit(data)

                unique, counts = np.unique(model.labels_, return_counts=True)
                print(f"  gamma={gamma}: {len(unique)} clusters, distribution={dict(zip(unique.tolist(), counts.tolist()))}")

                metric = show_metric(
                    labels_true, model.labels_, desc_data,
                    bool_show=True, name_descriptor=desc_name,
                    name_model=f"spectral_rbf_gamma_{gamma}", bool_return=True
                )
                metric["gamma"] = gamma
                all_metrics.append(metric)

                if metric["silhouette"] > best_silhouette:
                    best_silhouette = metric["silhouette"]
                    best_model = model
                    best_labels = model.labels_
                    best_gamma = gamma

            print(f"  -> meilleur gamma: {best_gamma} (silhouette={best_silhouette:.4f})")
            cluster_results[(desc_key, "spectral")] = {
                "model": best_model,
                "data": data,
                "labels": best_labels,
            }
            continue

        if TEST_N_NEIGHBORS_GRID:
            print(f"- spectral + {desc_name} [nearest_neighbors grid] ...")

            best_silhouette = -np.inf
            best_model = None
            best_labels = None
            best_n_neighbors = None

            for n_neighbors in N_NEIGHBORS_GRID:
                model = SpectralClustering(
                    n_clusters=number_cluster,
                    affinity='nearest_neighbors',
                    n_neighbors=n_neighbors,
                    assign_labels='cluster_qr',
                    pca_components=64,
                    random_state=42
                )
                model.fit(data)

                unique, counts = np.unique(model.labels_, return_counts=True)
                print(f"  n_neighbors={n_neighbors}: {len(unique)} clusters, distribution={dict(zip(unique.tolist(), counts.tolist()))}")

                metric = show_metric(
                    labels_true, model.labels_, desc_data,
                    bool_show=True, name_descriptor=desc_name,
                    name_model=f"spectral_knn_neighbors_{n_neighbors}", bool_return=True
                )
                metric["n_neighbors"] = n_neighbors
                all_metrics.append(metric)

                if metric["silhouette"] > best_silhouette:
                    best_silhouette = metric["silhouette"]
                    best_model = model
                    best_labels = model.labels_
                    best_n_neighbors = n_neighbors

            print(f"  -> meilleur n_neighbors: {best_n_neighbors} (silhouette={best_silhouette:.4f})")
            cluster_results[(desc_key, "spectral")] = {
                "model": best_model,
                "data": data,
                "labels": best_labels,
            }
            continue

        models = _create_models(number_cluster, desc_name)

        for model_name, model in models.items():
            print(f"- {model_name} + {desc_name} ...")
            model.fit(data)

            unique, counts = np.unique(model.labels_, return_counts=True)
            print(f"  {len(unique)} clusters, distribution={dict(zip(unique.tolist(), counts.tolist()))}")

            metric = show_metric(
                labels_true, model.labels_, desc_data,
                bool_show=True, name_descriptor=desc_name,
                name_model=model_name, bool_return=True
            )
            all_metrics.append(metric)
            cluster_results[(desc_key, model_name)] = {
                "model": model,
                "data": data,
                "labels": model.labels_,
            }

    '''
    # ── Silhouette Score Tracking ─────────────────────────────────
    print("\n\n##### Silhouette Score Tracking ######")
    k_values = [5, 10, 15, 20, 25]
    silhouette_records = []

    for k in k_values:
        print(f"--- k = {k} ---")
        for desc_name, desc_data in descriptors.items():
            data = np.array(desc_data)

            km = KMeans(n_clusters=k, random_state=42)
            km.fit(data)
            sil_km = silhouette_score(data, km.labels_)

            sc_cfg = SPECTRAL_CONFIGS[desc_name]
            spectral_kwargs = {
                "n_clusters": k,
                "affinity": sc_cfg.get("affinity", "nearest_neighbors"),
                "assign_labels": 'cluster_qr',
                "pca_components": 64,
                "random_state": 42,
            }
            if spectral_kwargs["affinity"] == "nearest_neighbors":
                spectral_kwargs["n_neighbors"] = sc_cfg["n_neighbors"]
            elif spectral_kwargs["affinity"] == "rbf":
                spectral_kwargs["gamma"] = sc_cfg.get("gamma", 3.0)

            sc = SpectralClustering(**spectral_kwargs)
            sc.fit(data)
            sil_sc = silhouette_score(data, sc.labels_)

            ac = AgglomerativeClustering(n_clusters=k)
            ac.fit(data)
            sil_ac = silhouette_score(data, ac.labels_)
            
            diana_model = DIANA(n_clusters=k, random_state=42)
            diana_model.fit(data)
            sil_diana = silhouette_score(data, diana_model.labels_)

            silhouette_records.extend([
                {"k": k, "descriptor": desc_name, "model": "kmeans",        "silhouette": sil_km},
                {"k": k, "descriptor": desc_name, "model": "spectral",      "silhouette": sil_sc},
                {"k": k, "descriptor": desc_name, "model": "agglomerative", "silhouette": sil_ac},
                {"k": k, "descriptor": desc_name, "model": "diana",         "silhouette": sil_diana},
            ])
            print(f"  {desc_name}: km={sil_km:.4f}  sp={sil_sc:.4f}  ag={sil_ac:.4f}  di={sil_diana:.4f}")

    df_silhouette = pd.DataFrame(silhouette_records)
    '''

    # ── Export des données ────────────────────────────────────────
    print("\n\n- export des données vers le dashboard")
    df_metric = pd.DataFrame(all_metrics)

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    scaler = StandardScaler()
    for desc_name, desc_data in descriptors.items():
        desc_key = DESC_KEY_MAP[desc_name]
        data = np.array(desc_data)
        data_norm = scaler.fit_transform(data)
        x_3d = conversion_3d(data_norm)

        ##for model_name in ["kmeans", "spectral", "agglomerative", "diana"]:
        for model_name in ["spectral"]:
            result = cluster_results[(desc_key, model_name)]
            df_export = create_df_to_export(x_3d, labels_true, result["labels"])
            df_export.to_excel(f"{path_output}/save_clustering_{desc_key}_{model_name}.xlsx")

            model_obj = result["model"]
            if hasattr(model_obj, 'cluster_centers_') and model_obj.cluster_centers_ is not None:
                np.save(f"{path_output}/centroids_{model_name}_{desc_key}.npy", model_obj.cluster_centers_)
            else:
                centroids = np.array([data[result["labels"] == i].mean(axis=0)
                                      for i in range(number_cluster)])
                np.save(f"{path_output}/centroids_{model_name}_{desc_key}.npy", centroids)

    df_metric.to_excel(f"{path_output}/save_metric.xlsx")
    #df_silhouette.to_excel(f"{path_output}/save_silhouette_tracking.xlsx", index=False)

    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de clustering d'images")
    parser.add_argument('--path_data', type=str, default=PATH_DATA, help="Chemin vers les données images")
    parser.add_argument('--path_output', type=str, default=PATH_OUTPUT, help="Chemin vers le dossier de sortie")
    args = parser.parse_args()
    pipeline(path_data=args.path_data, path_output=args.path_output)
