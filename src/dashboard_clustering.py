import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
import argparse
import os

from utils import *
from features import *
from simclr_model import SimCLRModel
from constant import PATH_DATA, PATH_ALL_DATA, IMG_SIZE, MODELS_DIR

# --- CLI args (compatible streamlit : streamlit run dashboard_clustering.py -- --path_data output) ---
parser = argparse.ArgumentParser()
parser.add_argument('--path_data', type=str, default='output', help="Chemin vers les résultats du pipeline")
args, _ = parser.parse_known_args()

path_results = args.path_data

# ── Mapping descripteurs / modèles ────────────────────────────────
DESC_MAP = {"HISTOGRAM": "hist", "HOG": "hog", "LBP": "lbp", "SIMCLR": "simclr", "COLOR_HIST": "color"}
MODEL_MAP = {"KMeans": "kmeans", "Spectral Clustering": "spectral", "Agglomerative": "agglomerative", "DIANA": "diana"}

# ── Chargement dynamique des résultats de clustering ──────────────
@st.cache_data
def load_all_clustering_data(results_dir):
    data = {}
    for desc_name, desc_key in DESC_MAP.items():
        for model_name, model_key in MODEL_MAP.items():
            fpath = f"{results_dir}/save_clustering_{desc_key}_{model_key}.xlsx"
            if os.path.exists(fpath):
                data[(desc_name, model_name)] = pd.read_excel(fpath)
    return data

clustering_data = load_all_clustering_data(path_results)
available_descriptors = sorted(set(d for d, m in clustering_data.keys()))
available_models = sorted(set(m for d, m in clustering_data.keys()))

# ── Métriques ─────────────────────────────────────────────────────
metric_path = f"{path_results}/save_metric.xlsx"
if os.path.exists(metric_path):
    df_metric = pd.read_excel(metric_path)
    if 'Unnamed: 0' in df_metric.columns:
        df_metric.drop(columns="Unnamed: 0", inplace=True)
else:
    df_metric = pd.DataFrame()

# ── Silhouette tracking ──────────────────────────────────────────
silhouette_path = f"{path_results}/save_silhouette_tracking.xlsx"
if os.path.exists(silhouette_path):
    df_silhouette = pd.read_excel(silhouette_path)
    silhouette_loaded = True
else:
    silhouette_loaded = False

# ── Images ────────────────────────────────────────────────────────
try:
    images, labels_true, img_paths = load_images(path_data=PATH_DATA)
    images_loaded = True
except Exception:
    images, labels_true, img_paths = [], [], []
    images_loaded = False

# ── Helpers ───────────────────────────────────────────────────────
@st.cache_data
def colorize_cluster(cluster_data, selected_cluster):
    fig = px.scatter_3d(cluster_data, x='x', y='y', z='z', color='cluster')
    filtered_data = cluster_data[cluster_data['cluster'] == selected_cluster]
    fig.add_scatter3d(x=filtered_data['x'], y=filtered_data['y'], z=filtered_data['z'],
                      mode='markers', marker=dict(color='red', size=10),
                      name=f'Cluster {selected_cluster}')
    return fig

@st.cache_data
def plot_metric(df_metric):
    fig = px.bar(df_metric, x='descriptor', y='ami', color='name_model', barmode='group',
                 title='Adjusted Mutual Information (AMI) Score',
                 labels={'ami': 'AMI Score', 'descriptor': 'Descriptor type', 'name_model': 'Modèle'})
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  ONGLETS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Analyse par descripteur",
    "Analyse global",
    "Prédiction en direct",
    "Entraînement SimCLR"
])

# ── Onglet 1 : Analyse par descripteur ───────────────────────────
with tab1:
    st.write('## Résultat de Clustering des données Snacks')
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser")

    if not available_descriptors or not available_models:
        st.warning("Aucun résultat de clustering trouvé. Lancez le pipeline d'abord.")
    else:
        model = st.sidebar.selectbox('Sélectionner un modèle', available_models)
        descriptor = st.sidebar.selectbox('Sélectionner un descripteur', available_descriptors)

        key = (descriptor, model)
        if key in clustering_data:
            df = clustering_data[key]
            selected_cluster = st.sidebar.selectbox('Sélectionner un Cluster', range(20))
            cluster_indices = df[df.cluster == selected_cluster].index

            st.write(f"###  Analyse du descripteur {descriptor}")
            st.write(f"#### Modèle: {model}")
            st.write(f"#### Analyse du cluster : {selected_cluster}")
            st.write(f"####  Visualisation 3D du clustering avec descripteur {descriptor}")

            fig = colorize_cluster(df, selected_cluster)
            fig.update_traces(marker_size=3)
            st.plotly_chart(fig, use_container_width=True)

            st.write(f"#### Exemple d'images classées dans le cluster {selected_cluster} :")
            nb_images = min(10, len(cluster_indices))

            if nb_images > 0 and images_loaded:
                cols = st.columns(nb_images)
                for i in range(nb_images):
                    index = cluster_indices[i]
                    if index < len(img_paths):
                        img_bgr = cv2.imread(img_paths[index], cv2.IMREAD_COLOR)
                        if img_bgr is None:
                            continue
                        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        fig_img, ax = plt.subplots(figsize=(1, 1))
                        ax.imshow(image_rgb, interpolation='nearest')
                        ax.axis('off')
                        cols[i].pyplot(fig_img)
            elif not images_loaded:
                st.warning("Images non disponibles. Vérifiez le chemin vers les images.")
            else:
                st.write("Ce cluster est vide.")
        else:
            st.info(f"Pas de données pour la combinaison {descriptor} + {model}.")

# ── Onglet 2 : Analyse global ────────────────────────────────────
with tab2:
    st.write('## Analyse Global des descripteurs')

    if not df_metric.empty:
        plot_metric(df_metric)
        st.write('## Métriques')
        st.dataframe(df_metric, use_container_width=True)
    else:
        st.warning("Fichier de métriques non trouvé.")

    st.write('## Suivi du Silhouette Score')
    if silhouette_loaded:
        sil_models = sorted(df_silhouette['model'].unique().tolist())
        model_sil = st.selectbox('Modèle', sil_models, key="sil_model")
        df_sil_filtered = df_silhouette[df_silhouette['model'] == model_sil]
        fig_sil = px.line(
            df_sil_filtered, x='k', y='silhouette', color='descriptor',
            title=f'Silhouette Score par nombre de clusters ({model_sil})',
            labels={'k': 'Nombre de clusters', 'silhouette': 'Silhouette Score', 'descriptor': 'Descripteur'},
            markers=True
        )
        st.plotly_chart(fig_sil, use_container_width=True)
    else:
        st.warning("Données de suivi silhouette non disponibles. Relancez le pipeline.")

# ── Onglet 3 : Prédiction en direct ──────────────────────────────
with tab3:
    st.write('## Prédiction en direct')
    st.write("Uploadez une image pour prédire son cluster.")

    col1, col2 = st.columns(2)
    with col1:
        pred_model = st.selectbox('Modèle', list(MODEL_MAP.keys()), key="pred_model")
    with col2:
        pred_descriptor = st.selectbox('Descripteur', list(DESC_MAP.keys()), key="pred_desc")

    uploaded_file = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png', 'bmp'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            img_resized = cv2.resize(img_bgr, IMG_SIZE)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Image uploadée", width=250)

            # Extraction du descripteur
            if pred_descriptor == "SIMCLR":
                features = np.array(compute_simclr_descriptor_single(img_bgr, models_dir=MODELS_DIR))
            elif pred_descriptor == "HOG":
                features = np.array(compute_hog_descriptors([img_gray])[0])
            elif pred_descriptor == "HISTOGRAM":
                features = np.array(compute_gray_histograms([img_gray])[0])
            elif pred_descriptor == "LBP":
                features = np.array(compute_lbp_descriptors([img_gray])[0])
            elif pred_descriptor == "COLOR_HIST": 
                img_small = cv2.resize(img_bgr, (64, 64))
                hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
                h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
                s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
                features = np.concatenate([h_hist, s_hist])
                if features.sum() > 0: features /= features.sum()

            # Chargement des centroids
            model_key = MODEL_MAP[pred_model]
            desc_key = DESC_MAP[pred_descriptor]
            centroids_path = f"{path_results}/centroids_{model_key}_{desc_key}.npy"

            if os.path.exists(centroids_path):
                centroids = np.load(centroids_path)
                distances = np.linalg.norm(centroids - features, axis=1)
                predicted_cluster = int(np.argmin(distances))

                st.success(f"Cluster prédit : **{predicted_cluster}**")

                # Exemples d'images du cluster prédit
                data_key = (pred_descriptor, pred_model)
                if data_key in clustering_data and images_loaded:
                    df_pred = clustering_data[data_key]
                    cluster_indices_pred = df_pred[df_pred.cluster == predicted_cluster].index
                    nb = min(5, len(cluster_indices_pred))

                    if nb > 0:
                        st.write(f"#### Exemples d'images du cluster {predicted_cluster} :")
                        cols_pred = st.columns(nb)
                        for i in range(nb):
                            idx = cluster_indices_pred[i]
                            if idx < len(img_paths):
                                img_ex = cv2.imread(img_paths[idx], cv2.IMREAD_COLOR)
                                if img_ex is not None:
                                    img_ex_rgb = cv2.cvtColor(img_ex, cv2.COLOR_BGR2RGB)
                                    cols_pred[i].image(img_ex_rgb, width=100)
            else:
                st.error("Centroids non trouvés. Veuillez relancer le pipeline.")
        else:
            st.error("Impossible de lire l'image uploadée.")

# ── Onglet 4 : Entraînement SimCLR ───────────────────────────────
with tab4:
    st.write('## Entraînement du modèle SimCLR')
    st.write(
        "Le descripteur SimCLR utilise un ResNet50 fine-tuné par apprentissage "
        "contrastif. Vous pouvez entraîner ou ré-entraîner le modèle ici."
    )

    simclr_model = SimCLRModel(models_dir=MODELS_DIR)
    if simclr_model.is_trained():
        st.success("Un modèle SimCLR entraîné est disponible.")
    else:
        st.warning("Aucun modèle SimCLR trouvé. Entraînez-en un ci-dessous.")

    st.write("### Paramètres d'entraînement")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        epochs = st.slider("Epochs", 10, 500, 300, key="simclr_epochs")
    with col_b:
        batch_size = st.selectbox("Batch size", [16, 32, 64], index=1, key="simclr_batch")
    with col_c:
        lr = st.select_slider(
            "Learning rate",
            options=[1e-4, 3e-4, 5e-4, 1e-3],
            value=3e-4,
            key="simclr_lr"
        )

    if st.button("Lancer l'entraînement SimCLR"):
        all_paths = scan_all_images(PATH_ALL_DATA)
        if len(all_paths) == 0:
            st.error(f"Aucune image trouvée dans `{PATH_ALL_DATA}`. Placez les images du dataset Snacks dans ce dossier.")
        else:
            st.info(f"{len(all_paths)} images trouvées dans `{PATH_ALL_DATA}`.")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def on_epoch(epoch, total, loss):
                progress_bar.progress(epoch / total)
                status_text.text(f"Epoch {epoch}/{total} — loss: {loss:.4f}")

            new_model = SimCLRModel(models_dir=MODELS_DIR)
            new_model.train(
                all_paths,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                callback=on_epoch
            )

            progress_bar.progress(1.0)
            status_text.text("Entraînement terminé !")
            st.success(f"Modèle entraîné sur {len(all_paths)} images et sauvegardé dans `{MODELS_DIR}/`.")
            st.info("Relancez le pipeline pour mettre à jour les résultats de clustering avec le nouveau modèle.")
