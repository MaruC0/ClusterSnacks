import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import cv2

from utils import *
from constant import PATH_DATA

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
    st.plotly_chart(fig, use_container_width = True)

        
# Chargement des données du clustering
df_hist = pd.read_excel("output/save_clustering_hist_spectral.xlsx")
df_hog = pd.read_excel("output/save_clustering_hog_spectral.xlsx")
df_lbp = pd.read_excel("output/save_clustering_lbp_spectral.xlsx")

df_hist_kmeans = pd.read_excel("output/save_clustering_hist_kmeans.xlsx")
df_hog_kmeans = pd.read_excel("output/save_clustering_hog_kmeans.xlsx")
df_lbp_kmeans = pd.read_excel("output/save_clustering_lbp_kmeans.xlsx")

df_metric = pd.read_excel("output/save_metric.xlsx")

if 'Unnamed: 0' in df_metric.columns:
    df_metric.drop(columns="Unnamed: 0", inplace=True)

images, labels_true, img_paths = load_images(path_data=PATH_DATA)

# Création de deux onglets
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global" ])

# Onglet numéro 1
with tab1:

    st.write('## Résultat de Clustering des données DIGITS')
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser" )
    
    # Sélection du modèle
    model = st.sidebar.selectbox('Sélectionner un modèle', ["Spectral Clustering", "KMeans"])
    
    # Sélection des descripteurs
    descriptor =  st.sidebar.selectbox('Sélectionner un descripteur', ["HISTOGRAM", "HOG", "LBP"])
    
    if descriptor=="HISTOGRAM":
        df = df_hist if model == "Spectral Clustering" else df_hist_kmeans
    elif descriptor=="HOG":
        df = df_hog if model == "Spectral Clustering" else df_hog_kmeans
    elif descriptor=="LBP":
        df = df_lbp if model == "Spectral Clustering" else df_lbp_kmeans
    # Ajouter un sélecteur pour les clusters
    selected_cluster =  st.sidebar.selectbox('Sélectionner un Cluster', range(20))
    # Filtrer les données en fonction du cluster sélectionné
    cluster_indices = df[df.cluster==selected_cluster].index    
    st.write(f"###  Analyse du descripteur {descriptor}" )
    st.write(f"#### Modèle: {model}")
    st.write(f"#### Analyse du cluster : {selected_cluster}")
    st.write(f"####  Visualisation 3D du clustering avec descripteur {descriptor}" )
    # Sélection du cluster choisi
    filtered_data = df[df['cluster'] == selected_cluster]
    # Création d'un graph 3D des clusters
    fig = colorize_cluster(df, selected_cluster)
    fig.update_traces(marker_size=3)
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"#### Exemple d'images classées dans le cluster {selected_cluster} :")
    nb_images = min(10, len(cluster_indices))

    if nb_images > 0:
        cols = st.columns(nb_images)
        for i in range (nb_images):
            index = cluster_indices[i]
            img_bgr = cv2.imread(img_paths[index], cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            fig_img, ax = plt.subplots(figsize=(1,1))
            ax.imshow(image_rgb, interpolation='nearest')
            ax.axis('off')
            cols[i].pyplot(fig_img)
    else:
        st.write("Ce cluster est vide.")

# Onglet numéro 2
with tab2:
    st.write('## Analyse Global des descripteurs' )
    # Complèter la fonction plot_metric() pour afficher les histogrammes du score AMI
    plot_metric(df_metric)
    st.write('## Métriques ')
    # affichage d'un tableau
    st.dataframe(df_metric, use_container_width=True)
