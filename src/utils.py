import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import os
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

from constant import PATH_DATA, IMG_SIZE, VALID_EXTENSIONS

def conversion_3d(X, n_components=3,perplexity=50,random_state=42, early_exaggeration=10,max_iter=3000):
    """
    Conversion des vecteurs de N dimensions vers une dimension précise (n_components) pour la visualisation
    Input : X (array-like) : données à convertir en 3D
            n_components (int) : nombre de dimensions cibles (par défaut : 3)
            perplexity (float) : valeur de perplexité pour t-SNE (par défaut : 50)
            random_state (int) : graine pour la génération de nombres aléatoires (par défaut : 42)
            early_exaggeration (float) : facteur d'exagération pour t-SNE (par défaut : 10)
            n_iter (int) : nombre d'itérations pour t-SNE (par défaut : 3000)
    Output : X_3d (array-like) : données converties en 3D
    """
    tsne = TSNE(n_components=n_components,
                random_state=random_state,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                max_iter=max_iter
               )
    X = np.array(X)
    X_3d = tsne.fit_transform(X)
    return X_3d


def create_df_to_export(data_3d, l_true_label,l_cluster):
    """
    Création d'un DataFrame pour stocker les données et les labels
    Input : data_3d (array-like) : données converties en 3D
            l_true_label (list) : liste des labels vrais
            l_cluster (list) : liste des labels de cluster
            l_path_img (list) : liste des chemins des images
    Output : df (DataFrame) : DataFrame contenant les données et les labels
    """
    df = pd.DataFrame(data_3d, columns=['x', 'y', 'z'])
    df['label'] = l_true_label
    df['cluster'] = l_cluster
    
    return df

def load_images(path_data=PATH_DATA):
    """
    Charge toutes les images depuis le répertoire de données.

    Les images sont organisées dans des sous-dossiers dont les noms
    représentent les vraies classes (étiquettes). Cette fonction
    parcourt tous les sous-dossiers, lit chaque image avec OpenCV,
    la redimensionne et la convertit en niveaux de gris.

    Paramètres:
        path_data (str): Chemin vers le dossier racine contenant les sous-dossiers de classes.

    Retourne:
        images (list): Liste de tableaux numpy (images en niveaux de gris, 128x128).
        labels (list): Liste des vraies étiquettes (noms des sous-dossiers).
        img_paths (list): Liste des chemins vers chaque image (pour le dashboard).
    """
    images = []
    labels = []
    img_paths = []

    # Résolution du chemin absolu pour éviter toute ambiguïté
    abs_path_data = os.path.abspath(path_data)

    # Vérification que le répertoire de données existe
    if not os.path.exists(abs_path_data):
        raise FileNotFoundError(
            f"Le répertoire de données est introuvable : '{abs_path_data}'\n"
            f"Vérifiez que vous lancez les scripts depuis le dossier 'images/' "
            f"et que la structure 'data/test/' existe bien."
        )

    # Récupération et tri des sous-dossiers (chaque sous-dossier = une classe)
    # On filtre les dossiers cachés comme MACOSX
    class_folders = sorted([
        d for d in os.listdir(abs_path_data)
        if os.path.isdir(os.path.join(abs_path_data, d))
        and not d.startswith('.')
        and not d.startswith('__')
    ])
    if not class_folders:
        raise ValueError(f"Aucun sous-dossier de classe trouvé dans : '{abs_path_data}'")

    print(f"[INFO] {len(class_folders)} classes trouvées dans '{abs_path_data}'.")
    print(f"[INFO] Classes : {class_folders}")

    # Parcours de chaque sous-dossier de classe
    for class_name in class_folders:
        class_folder_path = os.path.join(abs_path_data, class_name)

        # Récupération de tous les fichiers images dans le sous-dossier
        # On ignore les fichiers cachés (commençant par '.')
        image_files = sorted([
            f for f in os.listdir(class_folder_path)
            if f.lower().endswith(VALID_EXTENSIONS)
            and not f.startswith('.')
            and not f.startswith('__')
        ])

        if not image_files:
            print(f"[AVERTISSEMENT] Aucune image trouvée dans la classe : '{class_name}'")
            continue
        # Lecture et prétraitement de chaque image
        for img_file in image_files:
            img_path = os.path.join(class_folder_path, img_file)

            # Lecture de l'image en couleur avec OpenCV
            img_bgr = cv2.imread(img_path)

            # Ignorer les fichiers illisibles
            if img_bgr is None:
                print(f"[AVERTISSEMENT] Impossible de lire : '{img_path}'")
                continue

            # Redimensionnement de l'image à la taille cible (128x128)
            img_resized = cv2.resize(img_bgr, IMG_SIZE)

            # Conversion en niveaux de gris (SIFT fonctionne sur images en gris)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            # Ajout de l'image et de ses métadonnées aux listes
            images.append(img_gray)
            labels.append(class_name)
            img_paths.append(img_path)  # Chemin absolu pour le dashboard

    print(f"[INFO] {len(images)} images chargées avec succès.")
    return images, labels, img_paths