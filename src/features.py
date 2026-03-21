import os
import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog, local_binary_pattern
from skimage import transform
from sklearn.preprocessing import normalize
import itertools

def compute_gray_histograms(images, n_bins=64):
    """
    Calcule les histogrammes de niveau de gris pour les images MNIST.
    Input : images (list) : liste des images en niveaux de gris
    Output : descriptors (list) : liste des descripteurs d'histogrammes de niveau de gris
    """
    descriptors = []
    for image in images:
        image_uint8 = image.astype(np.uint8)
        hist = cv2.calcHist(images=[image_uint8], channels=[0], mask=None, histSize=[n_bins], ranges=[0, 256])
        hist = hist.flatten().astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s  # normalisation L1 pour stabiliser les distances
        descriptors.append(hist)
    return descriptors

def compute_hog_descriptors(images):
    """
    Calcule les descripteurs HOG pour les images en niveaux de gris.
    Input : images (array) : tableau numpy des images
    Output : descriptors (list) : liste des descripteurs HOG
    """
    descriptors = []
    def calc(img) :
        img = img.astype(np.float32) / 255.0
        return hog(
            img,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
    descriptors = Parallel(n_jobs=-1)(
        delayed(calc)(img) for img in images
    )
    return descriptors


def compute_lbp_descriptors(
    images,
    scales=((8, 1), (16, 2), (24, 3)),
    spatial_levels=(1, 2),
    method='uniform'
):
    """
    Calcule des descripteurs LBP multi-échelle + spatial pyramid.

    Input :
    - images (array-like) : liste/tableau des images grayscale
    - scales (tuple) : tuple de couples (P, R) pour LBP multi-échelle
      ex: ((8,1), (16,2), (24,3))
    - spatial_levels (tuple) : niveaux de grille spatiale (1 => 1x1, 2 => 2x2, ...)
    - method (str) : méthode LBP (par défaut 'uniform')

    Output :
    - descriptors (list) : descripteurs concaténés et normalisés
    """
    descriptors = []

    for image in images:
        img = image.astype(np.uint8)
        h, w = img.shape[:2]
        feature_vector = []

        for n_points, radius in scales:
            n_bins = n_points + 2 if method == 'uniform' else int(2 ** n_points)
            lbp = local_binary_pattern(img, P=n_points, R=radius, method=method)

            for level in spatial_levels:
                cell_h = h // level
                cell_w = w // level

                for i in range(level):
                    for j in range(level):
                        y0 = i * cell_h
                        x0 = j * cell_w
                        y1 = h if i == level - 1 else (i + 1) * cell_h
                        x1 = w if j == level - 1 else (j + 1) * cell_w

                        lbp_cell = lbp[y0:y1, x0:x1]
                        hist, _ = np.histogram(
                            lbp_cell.ravel(),
                            bins=np.arange(0, n_bins + 1),
                            range=(0, n_bins)
                        )
                        hist = hist.astype(np.float32)
                        hist /= (hist.sum() + 1e-12)
                        feature_vector.append(hist)

        feature_vector = np.concatenate(feature_vector).astype(np.float32)
        feature_vector /= (np.linalg.norm(feature_vector) + 1e-12)
        descriptors.append(feature_vector)

    return descriptors


def compute_color_histograms(img_paths, h_bins=18, s_bins=8, size=64):
    """
    Calcule les histogrammes de couleur HSV (Hue + Saturation) à partir des images couleur.
    Le Hue capture la teinte pure (indépendant de la luminosité),
    la Saturation sépare les fonds gris des aliments colorés.

    Input :
    - img_paths (list) : liste des chemins vers les images
    - h_bins (int) : nombre de bins pour la teinte (par défaut 18, couvrant 360° par pas de 20°)
    - s_bins (int) : nombre de bins pour la saturation (par défaut 8)

    Output :
    - descriptors (list) : liste des descripteurs HSV normalisés
    """
    descriptors = []
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            descriptors.append(np.zeros(h_bins + s_bins, dtype=np.float32))
            continue
        img = cv2.resize(img, (size, size))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256]).flatten()
        hist = np.concatenate([h_hist, s_hist])
        s = hist.sum()
        if s > 0:
            hist /= s
        descriptors.append(hist.astype(np.float32))
    return descriptors


def compute_simclr_descriptors(img_paths, models_dir="trained-models",
                                simclr_weight=0.7, color_weight=0.3):
    """
    Calcule les descripteurs SimCLR (ResNet50 fine-tuné) combinés avec
    les histogrammes de couleur HSV.

    Le SimCLR capture forme et texture, les histogrammes HSV capturent
    la distribution de couleur dominante — la combinaison permet de
    distinguer des aliments de forme similaire mais de couleur différente.

    Input :
    - img_paths (list) : liste des chemins vers les images
    - models_dir (str) : chemin vers le dossier des modèles entraînés
    - simclr_weight (float) : poids des features SimCLR (par défaut 0.7)
    - color_weight (float) : poids des features couleur (par défaut 0.3)

    Output :
    - descriptors (list) : liste des descripteurs combinés
    """
    from simclr_model import SimCLRModel

    model = SimCLRModel(models_dir=models_dir)
    simclr_feats = model.extract_features(img_paths)
    color_feats = np.array(compute_color_histograms(img_paths))

    simclr_normed = normalize(simclr_feats) * simclr_weight
    color_normed = normalize(color_feats) * color_weight

    combined = np.concatenate([simclr_normed, color_normed], axis=1)
    return [combined[i] for i in range(combined.shape[0])]


def compute_simclr_descriptor_single(img_bgr, models_dir="trained-models",
                                      simclr_weight=0.7, color_weight=0.3):
    """
    Calcule le descripteur SimCLR + couleur pour une seule image BGR.
    Utilisé pour la prédiction en direct dans le dashboard.

    Input :
    - img_bgr (np.array) : image BGR (numpy array)
    - models_dir (str) : chemin vers le dossier des modèles entraînés

    Output :
    - descriptor (np.array) : vecteur descripteur combiné
    """
    from simclr_model import SimCLRModel

    model = SimCLRModel(models_dir=models_dir)
    simclr_feat = model.extract_features_from_array(img_bgr)

    img_small = cv2.resize(img_bgr, (64, 64))
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    color_hist = np.concatenate([h_hist, s_hist]).astype(np.float32)
    s = color_hist.sum()
    if s > 0:
        color_hist /= s

    simclr_normed = normalize(simclr_feat.reshape(1, -1))[0] * simclr_weight
    color_normed = normalize(color_hist.reshape(1, -1))[0] * color_weight
    return np.concatenate([simclr_normed, color_normed])


