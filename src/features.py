import os
import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog, local_binary_pattern
from skimage import transform
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
    
