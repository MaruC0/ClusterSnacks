from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import SpectralClustering as SklearnSpectralClustering
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerative
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Initialise un objet KMeans.

        Entrées:
        - n_clusters (int): Le nombre de clusters à former (par défaut 8).
        - max_iter (int): Le nombre maximum d'itérations pour l'algorithme (par défaut 300).
        - random_state (int ou None): La graine pour initialiser le générateur de nombres aléatoires (par défaut None).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def initialize_centers(self, X):
        """
        Initialise les centres de clusters avec n_clusters points choisis aléatoirement à partir des données X.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - Aucune sortie directe, mais les centres de clusters sont stockés dans self.cluster_centers_.
        """
        self.cluster_centers_ = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

    def nearest_cluster(self, X):
        """
        Calcule la distance euclidienne entre chaque point de X et les centres de clusters,
        puis retourne l'indice du cluster le plus proche pour chaque point.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - np.array: Un tableau d'indices représentant le cluster le plus proche pour chaque point.
        """
        labels = []
        for pt in X:
            distances = np.linalg.norm(self.cluster_centers_ - pt, axis=1)
            nearest = int(np.argmin(distances))
            labels.append(nearest)
        return np.array(labels)

    def fit(self, X):
        """
        Exécute l'algorithme K-means sur les données X.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - Aucune sortie directe, mais les centres de clusters sont stockés dans self.cluster_centers_.
        """
        # Génère la seed
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # initialise les centres
        self.initialize_centers(X)
        
        for i in range(self.max_iter):
            # les labels sont pour chaque point le cluster le plus proche
            self.labels_ = self.nearest_cluster(X)

            # On garde les centres précédents pour comparer
            old_centers = self.cluster_centers_.copy()
            new_centers = []
            for k in range(self.n_clusters):
                points_k = X[self.labels_ == k]
                if len(points_k) == 0:
                    # si le cluster n'a aucun point, on le rajoute comme son propre centre
                    new_centers.append(self.cluster_centers_[k])
                else:
                    # sinon le centre est la moyenne de tous ses points
                    new_centers.append(points_k.mean(axis=0))
            new_centers = np.array(new_centers)
            self.cluster_centers_ = new_centers

            # Si les clusters arrêtent de trop bouger
            if(np.allclose(old_centers, new_centers, atol=1e-10)):
                break
        

    def predict(self, X):
        """
        Prédit l'appartenance aux clusters pour les données X en utilisant les centres de clusters appris pendant l'entraînement.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - np.array: Un tableau d'indices représentant le cluster prédit pour chaque point.
        """
        return self.nearest_cluster(X)

class SpectralClustering:
    def __init__(
        self,
        n_clusters=8,
        affinity='nearest_neighbors',
        n_neighbors=15,
        assign_labels='cluster_qr',
        pca_components=64,
        random_state=42
    ):
        """
        Initialise un objet SpectralClustering.

        Entrées:
        - n_clusters (int): Le nombre de clusters à former (par défaut 8).
        - affinity (str): Type d'affinité pour la matrice de similarité (par défaut 'rbf').
        - random_state (int ou None): La graine pour initialiser le générateur de nombres aléatoires (par défaut None).
        """
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.assign_labels = assign_labels
        self.pca_components = pca_components
        self.random_state = random_state
        self.labels_ = None
        self.clusterer_ = None
        self.used_n_neighbors_ = None

    def _preprocess(self, X):
        """
        Prétraitement pour stabiliser Spectral Clustering :
        - standardisation
        - normalisation L2
        - réduction de dimension (PCA)
        """
        X = np.asarray(X, dtype=np.float32)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = normalize(X)

        if self.pca_components is not None and X.shape[1] > self.pca_components and X.shape[0] > 2:
            n_comp = min(self.pca_components, X.shape[0] - 1, X.shape[1])
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            X = pca.fit_transform(X)

        return X

    def _get_connected_n_neighbors(self, X):
        """
        Augmente progressivement k pour obtenir un graphe k-NN connecté.
        """
        n_samples = X.shape[0]
        if n_samples <= 2:
            return 1

        k = min(max(5, self.n_neighbors), n_samples - 1)
        while True:
            graph = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
            n_components = connected_components(graph, directed=False, return_labels=False)
            if n_components == 1 or k >= n_samples - 1:
                return k
            k = min(k + 5, n_samples - 1)

    def _enforce_exact_number_of_clusters(self, labels, X):
        """
        Force l'existence de n_clusters labels distincts si nécessaire,
        en réaffectant quelques points des clusters majoritaires.
        """
        labels = np.asarray(labels, dtype=int).copy()
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"Nombre d'échantillons ({X.shape[0]}) inférieur à n_clusters ({self.n_clusters})."
            )

        unique_labels = set(np.unique(labels).tolist())
        expected_labels = set(range(self.n_clusters))
        missing_labels = sorted(list(expected_labels - unique_labels))

        if not missing_labels:
            return labels

        for missing in missing_labels:
            counts = np.bincount(labels, minlength=self.n_clusters)
            donor = int(np.argmax(counts))
            donor_idx = np.where(labels == donor)[0]

            # On évite de vider un cluster existant
            if donor_idx.size <= 1:
                continue

            centroid = X[donor_idx].mean(axis=0)
            distances = np.linalg.norm(X[donor_idx] - centroid, axis=1)
            idx_to_move = donor_idx[int(np.argmax(distances))]
            labels[idx_to_move] = missing

        return labels

    def fit(self, X):
        """
        Exécute l'algorithme Spectral Clustering sur les données X.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - Aucune sortie directe, mais les labels sont stockés dans self.labels_.
        """
        X = self._preprocess(X)

        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"Nombre d'échantillons ({X.shape[0]}) inférieur à n_clusters ({self.n_clusters})."
            )

        spectral_kwargs = {
            "n_clusters": self.n_clusters,
            "affinity": self.affinity,
            "random_state": self.random_state,
            "assign_labels": self.assign_labels,
        }

        if self.affinity == 'nearest_neighbors':
            self.used_n_neighbors_ = self._get_connected_n_neighbors(X)
            spectral_kwargs["n_neighbors"] = self.used_n_neighbors_

        self.clusterer_ = SklearnSpectralClustering(**spectral_kwargs)
        labels = self.clusterer_.fit_predict(X)
        self.labels_ = self._enforce_exact_number_of_clusters(labels, X)

    def predict(self, X):
        """
        Prédit l'appartenance aux clusters pour les données X.
        Note: Spectral Clustering ne supporte pas la prédiction directe sur nouvelles données.
        Cette méthode retourne les labels du dernier fit.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - np.array: Un tableau d'indices représentant le cluster pour chaque point.
        """
        return self.labels_


class AgglomerativeClustering:
    def __init__(self, n_clusters=8, linkage='ward', random_state=None):
        """
        Initialise un objet AgglomerativeClustering (wrapper sklearn).

        Entrées:
        - n_clusters (int): Le nombre de clusters à former (par défaut 8).
        - linkage (str): Critère de liaison ('ward', 'complete', 'average', 'single').
        - random_state: Ignoré, présent pour compatibilité d'interface.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        """
        Exécute l'algorithme Agglomerative Clustering sur les données X.

        Entrée:
        - X (np.array): Les données d'entrée.
        """
        X = np.asarray(X, dtype=np.float32)
        model = SklearnAgglomerative(n_clusters=self.n_clusters, linkage=self.linkage)
        self.labels_ = model.fit_predict(X)
        self.cluster_centers_ = np.array([
            X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)
        ])

    def predict(self, X):
        """
        Retourne les labels du dernier fit.
        """
        return self.labels_

class DIANA:
    def __init__(self, n_clusters=8, random_state=42):
        """
        Initialise l'algorithme DIANA (Divisive Analysis).
        Approche Top-Down : commence avec 1 cluster et divise récursivement.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        
        # On commence avec tous les points dans le cluster 0
        current_labels = np.zeros(n_samples, dtype=int)
        n_current_clusters = 1

        while n_current_clusters < self.n_clusters:
            # 1. Trouver le cluster avec la plus grande inertie (le plus "étalé")
            max_inertia = -1
            cluster_to_split = -1
            
            for i in range(n_current_clusters):
                cluster_points = X[current_labels == i]
                if len(cluster_points) <= 1: continue
                
                centroid = cluster_points.mean(axis=0)
                inertia = np.sum((cluster_points - centroid) ** 2)
                
                if inertia > max_inertia:
                    max_inertia = inertia
                    cluster_to_split = i
            
            if cluster_to_split == -1: break # Impossible de diviser plus

            # 2. Diviser ce cluster en 2 avec un KMeans binaire
            split_data = X[current_labels == cluster_to_split]
            from sklearn.cluster import KMeans as SklearnKMeans
            km = SklearnKMeans(n_clusters=2, random_state=self.random_state, n_init=10)
            split_labels = km.fit_predict(split_data)

            # 3. Mettre à jour les labels
            # Les points avec split_labels == 1 reçoivent un nouvel ID de cluster
            new_cluster_id = n_current_clusters
            idx_in_original = np.where(current_labels == cluster_to_split)[0]
            
            for i, val in enumerate(split_labels):
                if val == 1:
                    current_labels[idx_in_original[i]] = new_cluster_id
            
            n_current_clusters += 1

        self.labels_ = current_labels
        # Calcul des centroïdes pour la prédiction
        self.cluster_centers_ = np.array([
            X[self.labels_ == i].mean(axis=0) if np.any(self.labels_ == i) else np.zeros(X.shape[1])
            for i in range(self.n_clusters)
        ])

    def predict(self, X):
        """Retourne les labels basés sur la distance aux centroïdes calculés."""
        X = np.asarray(X, dtype=np.float32)
        preds = []
        for pt in X:
            dist = np.linalg.norm(self.cluster_centers_ - pt, axis=1)
            preds.append(np.argmin(dist))
        return np.array(preds)

def show_metric(labels_true, labels_pred, descriptors,bool_return=False,name_descriptor="", name_model="kmeans",bool_show=True):
    """
    Fonction d'affichage et création des métrique pour le clustering.
    Input :
    - labels_true : étiquettes réelles des données
    - labels_pred : étiquettes prédites des données
    - descriptors : ensemble de descripteurs utilisé pour le clustering
    - bool_return : booléen indiquant si les métriques doivent être retournées ou affichées
    - name_descriptor : nom de l'ensemble de descripteurs utilisé pour le clustering
    - name_model : nom du modèle de clustering utilisé
    - bool_show : booléen indiquant si les métriques doivent être affichées ou non

    Output :
    - dictionnaire contenant les métriques d'évaluation des clusters
    """

    # Encodage des labels vrais en entiers si nécessaire
    if isinstance(labels_true[0], str):
        le = LabelEncoder()
        labels_true = le.fit_transform(labels_true)

    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    jaccard = metrics.jaccard_score(labels_true, labels_pred, average='macro')
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    silhouette = silhouette_score(descriptors, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    # Affichons les résultats
    if bool_show :
        print(f"########## Métrique descripteur : {name_descriptor}")
        print(f"Adjusted Rand Index: {ari}")
        print(f"Jaccard Index: {jaccard}")
        print(f"Homogeneity: {homogeneity}")
        print(f"Completeness: {completeness}")
        print(f"V-measure: {v_measure}")
        print(f"Silhouette Score: {silhouette}")
        print(f"Adjusted Mutual Information: {ami}")
    if bool_return:
        return {"ami":ami,
                "ari":ari, 
                "silhouette":silhouette,
                "homogeneity":homogeneity,
                "completeness":completeness,
                "v_measure":v_measure, 
                "jaccard":jaccard,
               "descriptor":name_descriptor,
               "name_model":name_model}
