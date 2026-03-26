# Projet de clustering des images Snacks pour ET4 info Polytech Paris Saclay

### step 1 : téléchargement des données et installation des packages
    - a. installer les requierements : "pip install -r requierements.txt"
    - b. télécharger les données des images des snacks : https://huggingface.co/datasets/Matthijs/snacks/tree/main
    => télécharger le zip images, dézipper, récupérer le dossier data dans le dossier images et placer le dossier data dans le projet dans le dossier src.

### step 2 : configuration du chemin vers les données
    - a. dans le dossier src/constant.py, modifier les variables PATH_DATA et PATH_ALL_DATA si nécessaire.

### step 3 : run de la pipeline clustering
    - a. aller dans le dossier src
    - b. exécutez la commande : "python pipeline.py --path_data images/data/test --path_output output"
    - Descripteurs : HOG, Histogram, LBP, SimCLR (entraîné automatiquement si absent)
    - Modèles : KMeans, Spectral Clustering, Agglomerative Clustering

### step 4 : lancement du dashboard
    - a. aller dans le dossier src
    - b. exécutez la commande : "streamlit run dashboard_clustering.py -- --path_data output"
    - Onglets : Analyse par descripteur, Analyse global, Prédiction en direct, Entraînement SimCLR

### step 5 : Docker
    - a. installer docker
    - b. pour build une image docker : docker build -t mon-app-python .
    - c. pour run l'image : docker run -d -p 8501:8501 mon-app-python