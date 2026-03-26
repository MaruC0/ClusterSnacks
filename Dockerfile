# Utiliser une image Python officielle et légère
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires pour OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances
COPY requierements.txt .

# Installer les bibliothèques Python
RUN pip install --no-cache-dir -r requierements.txt

# Copier tout le reste du code source
COPY . .

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Commande pour lancer le dashboard
# On ajoute "--", "--path_data", "src/output" pour lui indiquer le bon dossier
CMD ["streamlit", "run", "src/dashboard_clustering.py", "--server.port=8501", "--server.address=0.0.0.0", "--", "--path_data", "src/output"]