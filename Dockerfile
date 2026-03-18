FROM python:3.10

WORKDIR /app

# Copier les requirements et installer les dépendances
COPY requierements.txt .
RUN pip install --no-cache-dir -r requierements.txt

# Copier le code source
COPY src/ /app/src/
COPY src/output/ /app/src/output/

# Exposer le port Streamlit
EXPOSE 8501

# Configuration Streamlit
RUN mkdir -p ~/.streamlit && \
    echo "[server]\nheadless = true\nport = 8501\nenableXsrfProtection = false\n" > ~/.streamlit/config.toml

# Lancer le dashboard
CMD ["streamlit", "run", "src/dashboard_clustering.py"]