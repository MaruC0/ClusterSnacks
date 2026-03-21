FROM python:3.10

WORKDIR /app

COPY requierements.txt .
RUN pip install --no-cache-dir -r requierements.txt

COPY src/ ./src/

WORKDIR /app/src

EXPOSE 8501

RUN mkdir -p ~/.streamlit && \
    echo "[server]\nheadless = true\nport = 8501\nenableXsrfProtection = false\n" > ~/.streamlit/config.toml

CMD ["streamlit", "run", "dashboard_clustering.py"]
