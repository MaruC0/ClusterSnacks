FROM python:3.10-slim

WORKDIR /app/src

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requierements.txt .

RUN pip install --no-cache-dir -r requierements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard_clustering.py", "--server.port=8501", "--server.address=0.0.0.0", "--", "--path_data", "src/output"]