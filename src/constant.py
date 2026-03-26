import os

PATH_OUTPUT = "output"
MODEL_CLUSTERING = "spectral"
MODELS_DIR = "trained-models"
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
IMG_SIZE = (128, 128)

# Chemins pour Docker (qui tourne depuis la racine /app)
PATH_DATA = "data/test"
PATH_ALL_DATA = "data/tous"

if not os.path.exists(PATH_DATA) and os.path.exists("../data/test"):
    PATH_DATA = "../data/test"
    PATH_ALL_DATA = "../data/tous"