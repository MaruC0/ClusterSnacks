import os

PATH_OUTPUT = "output"
MODEL_CLUSTERING = "spectral"
MODELS_DIR = "trained-models"
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
IMG_SIZE = (128, 128)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATH_DATA = os.path.join(_ROOT, "data", "test")
PATH_ALL_DATA = os.path.join(_ROOT, "data", "tous")
