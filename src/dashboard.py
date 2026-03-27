"""Point d'entrée : python dashboard.py --path_data output"""
import argparse
import subprocess
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path_data", type=str, default="output")
args, extra = parser.parse_known_args()

here = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(here, "dashboard_clustering.py")

sys.exit(subprocess.call([
    sys.executable, "-m", "streamlit", "run", script,
    "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true",
    "--", "--path_data", args.path_data,
]))
