from pathlib import Path

# Project root (this file lives in the project root)
PROJECT_ROOT = Path(__file__).resolve().parent

# DATA_DIR used by notebooks/scripts — points to project root and ends with a slash
DATA_DIR = str(PROJECT_ROOT) + '/'
