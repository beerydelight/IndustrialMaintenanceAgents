import os
import json
from pathlib import Path
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file if present
# ==========================================
# 1. YOUR CONFIGURATION (The Blueprint)
# ==========================================
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")  # Ensure this is set in your environment
os.environ["KAGGLE_API_TOKEN"] = os.getenv("KAGGLE_KEY")  # Ensure this is set in your environment
kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")

KAGGLE_DATASETS = [
    {"slug": "stephanmatzka/predictive-maintenance-dataset-ai4i-2020", "csv_hint": "predictive_maintenance"},
    {"slug": "nphantawee/pump-sensor-data", "csv_hint": "pump_sensor_data"},
    {"slug": "arnabbiswas1/microsoft-azure-predictive-maintenance", "csv_hint": "telemetry"}
]

data_dir = Path(__file__).parent / "data"

# ==========================================
# 2. AUTHENTICATION SETUP (The Missing Link)
# ==========================================
def setup_kaggle_auth():
    """
    The Kaggle API strictly requires ~/.kaggle/kaggle.json. 
    This function injects your env vars into that file if it doesn't exist.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    credentials_path = kaggle_dir / "kaggle.json"

    if not credentials_path.exists():
        if not kaggle_username or not kaggle_key:
            raise EnvironmentError("KAGGLE_USERNAME and KAGGLE_KEY must be set in environment variables.")
        
        creds = {"username": kaggle_username, "key": kaggle_key}
        with open(credentials_path, 'w') as f:
            json.dump(creds, f)
        
        # Security: Kaggle API requires strict 600 permissions on Unix/Mac
        os.chmod(credentials_path, 0o600)
        print("Kaggle credentials configured.")

# ==========================================
# 3. THE DOWNLOAD LOOP (The Builder)
# ==========================================
def download_datasets():
    setup_kaggle_auth()
    
    api = KaggleApi()
    api.authenticate()
    
    data_dir.mkdir(exist_ok=True)
    
    for dataset in KAGGLE_DATASETS:
        slug = dataset["slug"]
        csv_hint = dataset["csv_hint"]
        target_csv = data_dir / f"{csv_hint}.csv"
        
        # CACHING: Don't re-download if the CSV already exists!
        if target_csv.exists():
            print(f"Cached: {csv_hint}.csv already exists. Skipping download.")
            continue
            
        print(f" Downloading {slug}...")
        try:
            # unzip=True extracts the files directly into data_dir
            api.dataset_download_files(slug, path=data_dir, unzip=True)
            print(f"Success: {slug}")
        except Exception as e:
            print(f"Failed to download {slug}: {e}")

if __name__ == "__main__":
    download_datasets()

