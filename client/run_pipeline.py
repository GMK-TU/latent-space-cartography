import os
import shutil
import uuid
from datasets_db import DatasetsDB
import server  # Imports your server.py file containing the corrected job functions

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_ID = "my_test_dataset"
DATASET_NAME = "Test Dataset"
DB_FILE = "datasets.db"

# Parameters for the pipeline
PARAMS_VECTORIZE = {
    'width': 64,
    'height': 64,
    'train_pct': 80,
    'latent_dims': "4, 32",
    'dataset_name': 'test_dset',
    'img_mode': 'RGB'
}

PARAMS_TRAIN = {
    'epochs': 5,
    'latent_dim': None
}

PARAMS_PCA = {} # No params needed

# ==========================================
# HELPER: MOCK ENVIRONMENT
# ==========================================
def setup_env():
    """Sets up the database and ensures raw data folder exists."""
    print(f"--- Setting up environment for {DATASET_ID} ---")
    
    # Initialize DB
    dsdb = DatasetsDB(DB_FILE)
    
    # Create Dataset Record
    dsdb.create_dataset(DATASET_ID, DATASET_NAME)
    
    # Ensure raw directory exists
    raw_dir = f"./data/{DATASET_ID}/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    # Check if user actually put images there
    valid_exts = {'.jpg', '.png', '.jpeg'}
    images = [f for f in os.listdir(raw_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not images:
        print(f"\n[!] WARNING: No images found in {raw_dir}")
        print(f"    Please manually copy some .jpg/.png files into {raw_dir} and run this script again.")
        exit(1)
        
    print(f"Found {len(images)} images in raw directory.")
    return dsdb

def run_job(job_name, job_func, params, dsdb):
    """Generic wrapper to run a job and monitor status."""
    job_id = uuid.uuid4().hex[:8]
    print(f"\n>>> STARTING JOB: {job_name} (ID: {job_id})")
    
    # Create job record
    dsdb.create_job(job_id=job_id, dataset_id=DATASET_ID, message="Starting...", stage="init")
    
    # Run the function (synchronously for this script)
    try:
        job_func(DATASET_ID, job_id, params, dsdb)
    except Exception as e:
        print(f"!!! CRASH in {job_name}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
        
    # Check result
    job = dsdb.get_job(job_id)
    if job['status'] == 'error':
        print(f"!!! JOB FAILED: {job['message']}")
        exit(1)
    else:
        print(f">>> JOB FINISHED: {job['message']}")

# ==========================================
# MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    # 1. Setup
    dsdb = setup_env()

    # 2. Step 1: Vectorize (Images -> HDF5 + Config)
    # Uses server.make_dataset_job
    run_job("Vectorization", server.make_dataset_job, PARAMS_VECTORIZE, dsdb)

    # 3. Step 2: Train VAE (HDF5 -> Model Weights)
    # Uses server.train_dataset_job
    run_job("Training", server.train_dataset_job, PARAMS_TRAIN, dsdb)

    # 4. Step 3: PCA (Latent Vectors -> 2D Projection)
    # Uses server.run_pca_job
    run_job("PCA", server.run_pca_job, PARAMS_PCA, dsdb)

    print("\n==========================================")
    print("PIPELINE COMPLETE SUCCESS")
    print("==========================================")
    print(f"Outputs located in ./data/{DATASET_ID}/")
    print(" - HDF5: ./data/{DATASET_ID}/img_vectors/")
    print(" - Models: ./data/{DATASET_ID}/models/")