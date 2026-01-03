#!/usr/bin/env python

'''
Start training a VAE model.
The model architecture is defined in model.py.
'''

from __future__ import print_function
import h5py
import numpy as np
import os
from PIL import Image

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, Callback, ReduceLROnPlateau
from keras import backend as K
import model  # Assumes model.py is in the same folder

class DbProgressCallback(Callback):
    """Updates the database with training progress per epoch."""
    def __init__(self, dsdb, job_id, total_epochs, start_progress, end_progress):
        super().__init__()
        self.dsdb = dsdb
        self.job_id = job_id
        self.total_epochs = total_epochs
        # Map epoch 0-N to progress range (e.g., 20% to 80%)
        self.p_start = start_progress
        self.p_end = end_progress

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Calculate percentage within the allocated range
        fraction = (epoch + 1) / self.total_epochs
        current_prog = int(self.p_start + fraction * (self.p_end - self.p_start))
        
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        msg = f"Epoch {epoch+1}/{self.total_epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}"
        
        self.dsdb.update_job(self.job_id, progress=current_prog, message=msg)

class Visualizer(Callback):
    def __init__(self, x_test, encoder, generator, out_dir, img_dims, img_mode):
        super().__init__()
        self.x_test = x_test
        self.encoder = encoder
        self.generator = generator
        self.out_dir = out_dir
        # Config provides (rows, cols, chns), but we might need to handle shaping for PIL
        self.img_rows, self.img_cols, self.img_chns = img_dims
        self.img_mode = img_mode

    def on_train_end(self, logs={}):
        self._save_images('end')

    def _save_images(self, suffix):
        # Select small batch for visualization
        n = min(25, len(self.x_test))
        sample = self.x_test[:n]
        
        encoded = self.encoder.predict(sample)
        decoded = self.generator.predict(encoded)
        
        # Create a grid (5x5)
        m = 5
        grid_h, grid_w = self.img_rows * m, self.img_cols * m
        original = np.zeros((grid_h, grid_w, self.img_chns), dtype='uint8')
        reconstructed = np.zeros((grid_h, grid_w, self.img_chns), dtype='uint8')
        
        def to_img(arr):
            # Reshape and denormalize
            arr = arr.reshape(self.img_rows, self.img_cols, self.img_chns)
            return (arr * 255).astype('uint8')

        for i in range(m):
            for j in range(m):
                idx = i * m + j
                if idx >= n: break
                
                orig_tile = to_img(sample[idx])
                recon_tile = to_img(decoded[idx])
                
                y0, y1 = i * self.img_rows, (i+1) * self.img_rows
                x0, x1 = j * self.img_cols, (j+1) * self.img_cols
                
                original[y0:y1, x0:x1] = orig_tile
                reconstructed[y0:y1, x0:x1] = recon_tile

        # Handle grayscale (squeeze channel) for PIL saving
        if self.img_chns == 1:
            original = original.squeeze()
            reconstructed = reconstructed.squeeze()
            
        Image.fromarray(original, mode=self.img_mode).save(f"{self.out_dir}/original.png")
        Image.fromarray(reconstructed, mode=self.img_mode).save(f"{self.out_dir}/reconstructed_{suffix}.png")

def load_data(h5_path, key, train_split):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found at {h5_path}")
        
    with h5py.File(h5_path, 'r') as f:
        if key not in f:
            raise ValueError(f"Dataset key '{key}' not found in HDF5 file.")
        dset = f[key]
        # We load everything to create the full split logic
        x_all = dset[:]
        
    return x_all

def train_vae(dataset_id, job_id, config, latent_dim, epochs, dsdb):
    """
    Main training function to be called by the job worker.
    """
    # 1. Setup Paths
    root = f'./data/{dataset_id}'
    vectors_dir = os.path.join(root, 'img_vectors')
    h5_path = os.path.join(vectors_dir, config.fn_raw)
    
    # Output directory for this specific latent dimension
    out_dir = os.path.join(root, 'models', str(latent_dim))
    os.makedirs(out_dir, exist_ok=True)
    
    mpath = os.path.join(out_dir, 'model.json')
    wpath = os.path.join(out_dir, 'weights.h5')
    logpath = os.path.join(out_dir, 'training_log.csv')
    embeddings_path = os.path.join(out_dir, 'latent_vectors.h5')

    # 2. Load Data
    try:
        x_all = load_data(h5_path, config.key_raw, config.train_split)
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {e}")

    # Split Data based on config
    train_split = config.train_split
    x_train = x_all[:train_split]
    x_test = x_all[train_split:]

    # Normalize (0-255 -> 0.0-1.0)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_all = x_all.astype('float32') / 255.
    
    # 3. Initialize Model
    K.clear_session()
    
    # IMPORTANT: model.py expects (Channels, Cols, Rows)
    # The config usually gives us row/col/chns. 
    # We pass it in the specific order the Vae class expects.
    model_dims = (config.img_chns, config.img_cols, config.img_rows)
    
    # For visualization, we need (Rows, Cols, Chns)
    vis_dims = (config.img_rows, config.img_cols, config.img_chns)

    vae_model = model.Vae(latent_dim=latent_dim, img_dim=model_dims)
    vae, encoder, generator = vae_model.init_model(mpath, wpath)

    # 4. Callbacks
    cp = ModelCheckpoint(wpath, save_best_only=True, save_weights_only=True)
    stop = EarlyStopping(monitor='val_loss', patience=15, verbose=0)
    csv_logger = CSVLogger(logpath, append=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    
    # Visualizer
    vis = Visualizer(x_test, encoder, generator, out_dir, vis_dims, config.img_mode)
    
    # DB Updater (Runs from 10% to 90% of job progress)
    db_cb = DbProgressCallback(dsdb, job_id, epochs, start_progress=10, end_progress=90)

    # 5. Fit
    print(f"Starting training for dim={latent_dim}...")
    vae.fit(
        x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=100, 
        validation_data=(x_test, x_test),
        callbacks=[cp, stop, csv_logger, reduce_lr, vis, db_cb]
    )

    # =========================================================
    # 6. Generate and Save Embeddings
    # =========================================================
    dsdb.update_job(job_id, message="Generating latent vectors for all images...", progress=95)
    
    # Reload best weights to ensure optimal encoding
    vae.load_weights(wpath)
    
    # Predict latent space for ALL images (training + test)
    # This generates the vectors you need for UMAP/t-SNE later.
    latent_vectors = encoder.predict(x_all, batch_size=100)
    
    # Save to HDF5
    with h5py.File(embeddings_path, 'w') as f:
        f.create_dataset('vectors', data=latent_vectors, compression="gzip")
        # Save indices to map back to original data if needed
        f.create_dataset('indices', data=np.arange(len(latent_vectors)))

    print(f"Saved {len(latent_vectors)} vectors to {embeddings_path}")
    
    return True