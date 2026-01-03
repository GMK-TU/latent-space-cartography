#!/usr/bin/env python

'''
Run PCA on the latent dim.
Refactored for integration with dataset jobs.
'''

from sklearn.decomposition import PCA
import h5py
import json
import os
import numpy as np

def run_pca(dataset_id, latent_dim):
    """
    Reads latent vectors from HDF5, performs 2D PCA, and saves the results.
    
    Args:
        dataset_id (str): The dataset identifier.
        latent_dim (int): The specific latent dimension to process.
        
    Returns:
        dict: Summary stats (explained variance) or None on failure.
    """
    
    # 1. Setup Paths matching new_train.py structure
    # data/<id>/models/<dim>/latent_vectors.h5
    base_dir = f'./data/{dataset_id}/models/{latent_dim}'
    inpath = os.path.join(base_dir, 'latent_vectors.h5')
    
    # Outputs
    # data/<id>/models/<dim>/pca_2d.json
    # data/<id>/models/<dim>/pca_2d.h5
    json_path = os.path.join(base_dir, 'pca_2d.json')
    h5_path = os.path.join(base_dir, 'pca_2d.h5')

    if not os.path.exists(inpath):
        raise FileNotFoundError(f"Latent vectors not found at {inpath}. Did you run training?")

    # 2. Load Data
    with h5py.File(inpath, 'r') as f:
        if 'vectors' not in f:
            raise ValueError(f"Key 'vectors' not found in {inpath}")
        # Load all vectors into memory
        data = f['vectors'][:] 

    # 3. Perform PCA
    # We only want 2 components for the 2D viewer
    pca = PCA(n_components=2)
    result = pca.fit_transform(data)
    variance_ratio = pca.explained_variance_ratio_

    # 4. Save HDF5 (Fast Bulk Write)
    # The original code wrote row-by-row; bulk write is standard and faster.
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('pca', data=result, compression="gzip")
        # Store components and mean to allow "backward projection" later
        f.create_dataset('components', data=pca.components_)
        f.create_dataset('mean', data=pca.mean_)
        f.create_dataset('explained_variance', data=variance_ratio)

    # 5. Save JSON
    # Creates a lightweight list of objects for the frontend
    json_res = []
    for i, point in enumerate(result):
        json_res.append({
            'i': i,
            'x': float(f"{point[0]:.3f}"),
            'y': float(f"{point[1]:.3f}")
        })

    with open(json_path, 'w') as outfile:
        json.dump(json_res, outfile)

    print(f"PCA (dim={latent_dim}) finished. Explained Variance: {variance_ratio}")
    return variance_ratio.tolist()