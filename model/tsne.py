#!/usr/bin/env python

'''
Run t-SNE on the latent dim.
Refactored for integration with dataset jobs.
'''

from sklearn.manifold import TSNE
import numpy as np
import h5py
import json
import os
import time

def run_tsne(dataset_id, latent_dim, perplexity):
    """
    Reads latent vectors, runs t-SNE for a specific perplexity, and saves results.
    """
    
    # 1. Setup Paths
    # Input: data/<id>/models/<dim>/latent_vectors.h5
    base_dir = f'./data/{dataset_id}/models/{latent_dim}'
    inpath = os.path.join(base_dir, 'latent_vectors.h5')
    
    # Output: data/<id>/models/<dim>/tsne_perp<perp>.h5 (and .json)
    out_name = f'tsne_perp{perplexity}'
    h5_path = os.path.join(base_dir, f'{out_name}.h5')
    json_path = os.path.join(base_dir, f'{out_name}.json')

    if not os.path.exists(inpath):
        raise FileNotFoundError(f"Latent vectors not found at {inpath}")

    # 2. Load Data
    print(f"Loading data from {inpath}...")
    with h5py.File(inpath, 'r') as f:
        if 'vectors' not in f:
            raise ValueError("Key 'vectors' not found in HDF5 file.")
        data = f['vectors'][:]

    # 3. Run t-SNE
    print(f"Starting t-SNE (dim={latent_dim}, perp={perplexity})...")
    start_time = time.time()
    
    # Standard Euclidean t-SNE (usually preferred for latent spaces unless normalized)
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=1000)
    result = tsne.fit_transform(data)
    
    duration = time.time() - start_time
    print(f"t-SNE finished in {duration:.2f}s. KL Divergence: {tsne.kl_divergence_}")

    # 4. Save Results (HDF5)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('tsne', data=result, compression="gzip")
        f.create_dataset('kl_divergence', data=tsne.kl_divergence_)

    # 5. Save Results (JSON) for frontend compatibility
    res = []
    for i, point in enumerate(result):
        res.append({
            'i': i,
            'x': float(f"{point[0]:.3f}"),
            'y': float(f"{point[1]:.3f}")
        })
        
    with open(json_path, 'w') as outfile:
        json.dump(res, outfile)

    return tsne.kl_divergence_