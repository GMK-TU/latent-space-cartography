#!flask/bin/python
import json
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree
from sklearn import preprocessing
from scipy.stats import norm
import umap
import pickle
import numpy as np
import h5py
import sys
import os
import time
import csv
import shutil
import threading
import zipfile
import uuid
import math
from werkzeug.utils import secure_filename
from datasets_db import DatasetsDB
from datasets_import import import_metadata_csv
from datasets_schema import ensure_dataset_feature_tables
import pandas as pd
from PIL import Image
import importlib.util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.new_train import train_vae
from model.pca import run_pca
from model.tsne import run_tsne

# ugly way to import a file from another directory ...
sys.path.append(os.path.join(os.path.dirname(__file__), '../model'))
import model

from flask import Flask, send_from_directory, send_file
from flask import request, jsonify, abort
import sqlite3

# dataset we're working with
from config_glove_6b import *

# re-use keras models
models = {}
# re-use umap fit
umap_fit = {}
umap_seed = 22

# paths
P_TEMP = './data/temp/'

# for absolute path
def abs_path (rel_path):
    return os.path.join(os.path.dirname(__file__), rel_path)

# wrapper class for sqlite3 database
class DB:
    conn = None
    filename = abs_path('./data/lsc.db')

    def execute(self, query):
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
        except: #TODO: handle specific error
            self.conn = sqlite3.connect(self.filename)
            print('sqlite3 connected!')
            cursor = self.conn.cursor()
            cursor.execute(query)
        return cursor, self.conn
    
    def safe_commit(self, conn, cursor):
        try:
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
        finally:
            cursor.close()

# things to do right before starting the server
def init_server ():
    # create the temp folder to store interpolation images
    p_temp = abs_path(P_TEMP)
    if os.path.exists(p_temp):
        shutil.rmtree(p_temp)
    os.makedirs(p_temp)

# instantiate the model from our image generation VAE
def create_model (latent_dim):
    base = './data/{}/models/{}/'.format(dset, latent_dim)
    mpath = abs_path(base + '{}_model_dim={}.json'.format(dset, latent_dim))
    wpath = abs_path(base + '{}_model_dim={}.h5'.format(dset, latent_dim))
    m = model.Vae(latent_dim = latent_dim, img_dim=(img_chns, img_rows, img_cols))
    models[latent_dim] = m.read(mpath, wpath) + (m,)

# instantiate the model that's simply an h5py file
def load_model (latent_dim):
    from keras.models import load_model

    mpath = './data/{}/models/{}_model_dim={}.h5'.format(dset, dset, latent_dim)
    decoder = load_model(mpath)
    models[latent_dim] = decoder

# read latent space
def read_ls (latent_dim):
    rawpath = abs_path('./data/{}/latent/latent{}.h5'.format(dset, latent_dim))
    df = pd.read_hdf(rawpath, key="latent")
    X = np.copy(df.drop(columns=['word']))

    #    Original version. Check you do generlize
    #    with h5py.File(rawpath, 'r') as f:
    #        X = np.asarray(f['latent'])
    return X

def read_raw ():
    p_raw = abs_path('./data/{}/raw.h5'.format(dset))
    with h5py.File(p_raw, 'r') as f:
        X = np.asarray(f['data'])
    return X

# read csv, discarding (optionally) the first row
def read_csv (fn, row_start=1):
    res = []
    if not os.path.exists(fn):
        return res
    with open(fn, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            res.append(row)
    return res[row_start:]

# given a list of points in latent space, generate their corresponding images
def _generate_image (latent_dim, points):
    if not latent_dim in models:
        create_model(latent_dim)
    vae, encoder, decoder, m = models[latent_dim]

    print('Predicting ...')
    images = []
    for idx, val in enumerate(points):
        val = val.reshape((1, latent_dim))
        recon = m.to_image(decoder.predict(val))
        img = Image.fromarray(recon, img_mode)
        images.append(img)
    print('Done.')

    return images

# given a list of points in latent space, reconstruct via decoder
# the reconstruction results are just arbitrary tensors
def _generate_other (latent_dim, points):
    if not latent_dim in models:
        load_model(latent_dim)
    decoder = models[latent_dim]
    points = np.asarray(points, float)
    return decoder.predict(points)

# given a vector w, return the index in top and bottom quantile
# the quantile is computed as highsd away in a standard normal distribution
def _top_and_bottom (w, highsd=2.5):
    n = w.shape[0]
    cutoff = n - int(n * norm.cdf(highsd))
    srt = np.argsort(w)
    w = w.tolist() # numpy float32 is not JSON serializable
    pos = [{'i': i, 'diff': w[i]} for i in srt[-cutoff:]]
    neg = [{'i': i, 'diff': w[i]} for i in srt[:cutoff]]
    return pos[::-1], neg

# number of points within L2 distance of a given point
# also return the nearest neighbor
def _num_neighbors (X, points, distance = 3.0):
    count = []
    nearest = []
    n, latent_dim = X.shape
    for idx, p in enumerate(points):
        P = np.repeat(p.reshape(1, -1), n, axis = 0)
        # d = np.abs(X - P)
        d = np.linalg.norm(X - P, axis = 1) # L2 distance
        qualify = np.less_equal(d, np.repeat(distance, n))
        indices = np.where(qualify)[0]
        count.append(len(indices))
        nearest.append(np.argmin(d))
    return count, nearest

# sample points along a vector
def _sample_vec (start, end, n_samples = 8, over = True):
    loc = []
    for i in range(n_samples + 1):
        k = float(i) / n_samples
        loc.append((1-k) * start + k * end)

    # overshoot
    if over:
        k = 1.5
        loc.append((1-k) * start + k * end)
    return loc

# interpolate between two points in a latent space
# return a list of reconstructed outputs sampled at equal steps along the path
def _interpolate (X, start, end):
    n, latent_dim = X.shape
    if data_type == 'image':
        loc = _sample_vec(start, end)
        # generate these images
        recon = _generate_image(latent_dim, loc)
    elif data_type == 'other':
        loc = _sample_vec(start, end, 1, False)
        # generate tensor outputs
        recon = _generate_other(latent_dim, loc)
    else:
        loc = _sample_vec(start, end, 1, False)
        recon = None
    count, nn = _num_neighbors(X, loc)

    return loc, recon, count, nn

# linear orthogonal transformation of all points to the given axis
def _project_axis (X, axis):
    n, latent_dim = X.shape

    # 1. make the axis a unit vector
    axis = np.asarray(axis, dtype=np.float64)
    v = preprocessing.normalize(axis.reshape(1, -1))

    # v is a row vector of shape (1, latent_dim)
    if v.shape[1] != latent_dim:
        print('Could not project to axis because axis and latent dimension shape mismatch.')
        return jsonify({'status': 'fail'}), 200

    # 2. center X to mean
    mean_ = np.mean(X, axis=0)
    X -= mean_

    # 3. substract the first axis from X (project X to the d-1 orthogonal space of axis)
    X_hat = X - np.dot(X, np.dot(v.T, v))

    # 4. perform PCA
    pca = PCA(n_components = 2)
    pca.fit(X_hat)
    y = pca.components_[0]
    va = pca.explained_variance_ratio_

    print('Explained variance ratio: {}'.format(va))

    U = np.append(v, y.reshape(1, -1), axis=0)
    X_transformed = np.dot(X, U.T)

    # compute the variation of v
    # FIXME: the variance doesn't seem correct
    print('Explained variance: {}'.format(pca.explained_variance_))
    total_var = pca.explained_variance_.sum()
    print('Total variance: {}'.format(total_var))
    s = np.dot(X, v.T)
    s = np.sum(s ** 2) / (n - 1)
    print('Variance of x axis: {}, {}%'.format(s, s / total_var))

    return X_transformed, U, mean_

# given a group ID, query the DB for image indices, as an int array
def _get_group_indices (gid):
    # find image indices in each group
    cursor, conn = db.execute('SELECT list FROM {}_group WHERE id={}'.format(dset, gid))
    d = cursor.fetchone()[0]
    id_list = d.split(',')

    # compute centroid
    indices = np.asarray(id_list, dtype=np.int16)
    return indices

# compute the centroid of a group
def _compute_group_centroid (X, gid):
    indices = _get_group_indices(gid)
    centroid = np.sum(X[indices], axis=0) / indices.shape[0]

    return centroid

# compute the average inter-point distance (L2) between each point pair
def _pointwise_dist (X, Y=None):
    R = X if Y is None else Y
    m, _ = X.shape
    n, _ = R.shape

    s = 0
    for i in range(m):
        # left hand matrix: repeat an element N times
        L = np.repeat([X[i]], n, axis=0)
        D = np.linalg.norm(L - R, axis=1)
        # for intra-cluster distance, exclude self
        denom = n - 1 if Y is None else n
        s += np.sum(D) / float(denom)
    
    return s / float(m)

# compute k nearest neighbors using cosine distance
def _knn_cosine (X, v, kn = 20):
    tree = KDTree(preprocessing.normalize(X))
    _, idx = tree.query(v, k=kn)
    dist = cosine_similarity(X[idx[0]], np.repeat(v, kn, axis=0))
    return dist[:, 0], idx[0]

# compute the screen coordinates of given paths in a global projection
def _project_path (X, projection, locs, params={}):
    # t-SNE: use the coordinate of nearest neighbors
    if projection == 'tsne':
        # use kd-tree to compute k nearest neighbors
        if metric == 'cosine':
            tree = KDTree(preprocessing.normalize(X))
        else:
            tree = KDTree(X)

        # read t-SNE coordinates
        perp = params['perplexity']
        latent_dim = params['latent_dim']
        tpath = abs_path('./data/{}/tsne/tsne{}_perp{}.h5'.format(dset, latent_dim, perp))
        with h5py.File(tpath, 'r') as f:
            Y = np.asarray(f['tsne']) # shape: (n, 2)

        result = []
        for loc in locs:
            # k nearest neighbors
            kn = 1 if data_type == 'text' else 5 #FIXME
            dist, idx = tree.query(loc, k=kn)
            res = []
            for i in range(idx.shape[0]):
                # weighted average
                res.append(np.average(Y[idx[i]], weights=1/dist[i], axis=0))
            # remove duplicate control points
            res = np.asarray(res)
            dup = np.linalg.norm(res[1:] - res[:-1], axis=1)
            dedup = [res[0]]
            for i in range(idx.shape[0] - 1):
                if dup[i] > 0.00001:
                    dedup.append(res[i + 1])
            result.append(np.asarray(dedup).tolist())
        return result

    elif projection == 'umap':
        n_neighbors = params['n_neighbors']
        min_dist = params['min_dist']
        latent_dim = params['latent_dim']
        fit = _fit_umap(latent_dim, n_neighbors, min_dist)
        res = []
        for loc in locs:
            res.append(fit.transform(loc).tolist())
        return res

    # PCA: multiply projection matrix directly
    elif projection == 'pca':
        pca_dim = params['pca_dim']
        pca = PCA(n_components = pca_dim).fit(X)
        res = []
        for loc in locs:
            res.append(pca.transform(loc).tolist())
        return res

    # Custom vector projection: multiply custom matrix
    elif projection == 'vector':
        U = np.asarray(params['matrix'], dtype=np.float64)
        _mean = np.asarray(params['mean'], dtype=np.float64)
        res = []
        for loc in locs:
            res.append(np.dot(loc - _mean, U.T).tolist())
        return res

    return []

# fit umap to data
def _fit_umap (latent_dim, nn, dist):
    key = '{}-{}'.format(nn, dist)
    if key in umap_fit:
        return umap_fit[key]
    X = read_ls(latent_dim)
    d = umap.UMAP(n_neighbors=nn, min_dist=dist,
                  random_state=umap_seed).fit(X)
    umap_fit[key] = d
    return d

# pairwise cosine similarity between random pairs in the latent space
# this is precomputed
def _random_pairs (latent_dim):
    fn = abs_path('./data/{}/pairs.h5'.format(dset))
    with h5py.File(fn, 'r') as f:
        cs = np.asarray(f['cosine{}'.format(latent_dim)])
    return cs

# relative formulation of pairwise cosine
def _pair_alignment (latent_dim, gid):
    # read latent space
    X = read_ls(latent_dim)
    
    # data points in start and end group
    start = X[_get_group_indices(gid[0])]
    end = X[_get_group_indices(gid[1])]
    n, _ = start.shape
    m, _ = end.shape

    if (m == n):
        # one-to-one pairs
        V = end - start
    else:
        # all possible vector pairs between start and end
        L = np.repeat(start, m, axis=0)
        R = np.tile(end, (n, 1))
        V = L - R

    # cosine similarity
    cs = cosine_similarity(V)

    # we want only the lower triangle (excluding the diagonal)
    cs = np.tril(cs, k=-1)
    cs = cs[np.nonzero(cs)]

    # relative formulation: random pairs
    csr = _random_pairs(latent_dim)

    # effect size
    n1 = cs.shape[0]
    n2 = csr.shape[0]
    s1 = np.std(cs)
    s2 = np.std(csr)
    pooled = np.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    cohen = (np.mean(cs) - np.mean(csr)) / pooled

    return cs, csr, cohen, pooled

# global app and DB cursor
app = Flask(__name__, static_url_path='')
db = DB()
# init dataset registry DB (datasets + jobs tables)
_ds_db = DatasetsDB(db.filename)

# static files
@app.route('/')
def index ():
    return send_file('index.html')

# static files
@app.route('/build/<path:path>')
def serve_public (path):
    return send_from_directory('build', path)

@app.route('/data/<path:path>')
def serve_data (path):
    return send_from_directory('data', path)

# get umap data
@app.route('/api/get_umap', methods=['POST'])
def get_umap ():
    latent_dim = request.json['latent_dim']
    nn = request.json['n_neighbors']
    dist = request.json['min_dist']

    pkl_path = abs_path('./data/{}/umap/umap{}-nn{}-dist{}.pkl').format(dset, latent_dim, nn, dist)
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
            from umap.nndescent import make_initialisations, make_initialized_nnd_search
            data._random_init, data._tree_init = make_initialisations(
                data._distance_func, data._dist_args
            )
            data._search = make_initialized_nnd_search(
                data._distance_func, data._dist_args
            )
            umap_fit['{}-{}'.format(nn, dist)] = data
            d = data.embedding_
    else:
        d = _fit_umap(latent_dim, nn, dist).embedding_
    return jsonify({'data': d.tolist()}), 200

# get pca data
@app.route('/api/get_pca', methods=['POST'])
def get_pca():
    if not request.json or 'dataset_id' not in request.json:
        abort(400)

    if not request.json or 'latent_dim' not in request.json:
        abort(400)

    dsid, payload = _get_dataset_id_from_request()
    
    latent_dim = request.json['latent_dim']
    pca_dim = int(request.json['pca_dim'])
    indices = np.asarray(request.json['indices'], dtype=np.int16)

    rawpath = abs_path(f'./data/{dsid}/models/{latent_dim}/latent_vectors.h5')

    if not os.path.exists(rawpath):
         return jsonify({'error': f'Latent file not found at {rawpath}'}), 404

    try:
        with h5py.File(rawpath, 'r') as f:
            if 'vectors' not in f:
                return jsonify({'error': 'Key "vectors" not found in H5 file'}), 500
            raw = f['vectors'][:]
    except Exception as e:
        return jsonify({'error': f'Failed to read HDF5: {str(e)}'}), 500

    length = indices.shape[0]
    if length > 0:
        raw = raw[indices]

    pca = PCA(n_components=pca_dim)
    d = pca.fit_transform(raw)
    va = pca.explained_variance_ratio_

    print(f'Explained variation per principal component: {va}')

    return jsonify({'data': d.tolist(), 'variation': va.tolist()}), 200

# PCA backward projection
@app.route('/api/pca_back', methods=['POST'])
def pca_back ():
    if not request.json:
        abort(400)
    
    latent_dim = request.json['latent_dim']
    x = float(request.json['x'])
    y = float(request.json['y'])
    i = int(request.json['i'])
    
    # project from 2D to latent space
    rawpath = abs_path('./data/{}/latent/latent{}.h5'.format(dset, latent_dim))
    with h5py.File(rawpath, 'r') as f:
        raw = f['latent']
        pca = PCA(n_components=2)
        pca.fit(raw)

        d = pca.transform(raw)
        d[i] = [x, y]

        re = pca.inverse_transform(d)

    # project from latent space to image
    img =  _generate_image(latent_dim, re[i:i+1])[0]
    img_fn = '{}.png'.format(int(time.time()))
    img.save(abs_path(P_TEMP + img_fn))

    return jsonify({'latent': re[i].tolist(), 'image': img_fn}), 200

# get tsne data
@app.route('/api/get_tsne', methods=['POST'])
def get_tsne():
    if not request.json or 'latent_dim' not in request.json:
        abort(400)
    
    dsid, payload = _get_dataset_id_from_request()
    
    latent_dim = request.json['latent_dim']
    perp = request.json.get('perplexity', 30) # Default to 30 if missing
    
    fn = abs_path(f'./data/{dsid}/models/{latent_dim}/tsne_perp{perp}.h5')
    
    if not os.path.exists(fn):
        return jsonify({'error': f't-SNE file not found: {fn}'}), 404

    try:
        with h5py.File(fn, 'r') as f:
            if 'tsne' not in f:
                 return jsonify({'error': 'Key "tsne" not found in H5 file'}), 500
            data = f['tsne'][:]
    except Exception as e:
        return jsonify({'error': f'Failed to read HDF5: {str(e)}'}), 500

    return jsonify({'data': data.tolist()}), 200

# get meta data
@app.route('/api/get_meta', methods=['POST'])
def get_meta ():
    dsid, payload = _get_dataset_id_from_request()

    query = 'SELECT {} FROM {}_meta'.format(schema_meta, dsid)
    cursor, conn = db.execute(query)
    data = [list(i) for i in cursor.fetchall()]
    reply = {'meta': data}

    # header is meta data on input data columns
    if schema_header:
        query = 'SELECT {} FROM {}_header'.format(schema_header, dsid)
        cursor.execute(query)
        header = [list(i) for i in cursor.fetchall()]
        reply['header'] = header

    return jsonify(reply), 200

# apply analogy
@app.route('/api/apply_analogy', methods=['POST'])
def apply_analogy ():
    latent_dim = request.json['latent_dim']
    pid = request.json['pid']
    gid = request.json['groups'].split(',')

    U = np.asarray(request.json['projection'], dtype=np.float64)
    _mean = np.asarray(request.json['mean'], dtype=np.float64)

    # read latent space
    rawpath = abs_path('./data/{}/latent/latent{}.h5'.format(dset, latent_dim))

    df = pd.read_hdf(rawpath, key="latent")
    X = np.copy(df.drop(columns=['word']))

#   Original version. Check how to generalize.
#    with h5py.File(rawpath, 'r') as f:
#        X = np.asarray(f['latent'])

    # compute centroid
    vec = _compute_group_centroid(X, gid[1]) - _compute_group_centroid(X, gid[0])

    start = X[int(pid)]
    end = start + vec

    if data_type == 'image':
        loc, images, count, nearest = _interpolate(X, start, end)
        fns = []
        for idx, img in enumerate(images):
            img_fn = 'analogy_{}_{}_{}.png'.format(latent_dim, pid, idx)
            fns.append(img_fn)
            img.save(abs_path(P_TEMP + img_fn))
        
        reply = { 'outputs': fns }
    else:
        loc, _, count, nearest = _interpolate(X, start, end)
        dist, idx = _knn_cosine(X, end.reshape(1, -1))
        reply = { 'knn_indices': idx.tolist(), 'knn_distances': dist.tolist() }

    loc = np.dot(loc - _mean, U.T)
    reply['locations'] = loc.tolist()
    reply['neighbors'] = count
    reply['nearest'] = nearest

    return jsonify(reply), 200

# visualize vectors together in a global projection
@app.route('/api/plot_vectors', methods=['POST'])
def plot_vectors ():
    latent_dim = request.json['latent_dim']
    projection = request.json['projection']
    vectors = request.json['vectors'].split(';')

    # read latent space
    X = read_ls(latent_dim)

    locs = []
    for gids in vectors:
        # compute centroid
        gid = gids.split(',')
        start = _compute_group_centroid(X, gid[0])
        end = _compute_group_centroid(X, gid[1])
        locs.append(_sample_vec(start, end, over=False))
    res = _project_path(X, projection, locs, request.json)

    return jsonify({'status': 'success', 'data': res}), 200

# visualize pairs together in a global projection
@app.route('/api/plot_pairs', methods=['POST'])
def plot_pairs ():
    latent_dim = request.json['latent_dim']
    projection = request.json['projection']
    pairs = request.json['pairs'].split(';')

    # read latent space
    X = read_ls(latent_dim)

    locs = []
    for pair in pairs:
        pair = [int(x) for x in pair.split(',')]
        locs.append(_sample_vec(X[pair[0]], X[pair[1]], over=False))
    res = _project_path(X, projection, locs, request.json)

    return jsonify({'status': 'success', 'data': res}), 200

# bring a vector to focus: interpolate along the path, and reproject all points
@app.route('/api/focus_vector', methods=['POST'])
def focus_vector():
    latent_dim = request.json['latent_dim']
    gid = request.json['groups'].split(',')
    reply = {}

    # read latent space
    X = read_ls(latent_dim)
    
    # compute centroid
    start = _compute_group_centroid(X, gid[0])
    end = _compute_group_centroid(X, gid[1])
    vec = end - start

    # project
    X_transformed, U, _mean = _project_axis(np.copy(X), vec)
    reply['points'] = X_transformed.tolist()
    reply['mean'] = _mean.tolist()
    reply['projection'] = U.tolist()

    # interpolate
    if data_type == 'image':
        loc, images, count, nearest = _interpolate(X, start, end)
        recon = []
        for idx, img in enumerate(images):
            img_fn = '{}_{}_{}.png'.format(latent_dim, 'to'.join(gid), idx)
            recon.append(img_fn)
            img.save(abs_path(P_TEMP + img_fn))
    elif data_type == 'other':
        loc, recon, count, nearest = _interpolate(X, start, end)
        # high weight genes
        diff = recon[-1] - recon[0]
        reply['top_end'], reply['top_start'] = _top_and_bottom(diff)
        recon = recon.tolist()
    else:
        loc, recon, count, nearest = _interpolate(X, start, end)

    loc = np.dot(loc - _mean, U.T)
    reply['locations'] = loc.tolist()
    reply['neighbors'] = count
    reply['nearest'] = nearest
    reply['outputs'] = recon

    return jsonify(reply), 200

@app.route('/api/vector_diff', methods=['POST'])
def vector_diff ():
    latent_dim = request.json['latent_dim']
    vid = request.json['vid']

    # get all attribute vectors from database
    query = 'SELECT a.start, a.end, a.id FROM {}_vector a'.format(dset)
    cursor, conn = db.execute(query)
    data = [list(i) for i in cursor.fetchall()]

    # compute vector coordinates
    X = read_ls(latent_dim)
    vecs = []
    idx = 0
    for i, v in enumerate(data):
        if v[2] == vid:
            idx = i
        vecs.append(_compute_group_centroid(X, v[1]) - _compute_group_centroid(X, v[0]))
    vecs = np.asarray(vecs)

    # compute cosine similarity between this vector and all others
    cos = []
    for i in range(len(vecs)):
        sim = cosine_similarity(vecs[i].reshape(1, -1), vecs[idx].reshape(1, -1))[0][0]
        cos.append({'id': data[i][2], 'cosine': sim})

    return jsonify({'data': cos}), 200

@app.route('/api/all_vector_diff', methods=['POST'])
def all_vector_diff ():
    # get all attribute vectors from database
    query = 'SELECT a.start, a.end FROM {}_vector a'.format(dset)
    cursor, conn = db.execute(query)
    data = [list(i) for i in cursor.fetchall()]

    # compute vector coordinates in each latent space
    vecs = {}
    for dim in dims:
        X = read_ls(dim)
        arr = []
        for v in data:
            arr.append(_compute_group_centroid(X, v[1]) - _compute_group_centroid(X, v[0]))
        vecs[dim] = np.asarray(arr)

    # compute cosine similarity between each possible vector pair
    cos = {}
    for dim in dims:
        vs = vecs[dim]
        arr = []
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                arr.append(cosine_similarity(vs[i].reshape(1, -1), vs[j].reshape(1, -1))[0][0])
        cos[dim] = np.asarray(arr)

    # compare each adjacent latent dim
    for i in range(len(dims) - 1):
        L = cos[dims[i]]
        R = cos[dims[i + 1]]
        diff = np.sum(np.abs(R - L)) / float(L.shape[0])
        print('{} and {}: {}'.format(dims[i], dims[i + 1], diff))

    return jsonify({'status': 'success'}), 200

# pairwise cosine similarity within an attribute vector
@app.route('/api/vector_score', methods=['POST'])
def vector_score ():
    latent_dim = request.json['latent_dim']
    gid = request.json['groups'].split(',')

    cs, csr, cohen, pooled = _pair_alignment(latent_dim, gid)

    # histograms
    hist, _ = np.histogram(cs, bins=np.arange(-1.0, 1.05, 0.1))
    histr, _ = np.histogram(csr, bins=np.arange(-1.0, 1.01, 0.02))

    mean = np.mean(cs)
    print('Vector score (GID {} & {}): average {}, max {}, min {}'
          .format(gid[0],gid[1], round(mean, 2), round(np.amax(cs), 2),  round(np.amin(cs), 2)))
    print('Cohen\'s d: {}, pooled sd: {}'.format(cohen, pooled))
 
    reply = {'mean': mean, 'cohen': cohen, 'unit': pooled}
    if 'histogram' in request.json:
        reply['histogram'] = hist.tolist()
        reply['random'] = histr.tolist()

    return jsonify(reply), 200

# compute a number to represent how tight a cluster is
@app.route('/api/cluster_score', methods=['POST'])
def cluster_score ():
    latent_dim = request.json['latent_dim']
    ids = request.json['ids']

    X = read_ls(latent_dim)
    a = _pointwise_dist(X[ids])
    b = _pointwise_dist(X[ids], np.delete(X, ids, axis=0))
    print('Intra-cluster distance: {}, Inter-cluster distance: {}'.format(a, b))
    # this score resembles silhouette score, but it replaces inter-cluster
    # distance with the average length of all edges with one node inside and one outside.
    score = (b - a) / max(a, b)

    return jsonify({'score': score}), 200

# get the k nearest neighbors of a point
@app.route('/api/get_knn', methods=['POST'])
def get_knn ():
    i = request.json['i']
    latent_dim = request.json['latent_dim']
    X = read_ls(latent_dim)
    dist, idx = _knn_cosine(X, X[i].reshape(1, -1))
    return jsonify({'knn_indices': idx.tolist(), 'knn_distances': dist.tolist()}), 200

# get the raw (input) data for a given index
# useful if the data type is arbitrary vector
@app.route('/api/get_raw', methods=['POST'])
def get_raw ():
    i = request.json['i']
    X = read_raw()
    return jsonify({'data': X[i].tolist()}), 200

# save a group
@app.route('/api/save_group', methods=['POST'])
def save_group ():
    if not request.json or not 'ids' in request.json:
        abort(400)

    ids = request.json['ids']
    alias = request.json['alias'] if 'alias' in request.json else ''

    query = """
    INSERT INTO {}_group (alias, list)
    VALUES('{}', '{}')
    """.format(dset, alias, ids)
    print(query)

    cursor, conn = db.execute(query)
    db.safe_commit(conn, cursor)
    return jsonify({'status': 'success'}), 200

# get all groups
@app.route('/api/get_groups', methods=['POST'])
def get_groups ():
    query = 'SELECT id, alias, list, timestamp FROM {}_group'.format(dset)
    cursor, conn = db.execute(query)
    data = [list(i) for i in cursor.fetchall()]
    return jsonify({'data': data[::-1]}), 200

# delete a group
@app.route('/api/delete_group', methods=['POST'])
def delete_group ():
    gid = request.json['id']
    query = 'DELETE FROM {}_group WHERE id={}'.format(dset, gid)
    print(query)

    cursor, conn = db.execute(query)
    db.safe_commit(conn, cursor)
    return jsonify({'status': 'success'}), 200

# create a vector
@app.route('/api/create_vector', methods=['POST'])
def create_vector():
    start = request.json['start']
    end = request.json['end']
    desc = request.json['desc']

    query = """INSERT INTO {}_vector (start, end, description)
    VALUES('{}', '{}', '{}')""".format(dset, start, end, desc)
    print(query)

    cursor, conn = db.execute(query)
    db.safe_commit(conn, cursor)
    return jsonify({'status': 'success'}), 200

# get all vectors
@app.route('/api/get_vectors', methods=['POST'])
def get_vectors ():
    dsid, payload = _get_dataset_id_from_request()

    query = """
    SELECT a.id, a.description, a.timestamp, a.start, a.end, b.list AS list_start,
      c.list AS list_end, b.alias AS alias_start, c.alias AS alias_end
    FROM {}_vector a
    LEFT OUTER JOIN (SELECT id, list, alias FROM {}_group) AS b ON a.start = b.id
    LEFT OUTER JOIN (SELECT id, list, alias FROM {}_group) AS c ON a.end = c.id
    """.format(dsid, dsid, dsid)
    cursor, conn = db.execute(query)
    data = [list(i) for i in cursor.fetchall()]
    return jsonify({'data': data[::-1]}), 200

# delete a vector
@app.route('/api/delete_vector', methods=['POST'])
def delete_vector ():
    vid = request.json['id']

    query = 'DELETE FROM {}_vector WHERE id={}'.format(dset, vid)
    print(query)

    cursor, conn = db.execute(query)
    db.safe_commit(conn, cursor)
    return jsonify({'status': 'success'}), 200

# cross-comparison
@app.route('/api/get_compare_page', methods=['POST'])
def get_compare_page ():
    initial = read_csv(abs_path('./data/{}/initial.csv').format(dset))
    vecs = read_csv(abs_path('./data/{}/vector_scores.csv').format(dset))
    reply = {'initial': initial, 'vectors': vecs}

    if data_type == 'image':
        # summary distribution on each axis
        res = []
        for dim in dims:
            X = read_ls(dim)
            q25 = np.percentile(X, 25, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q0 = X.min(axis=0)
            q100 = X.max(axis=0)
            res.append(np.asarray([q0, q25, q75, q100]).tolist())
        reply['axes'] = res
    return jsonify(reply), 200

# load client side config
@app.route('/api/load_config', methods=['POST'])
def load_config ():
    # read json
    fn = abs_path('./configs/config_{}.json'.format(dset))
    with open(fn, 'r') as f:
        cfg = json.load(f)

    # add more fields
    cfg['dataset'] = dset
    cfg['data_type'] = data_type
    cfg['dims'] = dims
    cfg['schema']['meta'] = [i.strip() for i in schema_meta.split(',')]
    if schema_header:
        cfg['schema']['header'] = [i.strip() for i in schema_header.split(',')]
    return jsonify({'config': cfg}), 200

@app.route('/api/_compare_vectors', methods=['POST'])
def _compare_vectors ():
    # get all attribute vectors
    query = 'SELECT a.start, a.end, a.id FROM {}_vector a'.format(dset)
    cursor, conn = db.execute(query)
    vecs = [list(i) for i in cursor.fetchall()]

    # compute pair alignment for all dims
    res = []
    for dim in dims:
        for gid in vecs:
            _, _, cohen, _ = _pair_alignment(dim, gid)
            print(dim, gid[2], cohen)
            res.append([dim, gid[2], cohen])

    # save to csv file
    out = abs_path('./data/{}/vector_scores.csv'.format(dset))
    if os.path.exists(out):
        os.remove(out)
    with open(out, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['dim', 'vector id', 'cohen d'])
        for row in res:
            writer.writerow(row)

    return jsonify({'status': 'success'}), 200

# ==========================
# DATASET IMPORT ENDPOINTS
# ==========================

def _require_ds_db():
    if _ds_db is None:
        # best effort: initialize lazily
        try:
            return DatasetsDB(db.filename)
        except Exception:
            return None
    return _ds_db

import re

_DATASET_ID_RE = re.compile(r"^[a-zA-Z0-9_]{3,64}$")

def _get_dataset_id_from_request():
    payload = request.get_json(silent=True) or {}
    dsid = payload.get("dataset_id") or dset  # fallback to legacy global
    if not isinstance(dsid, str) or not _DATASET_ID_RE.match(dsid):
        abort(400, description="Invalid dataset_id")
    return dsid, payload

def _dataset_root(dataset_id):
    return abs_path(f'./data/{dataset_id}')

def _ensure_dirs(dataset_id):
    root = _dataset_root(dataset_id)
    raw_dir = os.path.join(root, 'raw')
    meta_dir = os.path.join(root, 'meta')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    return root, raw_dir, meta_dir

def _preview_images(dataset_id, raw_dir, limit=12):
    exts = {'.jpg','.jpeg','.png','.gif','.webp','.bmp'}
    imgs = []
    for dirpath, _, filenames in os.walk(raw_dir):
        for fn in sorted(filenames):
            if os.path.splitext(fn.lower())[1] in exts:
                rel = os.path.relpath(os.path.join(dirpath, fn), abs_path('./data'))
                imgs.append(f'/data/{rel.replace(os.sep, "/")}')
                if len(imgs) >= limit:
                    return imgs
    return imgs

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500
    return jsonify(dsdb.list_datasets()), 200

@app.route('/api/datasets', methods=['POST'])
def create_dataset():

    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500

    payload = request.get_json(silent=True) or {}
    name = payload.get('name') or f'Dataset {time.strftime("%Y-%m-%d %H:%M:%S")}'
    dataset_id = payload.get('id') or f'ds{uuid.uuid4().hex[:12]}'

    _ensure_dirs(dataset_id)

    ensure_dataset_feature_tables(dsdb.conn, dataset_id)

    ds = dsdb.create_dataset(dataset_id=dataset_id, name=name)
    return jsonify(ds), 200

@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500
    ds = dsdb.get_dataset(dataset_id)
    if not ds:
        return jsonify({'error':'not found'}), 404
    return jsonify(ds), 200

@app.route('/api/datasets/<dataset_id>/raw-zip', methods=['POST'])
def upload_raw_zip(dataset_id):
    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500

    if 'file' not in request.files:
        return jsonify({'error':'missing file'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error':'empty filename'}), 400

    root, raw_dir, _ = _ensure_dirs(dataset_id)
    dsdb.update_dataset(dataset_id, status='uploading_raw', progress=0, message='Uploading ZIP...')

    up_dir = os.path.join(root, 'uploads')
    os.makedirs(up_dir, exist_ok=True)
    zip_name = secure_filename(f.filename)
    zip_path = os.path.join(up_dir, zip_name)
    f.save(zip_path)

    # extract zip into raw_dir (flattening not enforced)
    dsdb.update_dataset(dataset_id, status='uploading_raw', progress=10, message='Extracting ZIP...')
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(raw_dir)
    except zipfile.BadZipFile:
        dsdb.update_dataset(dataset_id, status='error', progress=0, message='Bad ZIP file.')
        return jsonify({'error':'bad zip'}), 400

    preview = _preview_images(dataset_id, raw_dir, limit=12)
    dsdb.update_dataset(dataset_id, status='raw_uploaded', progress=25, message='Images uploaded.', extra={'previewImages': preview})
    return jsonify({'previewImages': preview}), 200

@app.route('/api/datasets/<dataset_id>/metadata-csv', methods=['POST'])
def upload_metadata_csv(dataset_id):
    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500

    if 'file' not in request.files:
        return jsonify({'error':'missing file'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error':'empty filename'}), 400

    root, raw_dir, meta_dir = _ensure_dirs(dataset_id)
    dsdb.update_dataset(dataset_id, status='uploading_csv', progress=25, message='Uploading CSV…')

    csv_name = secure_filename(f.filename)
    csv_path = os.path.join(meta_dir, csv_name)
    f.save(csv_path)

    dsdb.update_dataset(dataset_id, status='uploading_csv', progress=28, message='Importing metadata into SQLite…')
    try:
        preview_meta, matched_preview = import_metadata_csv(db.filename, dataset_id, csv_path, raw_dir)
    except Exception as e:
        dsdb.update_dataset(dataset_id, status='error', progress=0, message='CSV import failed.', error=str(e))
        return jsonify({'error': str(e)}), 400

    dsdb.update_dataset(dataset_id, status='csv_uploaded', progress=35, message='Metadata uploaded.', extra={'previewMeta': preview_meta, 'matchedPreview': matched_preview})
    return jsonify({'previewMeta': preview_meta, 'matchedPreview': matched_preview}), 200

def _compute_worker(dataset_id, job_id):
    dsdb = _require_ds_db()
    if dsdb is None:
        return

    # IMPORTANT: This is a scaffold. Replace the sleeps with real computation calls:
    # - build latent space
    # - PCA, UMAP, t-SNE
    # - write artifacts to ./data/<dataset_id>/...
    stages = [
        ('latent', 0, 40, 2.0),
        ('pca', 40, 70, 2.0),
        ('umap', 70, 90, 2.0),
        ('finalize', 90, 100, 1.0),
    ]

    dsdb.update_dataset(dataset_id, status='computing', progress=35, message='Starting computations…')
    for stage, p0, p1, seconds in stages:
        dsdb.update_job(job_id, stage=stage, progress=p0, message=f'Computing {stage}…')
        steps = 10
        for k in range(steps):
            time.sleep(seconds / steps)
            prog = int(p0 + (p1 - p0) * ((k + 1) / steps))
            dsdb.update_job(job_id, stage=stage, progress=prog, message=f'Computing {stage}…')
            dsdb.update_dataset(dataset_id, status='computing', progress=min(100, max(35, prog)), message=f'Computing {stage}…')

    dsdb.update_job(job_id, status='done', stage='done', progress=100, message='Done.', done=True)
    dsdb.update_dataset(dataset_id, status='ready', progress=100, message='Ready.')

def _start_job_thread(dsdb, dataset_id: str, worker_fn, *, params=None, job_message="Queued…", job_stage="queued"):
    job_id = uuid.uuid4().hex
    dsdb.create_job(job_id=job_id, dataset_id=dataset_id, progress=0, message=job_message, stage=job_stage)

    # ensure params is serializable dict
    params = params or {}

    t = threading.Thread(target=worker_fn, args=(dataset_id, job_id, params), daemon=True)
    t.start()
    return job_id

def _run_job_safely(dsdb, dataset_id, job_id, fn, *, start_status="computing", start_msg="Starting…"):
    try:
        dsdb.update_dataset(dataset_id, status=start_status, message=start_msg)
        fn()
    except Exception as e:
        dsdb.update_job(job_id, status='error', stage='error', message=str(e), done=True)
        dsdb.update_dataset(dataset_id, status='error', message='Job failed.', error=str(e))
        raise

@app.route('/api/datasets/<dataset_id>/compute', methods=['POST'])
def start_compute(dataset_id):
    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500

    def worker(dataset_id, job_id, params):
        _compute_worker(dataset_id, job_id)

    job_id = _start_job_thread(dsdb, dataset_id, worker, job_message="Queued…", job_stage="queued")
    return jsonify({'jobId': job_id}), 200

@app.route('/api/datasets/<dataset_id>/vectorize', methods=['POST'])
def start_vectorize(dataset_id):
    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500

    params = request.get_json(silent=True) or {}

    def worker(dataset_id, job_id, params):
        dsdb_local = _require_ds_db()
        if dsdb_local is None:
            return
        def fn():
            make_dataset_job(dataset_id, job_id, params, dsdb_local)
        _run_job_safely(dsdb_local, dataset_id, job_id, fn, start_status="computing", start_msg="Vectorizing…")

    job_id = _start_job_thread(dsdb, dataset_id, worker, params=params, job_message="Queued (vectorize)…", job_stage="queued")
    return jsonify({'jobId': job_id}), 200

@app.route('/api/datasets/<dataset_id>/train', methods=['POST'])
def start_train(dataset_id):
    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500

    ds = dsdb.get_dataset(dataset_id)
    if not ds:
        return jsonify({'error': 'dataset not found'}), 404
    if ds.get("status") not in ("vectors_ready", "trained", "ready"):
        return jsonify({'error': 'dataset must be vectors_ready before training'}), 400

    params = request.get_json(silent=True) or {}

    def worker(dataset_id, job_id, params):
        dsdb_local = _require_ds_db()
        if dsdb_local is None:
            return
        def fn():
            train_dataset_job(dataset_id, job_id, params, dsdb_local)
        _run_job_safely(dsdb_local, dataset_id, job_id, fn, start_status="computing", start_msg="Training…")

    job_id = _start_job_thread(dsdb, dataset_id, worker, params=params, job_message="Queued (train)…", job_stage="queued")
    return jsonify({'jobId': job_id}), 200

@app.route('/api/datasets/<dataset_id>/pca', methods=['POST'])
def start_pca(dataset_id):
    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500

    ds = dsdb.get_dataset(dataset_id)
    if not ds:
        return jsonify({'error': 'dataset not found'}), 404
    if ds.get("status") not in ("vectors_ready", "trained", "ready"):
        return jsonify({'error': 'dataset must be vectors_ready (or trained) before PCA'}), 400

    params = request.get_json(silent=True) or {}

    def worker(dataset_id, job_id, params):
        dsdb_local = _require_ds_db()
        if dsdb_local is None:
            return
        def fn():
            run_pca_job(dataset_id, job_id, params, dsdb_local)
        _run_job_safely(dsdb_local, dataset_id, job_id, fn, start_status="computing", start_msg="Running PCA…")

    job_id = _start_job_thread(dsdb, dataset_id, worker, params=params, job_message="Queued (pca)…", job_stage="queued")
    return jsonify({'jobId': job_id}), 200

@app.route('/api/datasets/<dataset_id>/pipeline', methods=['POST'])
def start_pipeline(dataset_id):
    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500

    params = request.get_json(silent=True) or {}
    vec_params = params.get("vectorize", {})
    train_params = params.get("train", {})
    pca_params = params.get("pca", {})

    def worker(dataset_id, job_id, params):
        dsdb_local = _require_ds_db()
        if dsdb_local is None:
            return

        def fn():
            make_dataset_job(dataset_id, job_id, vec_params, dsdb_local)
            train_dataset_job(dataset_id, job_id, train_params, dsdb_local)
            run_pca_job(dataset_id, job_id, pca_params, dsdb_local)

        _run_job_safely(dsdb_local, dataset_id, job_id, fn, start_status="computing", start_msg="Running pipeline…")

    job_id = _start_job_thread(dsdb, dataset_id, worker, params=params, job_message="Queued (pipeline)…", job_stage="queued")
    return jsonify({'jobId': job_id}), 200

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500
    job = dsdb.get_job(job_id)
    if not job:
        return jsonify({'error':'not found'}), 404
    return jsonify(job), 200

@app.route('/api/datasets/<dataset_id>/jobs', methods=['GET'])
def list_dataset_jobs(dataset_id):
    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500
    limit = int(request.args.get("limit", "20"))
    return jsonify(dsdb.list_jobs_for_dataset(dataset_id, limit=limit)), 200

@app.route('/api/datasets/<dataset_id>/jobs/latest', methods=['GET'])
def latest_dataset_job(dataset_id):
    dsdb = _require_ds_db()
    if dsdb is None:
        return jsonify({'error':'dataset db not initialized'}), 500
    job = dsdb.get_latest_job_for_dataset(dataset_id)
    return jsonify(job or {}), 200


def _exec_and_commit(query: str):
    cursor, conn = db.execute(query)
    db.safe_commit(conn, cursor)

@app.route("/api/datasets/<dataset_id>", methods=["DELETE"])
def delete_dataset(dataset_id):
    try:
        # Delete registry rows
        _exec_and_commit(f"DELETE FROM datasets WHERE id='{dataset_id}'")
        _exec_and_commit(f"DELETE FROM dataset_jobs WHERE dataset_id='{dataset_id}'")

        # Drop per-dataset tables
        for suffix in ["_meta", "_group", "_vector"]:
            table = f"{dataset_id}{suffix}"
            _exec_and_commit(f"DROP TABLE IF EXISTS {table}")

        # Remove dataset files/folder
        dataset_dir = os.path.join("data", dataset_id)
        if os.path.isdir(dataset_dir):
            shutil.rmtree(dataset_dir)

        return jsonify({"status": "success", "dataset_id": dataset_id}), 200

    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "dataset_id": dataset_id}), 500

# Allowed modes for conversion
IMAGE_MODES = {
    'RGB': 3,
    'RGBA': 4,
    'L': 1  # Grayscale
}

def make_dataset_job(dataset_id, job_id, params, dsdb):
    """
    Worker function to process raw images into HDF5 and generate a config file.
    
    Args:
        dataset_id (str): The dataset unique ID.
        job_id (str): The job unique ID.
        params (dict): {
            'width': int,
            'height': int,
            'train_pct': int (1-99),
            'latent_dims': str ("4, 8, 16"),
            'dataset_name': str,
            'img_mode': str ('RGB', 'RGBA', 'L')
        }
        dsdb (DatasetsDB): Database instance for status updates.
    """
    
    # 1. Setup & Input Parsing
    # ---------------------------------------------------------
    try:
        target_w = int(params.get('width', 64))
        target_h = int(params.get('height', 64))
        pct = int(params.get('train_pct', 80))
        dset_name = params.get('dataset_name', 'dataset')
        img_mode = params.get('img_mode', 'RGB')
        latent_str = params.get('latent_dims', '')
        
        # Validate Latents
        latent_dims = [int(x.strip()) for x in latent_str.split(',') if x.strip()]
        if not latent_dims:
            raise ValueError("Latent dimensions cannot be empty.")
            
        # Validate Percentage
        if not (1 <= pct <= 99):
            raise ValueError(f"Training percentage must be between 1 and 99 (got {pct}).")
        
        # Validate Mode
        if img_mode not in IMAGE_MODES:
            raise ValueError(f"Invalid image mode {img_mode}. Supported: {list(IMAGE_MODES.keys())}")
        target_chns = IMAGE_MODES[img_mode]

    except Exception as e:
        dsdb.update_job(job_id, status='error', message=str(e), done=True)
        return

    # Paths
    root = f'./data/{dataset_id}'
    raw_dir = os.path.join(root, 'raw')
    out_dir = os.path.join(root, 'img_vectors')
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Image Processing Loop
    # ---------------------------------------------------------
    dsdb.update_job(job_id, stage='processing_images', progress=0, message='Scanning images...')
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
    filenames = sorted([
        f for f in os.listdir(raw_dir) 
        if os.path.splitext(f.lower())[1] in valid_exts
    ])
    
    if not filenames:
        dsdb.update_job(job_id, status='error', message='No uploaded images were found. Your ZIP file must contain the images directly, not within a subfolder.', done=True)
        return

    vectors = []
    warnings = []
    total_files = len(filenames)
    
    try:
        for idx, fname in enumerate(filenames):
            fpath = os.path.join(raw_dir, fname)
            
            with Image.open(fpath) as img:
                img = img.convert(img_mode)
                w, h = img.size
                
                # Critical Check: Image smaller than target
                if w < target_w or h < target_h:
                    error_msg = f"Image '{fname}' ({w}x{h}) is smaller than target ({target_w}x{target_h}). Processing aborted."
                    dsdb.update_job(job_id, status='error', message=error_msg, done=True)
                    return
                
                # Center Crop Logic (if not exact match)
                if w != target_w or h != target_h:
                    left = (w - target_w) / 2
                    top = (h - target_h) / 2
                    right = (w + target_w) / 2
                    bottom = (h + target_h) / 2
                    
                    img = img.crop((left, top, right, bottom))
                    # Record a warning (limit to first 5 to avoid spamming DB)
                    if len(warnings) < 5:
                        warnings.append(f"Cropped {fname}")
                
                # Convert to numpy array (uint8)
                arr = np.asarray(img, dtype='uint8')

                # FIX: Expand dimensions for Grayscale so it becomes (H, W, 1)
                # Keras expects 4D inputs (N, H, W, C), so 3D here is mandatory.
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=-1)

                vectors.append(arr)

            # Update Progress every 10 images
            if idx % 10 == 0:
                prog = int((idx / total_files) * 50) # First 50% of progress bar
                dsdb.update_job(job_id, progress=prog, message=f'Processed {idx}/{total_files} images...')

    except Exception as e:
        dsdb.update_job(job_id, status='error', message=f"Error processing {fname}: {str(e)}", done=True)
        return

    # Stack into one large array (N, H, W, C)
    np_vectors = np.array(vectors)

    # 3. HDF5 Creation
    # ---------------------------------------------------------
    dsdb.update_job(job_id, stage='saving_hdf5', progress=60, message='Writing HDF5 file...')
    
    h5_filename = f"{dset_name}.h5"
    h5_path = os.path.join(out_dir, h5_filename)
    
    try:
        with h5py.File(h5_path, 'w') as f:
            # Create dataset. Use compression to save space.
            dset = f.create_dataset(dset_name, data=np_vectors, compression="gzip")
            
            # Store metadata attributes inside HDF5 as well (optional but good practice)
            dset.attrs['dims'] = latent_dims
            dset.attrs['train_pct'] = pct
    except Exception as e:
        dsdb.update_job(job_id, status='error', message=f"HDF5 Write Failed: {str(e)}", done=True)
        return

    # 4. Config Generation
    # ---------------------------------------------------------
    dsdb.update_job(job_id, stage='generating_config', progress=80, message='Generating config...')

    # Calculate Split
    N = len(np_vectors)
    if pct >= 50:
        train_split = math.floor(N * (pct / 100.0))
    else:
        train_split = math.ceil(N * (pct / 100.0))

    config_content = f'''#!/usr/bin/env python
# configurations unique to {dset_name} dataset

dset = '{dset_name}'
data_type = 'image'
img_rows, img_cols, img_chns = {target_h}, {target_w}, {target_chns}
img_mode = '{img_mode}'
train_split = {train_split}
metric = 'l2'

fn_raw = '{h5_filename}'
key_raw = '{dset_name}' # the dataset key in hdf5 file

# dims = {latent_dims} # all latent dims
dims = {latent_dims}

# MySQL table schema
schema_meta = 'i, name, created_at, extra_field' 
schema_header = None
'''

    config_path = os.path.join(out_dir, f'config_{dset_name}.py')
    with open(config_path, 'w') as f:
        f.write(config_content)

    # 5. Finalize
    # ---------------------------------------------------------
    final_msg = "Done."
    if warnings:
        final_msg += f" (Note: {len(warnings)} images cropped, e.g., {warnings[0]})"
        
    dsdb.update_job(job_id, status='done', stage='done', progress=100, message=final_msg, done=True)
    
    # Optionally mark dataset as "processed" in main registry
    dsdb.update_dataset(dataset_id, status='vectors_ready', message='Vectors and Config generated.')

def train_dataset_job(dataset_id, job_id, params, dsdb):
    """
    Worker to run VAE training based on generated vectors/config.
    
    Args:
        dataset_id (str): ID of the dataset.
        job_id (str): ID of the current job.
        params (dict): {
            'epochs': int,
            'latent_dim': int (optional, if None, trains all dims in config)
        }
    """
    epochs = int(params.get('epochs', 100))
    target_dim = params.get('latent_dim') # If specific dim requested
    
    root = f'./data/{dataset_id}'
    vectors_dir = os.path.join(root, 'img_vectors')
    
    # 1. Locate Config File 
    try:
        config_files = [f for f in os.listdir(vectors_dir) if f.startswith('config_') and f.endswith('.py')]
        if not config_files:
            raise FileNotFoundError("No config file found. Did you run the vectorization job?")
        
        config_path = os.path.join(vectors_dir, config_files[0])
        
        # Dynamic Import
        spec = importlib.util.spec_from_file_location("dset_config", config_path)
        dset_config = importlib.util.module_from_spec(spec)
        # FIX: Do not inject into sys.modules to avoid race conditions
        spec.loader.exec_module(dset_config)
        
    except Exception as e:
        dsdb.update_job(job_id, status='error', message=f"Config Error: {e}", done=True)
        return

    # 2. Determine Dimensions to Train
    dims_to_train = []
    if target_dim:
        dims_to_train = [int(target_dim)]
    elif hasattr(dset_config, 'dims'):
        dims_to_train = dset_config.dims
    else:
        dims_to_train = [64]

    dsdb.update_job(job_id, stage='training', progress=5, message=f'Starting training for dims: {dims_to_train}')

    # 3. Loop through dimensions
    total_dims = len(dims_to_train)
    
    try:
        for i, dim in enumerate(dims_to_train):
            msg = f"Training latent_dim={dim} ({i+1}/{total_dims})"
            dsdb.update_job(job_id, message=msg)
            
            # Pass the loaded dset_config object explicitly
            train_vae(
                dataset_id=dataset_id, 
                job_id=job_id, 
                config=dset_config, 
                latent_dim=dim, 
                epochs=epochs, 
                dsdb=dsdb
            )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        dsdb.update_job(job_id, status='error', message=f"Training Failed: {str(e)}", done=True)
        return

    # 4. Finalize
    dsdb.update_dataset(dataset_id, status='trained', message='Model training complete.')
    dsdb.update_job(job_id, status='done', stage='done', progress=100, message='All models trained.', done=True)

def run_pca_job(dataset_id, job_id, params, dsdb):
    """
    Worker to run PCA dimensionality reduction on trained vectors.
    
    Args:
        dataset_id (str): ID of the dataset.
        job_id (str): ID of the current job.
        params (dict): Empty dict (no args required).
    """
    
    # 1. Discover Trained Models
    # We look into ./data/<id>/models/ for any subdirectories that are numbers
    models_root = f'./data/{dataset_id}/models'
    if not os.path.exists(models_root):
        dsdb.update_job(job_id, status='error', message='No models directory found. Train a model first.', done=True)
        return

    # Find directories like "64", "128", "32"
    trained_dims = [
        int(d) for d in os.listdir(models_root) 
        if os.path.isdir(os.path.join(models_root, d)) and d.isdigit()
    ]
    
    if not trained_dims:
        dsdb.update_job(job_id, status='error', message='No trained latent dimensions found.', done=True)
        return

    trained_dims.sort()
    total = len(trained_dims)
    dsdb.update_job(job_id, stage='pca', progress=0, message=f'Found {total} dimensions to process: {trained_dims}')

    # 2. Process Each Dimension
    try:
        for index, dim in enumerate(trained_dims):
            msg = f"Running PCA on latent dim {dim} ({index + 1}/{total})"
            
            # Calculate progress slice
            prog_start = int((index / total) * 100)
            dsdb.update_job(job_id, progress=prog_start, message=msg)

            # CALL THE PCA FUNCTION
            run_pca(dataset_id, dim)

    except Exception as e:
        import traceback
        traceback.print_exc()
        dsdb.update_job(job_id, status='error', message=f"PCA Failed on dim {dim}: {str(e)}", done=True)
        return

    # 3. Finalize
    dsdb.update_dataset(dataset_id, status='ready', message='PCA complete. Dataset ready for visualization.')
    dsdb.update_job(job_id, status='done', stage='done', progress=100, message='PCA calculation done.', done=True)

def run_tsne_job(dataset_id, job_id, params, dsdb):
    """
    Worker to calculate t-SNE for various perplexities on trained models.
    """
    
    # 1. Discover Trained Models
    models_root = f'./data/{dataset_id}/models'
    if not os.path.exists(models_root):
        dsdb.update_job(job_id, status='error', message='No models directory found.', done=True)
        return

    # Find directories (latent dims)
    trained_dims = [
        int(d) for d in os.listdir(models_root) 
        if os.path.isdir(os.path.join(models_root, d)) and d.isdigit()
    ]
    
    if not trained_dims:
        dsdb.update_job(job_id, status='error', message='No trained latent dimensions found.', done=True)
        return

    trained_dims.sort()
    
    # Define perplexities to run (Standard set from your original file)
    perplexities = [5, 10, 30, 50, 100]
    
    total_steps = len(trained_dims) * len(perplexities)
    current_step = 0

    dsdb.update_job(job_id, stage='tsne', progress=0, message=f'Starting t-SNE for dims: {trained_dims}')

    # 2. Process Loop
    try:
        for dim in trained_dims:
            for perp in perplexities:
                current_step += 1
                progress = int((current_step / total_steps) * 100)
                
                msg = f"Running t-SNE (Dim: {dim}, Perp: {perp})"
                dsdb.update_job(job_id, progress=progress, message=msg)
                
                # CALL THE TSNE FUNCTION
                run_tsne(dataset_id, dim, perp)
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        dsdb.update_job(job_id, status='error', message=f"t-SNE Failed: {str(e)}", done=True)
        return

    # 3. Finalize
    dsdb.update_job(job_id, status='done', stage='done', progress=100, message='t-SNE calculations complete.', done=True)

if __name__ == '__main__':
    init_server()
    print('\033[92m' + 'Server started!')
    print('Navigate to http://127.0.0.1:5000/ in your browser')
    print('Press CTRL+C to stop' + '\033[0m')
    app.run(debug=True)  # change to (host= '0.0.0.0') in production
    #app.run(host= '0.0.0.0')

