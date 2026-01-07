# Latent Space Cartography (Fork)

## 1. Project Overview

### Original Repository (uwdata)
The original Latent Space Cartography was a research tool designed to create and explore visual projections of vector space embeddings. While it contained core algorithms for projecting high-dimensional data, the workflow was static. Users had to manually prepare data and execute individual scripts with hardcoded parameters to generate visualizations. It primarily focused on image data, with no native code included for processing text embeddings.

### Fork Objectives
This fork re-engineers the project into a dynamic, full-stack web application.

* **Dynamic & Live Workflow (Primary Goal):** The core objective was to eliminate the manual, script-based pipeline. In this fork, users interact with a live GUI to ingest data, train models, and generate projections in real-time without touching code or restarting the server.
* **Modernization:** The codebase has been ported from Python 2.7 / TensorFlow 1.x to Python 3.12+ and TensorFlow 2.x / Keras 3.
* **Architecture Refactor:** Replaced external MySQL dependencies with self-contained SQLite and implemented an asynchronous Job/Queue system to manage heavy computational tasks.

---

## 2. Key Differences & Features

### Text Processing Evolution
* **Original Upstream:** Contained no logic for ingesting or processing text embeddings.
* **Fork:** Implemented a generic text pipeline (import_text_job). The system supports importing any standard text embedding file (e.g., .txt with word vectors) via the UI, enabling users to analyze custom NLP models dynamically.

### Dynamic Projection Jobs (t-SNE & PCA)
* **Static vs. Dynamic:** While the original repository contained code for t-SNE and PCA, the logic was locked inside static scripts with hardcoded parameters.
* **Job Implementation:** This fork refactors those algorithms into parameterized Jobs (run_pca_job, run_tsne_job).
* **User Control:** Parameters that were previously static (e.g., t-SNE perplexity, iterations, or PCA dimensions) are now exposed in the UI, allowing users to tune projections experimentally for every dataset.

### Additional Features
* **Emoji Crawler:** A dedicated testing tool (deploy/emoji/crawler.py) that scrapes and generates verifiable image datasets to validate the pipeline.
* **UI Overhaul:**
    * **Dataset Picker:** A dashboard for managing multiple datasets.
    * **Live Progress:** Real-time feedback bars for vectorization, training, and projection jobs.
    * **Status Badges:** Visual indicators of data processing states.

---

## 3. Technical Stack

| Component | Upstream (Original) | GMK-TU Fork |
| :--- | :--- | :--- |
| **Language** | Python 2.7 | **Python 3.12+** |
| **Backend Framework** | Flask 0.x | **Flask 3.x** |
| **ML Engine** | TensorFlow 1.x / Keras 2 | **TensorFlow 2.16+ / Keras 3** |
| **Database** | MySQL | **SQLite3** |
| **Frontend** | Vue.js / Legacy Webpack | **Vue.js 2.6 / Webpack 5** |
| **Numerical Libs** | Numpy (Legacy) | **NumPy, Pandas, Scikit-Learn 1.4+** |

---

## 4. Installation & Setup Guide

### Prerequisites
* **Python:** 3.10 or higher
* **Node.js:** v18 or higher (LTS recommended)

### Step 1: Backend Setup
1.  Clone the repository and navigate to the root directory.
2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Client Build & Execution
The application logic, including the server entry point (server.py), is located within the client folder in this fork.

1.  Navigate to the client directory:
    ```bash
    cd client
    ```
2.  Install Node dependencies:
    ```bash
    npm install
    ```

**Option A: Production / Standard Use**
To build the static assets and run the standard server:

1.  Build the Vue.js frontend:
    ```bash
    npm run build
    ```
    *This compiles the Vue assets into the client/build/ directory.*
2.  Execute the Python server:
    ```bash
    python server.py
    ```

**Option B: Development**
To run the frontend with hot-reloading enabled during development:

1.  Execute the development server:
    ```bash
    npm run dev
    ```
2.  Ensure the Python backend is running separately (via python server.py inside the client folder) to handle API requests.

**Access:**
Open your browser and navigate to http://localhost:5000 (or the port specified by the dev runner).

---

## 5. User Workflows

### Workflow A: Image Latent Space (VAE Pipeline)

This workflow transforms raw images into a navigable latent space using a Variational Autoencoder.

#### 1. Data Ingestion
* **Action:** Upload a ZIP file containing images via the "New Dataset" UI.
* **System Action:** Unzips images to a staging area in ./data/{dataset_id}/.

#### 2. Job: Vectorization
* **Endpoint:** server.make_dataset_job
* **Description:** Resizes images and converts them to HDF5 arrays.
* **Parameters:**
    * width / height: Target resolution (Standard: 64x64).
    * train_pct: Training/Validation split (e.g., 0.8).
    * img_mode: 'RGB' or 'RGBA'.
* **Output:** ./data/{dataset_id}/img_vectors/

#### 3. Job: Training
* **Endpoint:** server.train_dataset_job
* **Description:** Trains the VAE on the vectorized data.
* **Parameters:**
    * epochs: Number of passes (Default: 50+).
    * latent_dim: Size of the vector space (e.g., 1024).
    * filters: Convolutional depth.
    * kernel_size: Convolutional kernel size.
* **Output:** ./data/{dataset_id}/models/ (Saved Keras models).

#### 4. Job: Projection (Dynamic)
* **Endpoints:** server.run_pca_job / server.run_tsne_job
* **Description:** Reduces latent vectors to 2D. Unlike the upstream repo, these are fully parameterized.
* **Parameters (PCA):**
    * pca_dim: Target components.
* **Parameters (t-SNE):**
    * perplexity: Balance of local/global structure.
    * learning_rate: Optimization step size.
    * iterations: Total optimization steps.
* **Output:** ./data/{dataset_id}/projections/

---

### Workflow B: Text Latent Space (NLP)

This workflow visualizes semantic relationships using imported word vectors.

#### 1. Job: Text Import
* **Endpoint:** run_text_import_job
* **Description:** A generic importer that ingests raw text files where each line represents a word vector.
* **Input Format:** Standard .txt (Space-separated values).
    ```text
    king 0.55 0.32 -0.11 ...
    queen 0.57 0.35 -0.09 ...
    ```
* **Parameters:**
    * file: The source .txt file.
    * latent_dim: Dimensionality of vectors (must match file).
    * sample_percentage: (Optional) Fraction of data to import for testing.
* **Output:**
    * SQLite _vector tables (metadata).
    * HDF5 storage (vector data).

#### 2. Visualization
* **Action:** Open the dataset in the UI.
* **Features:** Perform vector arithmetic (Analogies) and inspect nearest neighbors in the 2D plot.

---

## 6. API & Configuration Reference

### Job System API
The frontend triggers asynchronous jobs via these REST endpoints:

* POST /api/datasets/{id}/vectorize: Queues image processing.
* POST /api/datasets/{id}/train: Queues VAE training.
* POST /api/datasets/{id}/projection: Queues PCA or t-SNE.
* GET /api/jobs/{job_id}/events: Streams progress logs and percentage completion via Server-Sent Events (SSE).

### Database Schema (SQLite)
Data is managed in datasets.db (auto-created in the root).

* **datasets Table:** Tracks dataset ID, type ('image'/'text'), and pipeline status.
* **dataset_jobs Table:** Logs execution history, job type (vectorize, train, tsne), and success/failure states.

### Configuration
* **config_default.json:** Stores global defaults.
* **Dataset Overrides:** Specific job parameters (like t-SNE perplexity) are saved per dataset in the SQLite database, ensuring reproducibility.

## 7. Example videos
### Image imports
![](Images1.mp4)
![](Images2.mp4)
### Text imports
![](Text.mp4)