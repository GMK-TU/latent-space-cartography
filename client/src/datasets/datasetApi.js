/**
 * Backend API wrapper for dataset operations.
 *
 * Replace BASE_URL and endpoints to match your backend.
 * All functions return plain JS objects that the datasetStore can merge into state.
 */

const BASE_URL = ""; // e.g. "http://localhost:8000" (leave "" to use same-origin)

function url(path) {
  return `${BASE_URL}${path}`;
}

async function jsonOrThrow(res) {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Request failed (${res.status})`);
  }
  return res.json();
}

export async function loadConfig(datasetId) {
  const res = await fetch(url(`/api/load_config`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ dataset_id: datasetId }),
  });
  const json = await jsonOrThrow(res);
  return json.config; // { dims, capabilities, initial_dim, initial_projection, ... }
}


/**
 * List datasets visible to the user.
 * Expected response example:
 * [{ id, name, createdAt, status, progress, message }]
 */
export async function listDatasets() {
  const res = await fetch(url("/api/datasets"), { credentials: "include" });
  return jsonOrThrow(res);
}

/**
 * Create a new dataset record (optional convenience).
 * Expected response: { id, name, createdAt, status, progress }
 */
export async function createDataset({ name }) {
  const res = await fetch(url("/api/datasets"), {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return jsonOrThrow(res);
}

/**
 * Upload ZIP of images for a dataset.
 * - onProgress: (fraction 0..1) => void
 * Expected response example:
 * { previewImages: [url1, url2, ...], imageCount }
 */
export async function uploadRawZip(datasetId, file, onProgress) {
  const form = new FormData();
  form.append("file", file);

  // Prefer XHR for upload progress (fetch doesn't reliably provide it)
  const endpoint = url(`/api/datasets/${encodeURIComponent(datasetId)}/raw-zip`);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", endpoint, true);
    xhr.withCredentials = true;

    xhr.upload.onprogress = (evt) => {
      if (!onProgress) return;
      if (evt.lengthComputable) onProgress(evt.loaded / evt.total);
    };

    xhr.onload = () => {
      try {
        const ok = xhr.status >= 200 && xhr.status < 300;
        if (!ok) return reject(new Error(xhr.responseText || `Upload failed (${xhr.status})`));
        resolve(JSON.parse(xhr.responseText || "{}"));
      } catch (e) {
        reject(e);
      }
    };

    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.send(form);
  });
}

/**
 * Upload CSV metadata for a dataset.
 * Expected response example:
 * { previewMeta: [...], matchedPreview: [{ imageUrl, metaRow }, ...] }
 */
export async function uploadCsv(datasetId, file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(url(`/api/datasets/${encodeURIComponent(datasetId)}/metadata-csv`), {
    method: "POST",
    credentials: "include",
    body: form,
  });
  return jsonOrThrow(res);
}

/**
 * Start server-side computations (latent space, PCA, etc.)
 * Expected response: { jobId }
 */
export async function startComputations(datasetId) {
  const res = await fetch(url(`/api/datasets/${encodeURIComponent(datasetId)}/compute`), {
    method: "POST",
    credentials: "include",
  });
  return jsonOrThrow(res);
}

export async function startVectorize(datasetId, params) {
  const res = await fetch(url(`/api/datasets/${encodeURIComponent(datasetId)}/vectorize`), {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params || {}),
  });
  return jsonOrThrow(res);
}

export async function startTrain(datasetId, params) {
  const res = await fetch(url(`/api/datasets/${encodeURIComponent(datasetId)}/train`), {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params || {}),
  });
  return jsonOrThrow(res);
}

export async function startPca(datasetId, params) {
  const res = await fetch(url(`/api/datasets/${encodeURIComponent(datasetId)}/pca`), {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params || {}),
  });
  return jsonOrThrow(res);
}

export async function startTsne(datasetId, tsneParams = {}) {
  const res = await fetch(url(`/api/datasets/${encodeURIComponent(datasetId)}/tsne`), {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(tsneParams),
  });
  return jsonOrThrow(res);
}

export async function startPipeline(datasetId, params) {
  const res = await fetch(url(`/api/datasets/${encodeURIComponent(datasetId)}/pipeline`), {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params || {}),
  });
  return jsonOrThrow(res);
}

export async function getLatestDatasetJob(datasetId) {
  const res = await fetch(url(`/api/datasets/${encodeURIComponent(datasetId)}/jobs/latest`), {
    credentials: "include",
  });
  return jsonOrThrow(res);
}

/**
 * Fetch dataset detail (status/progress/message/etc).
 * Expected response: Dataset object
 */
export async function getDataset(datasetId) {
  const res = await fetch(url(`/api/datasets/${encodeURIComponent(datasetId)}`), {
    credentials: "include",
  });
  return jsonOrThrow(res);
}

/**
 * Poll job progress endpoint (fallback if SSE isn't available).
 * Expected response example:
 * { overallProgress: 0..100, message, stage, stageProgress: 0..1, done: boolean }
 */
export async function getJobProgress(jobId) {
  const res = await fetch(url(`/api/jobs/${encodeURIComponent(jobId)}`), {
    credentials: "include",
  });
  return jsonOrThrow(res);
}

/**
 * SSE endpoint for job progress events.
 * You can change this to whatever your backend serves.
 */
export function jobEventsUrl(jobId) {
  return url(`/api/jobs/${encodeURIComponent(jobId)}/events`);
}
