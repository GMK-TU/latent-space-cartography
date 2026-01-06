/**
 * Minimal global dataset store without adding new dependencies (Pinia/Vuex).
 *
 * Works in both Vue 2 and Vue 3:
 * - Vue 3: uses reactive()
 * - Vue 2: uses Vue.observable()
 *
 * Usage:
 *   import { datasetStore } from "@/datasets/datasetStore";
 *   datasetStore.actions.fetchDatasets();
 *   datasetStore.state.activeDatasetId ...
 */

import Vue from "vue";

import { DatasetStatus } from "./datasetTypes";
import * as api from "./datasetApi";
import { subscribeJobProgress } from "./jobProgress";

const state = Vue.observable({
  datasets: [],
  activeDatasetId: null,

  configById: {},     // datasetId -> config
  activeConfig: null, // convenience pointer

  // for UI convenience
  lastError: null,
});

function findIndex(id) {
  return state.datasets.findIndex((d) => d.id === id);
}

function patch(id, partial) {
  const i = findIndex(id);
  if (i === -1) return;
  const next = { ...state.datasets[i], ...partial };
  state.datasets.splice(i, 1, next);
}

function setError(err) {
  state.lastError = err ? String(err.message || err) : null;
}

// --- actions ---
const actions = {
  async fetchDatasetConfig(datasetId) {
    try {
      const cfg = await api.loadConfig(datasetId);
      Vue.set(state.configById, datasetId, cfg);
      if (state.activeDatasetId === datasetId) {
        state.activeConfig = cfg;
      }
      return cfg;
    } catch (e) {
      setError(e);
      throw e;
    }
  },
  async fetchDatasets() {
    try {
      const list = await api.listDatasets();
      state.datasets = Array.isArray(list) ? list.map(normalizeDataset) : [];
      if (!state.activeDatasetId && state.datasets.length) {
        state.activeDatasetId = state.datasets[0].id;
      }
    } catch (e) {
      setError(e);
    }
  },

  async createDataset(name) {
    try {
      const ds = await api.createDataset({ name });
      const normalized = normalizeDataset(ds);
      state.datasets = [normalized, ...state.datasets];
      state.activeDatasetId = normalized.id;
      return normalized;
    } catch (e) {
      setError(e);
      throw e;
    }
  },

  setActiveDataset(id) {
    state.activeDatasetId = id;
    state.activeConfig = state.configById[id] || null;
  },

  async refreshDataset(datasetId) {
    try {
      const ds = await api.getDataset(datasetId);
      patch(datasetId, normalizeDataset(ds));
    } catch (e) {
      setError(e);
    }
  },

  async uploadRawZip(datasetId, file, { weightStart = 0, weightEnd = 25 } = {}) {
    patch(datasetId, {
      status: DatasetStatus.UPLOADING_RAW,
      progress: Math.max(0, Number(weightStart) || 0),
      message: "Uploading images...",
      error: null,
    });

    try {
      const res = await api.uploadRawZip(datasetId, file, (fraction) => {
        const p = lerp(weightStart, weightEnd, fraction);
        console.log("patching with progress " + Math.round(p))
        patch(datasetId, { progress: Math.round(p) });
      });

      patch(datasetId, {
        status: DatasetStatus.RAW_UPLOADED,
        message: "Images uploaded.",
        previewImages: res.previewImages || [],
        imageCount: res.imageCount,
      });
      return res;
    } catch (e) {
      patch(datasetId, { status: DatasetStatus.ERROR, message: "Upload failed.", error: String(e) });
      setError(e);
      throw e;
    }
  },

  async uploadCsv(datasetId, file, { weightStart = 25, weightEnd = 35 } = {}) {
    patch(datasetId, {
      status: DatasetStatus.UPLOADING_CSV,
      progress: Math.max(0, Number(weightStart) || 0),
      message: "Uploading metadata...",
      error: null,
    });

    try {
      const res = await api.uploadCsv(datasetId, file);
      patch(datasetId, {
        status: DatasetStatus.CSV_UPLOADED,
        progress: Math.round(weightEnd),
        message: "Metadata uploaded.",
        previewMeta: res.previewMeta || [],
        matchedPreview: res.matchedPreview || [],
      });
      return res;
    } catch (e) {
      patch(datasetId, { status: DatasetStatus.ERROR, message: "CSV upload failed.", error: String(e) });
      setError(e);
      throw e;
    }
  },

  async startComputations(datasetId, { weightStart = 35, weightEnd = 100 } = {}) {
    patch(datasetId, {
      status: DatasetStatus.COMPUTING,
      progress: Math.round(weightStart),
      message: "Starting computations…",
      error: null,
    });

    let stop = null;

    try {
      const { jobId } = await api.startComputations(datasetId);

      stop = subscribeJobProgress({
        jobId,
        onEvent: (evt) => {
          const overall = lerp(weightStart, weightEnd, evt.overallProgress / 100);
          patch(datasetId, {
            status: evt.done ? DatasetStatus.READY : DatasetStatus.COMPUTING,
            progress: Math.round(overall),
            message: evt.message || (evt.done ? "Ready." : "Computing..."),
            jobStage: evt.stage,
            jobStageProgress: evt.stageProgress,
            jobId,
          });
        },
        onError: (err) => {
          patch(datasetId, { status: DatasetStatus.ERROR, message: "Computation error.", error: String(err) });
          setError(err);
        },
      });

      return { jobId, stop };
    } catch (e) {
      if (stop) stop();
      patch(datasetId, { status: DatasetStatus.ERROR, message: "Failed to start computations.", error: String(e) });
      setError(e);
      throw e;
    }
  },
  async _startJobGeneric(datasetId, startFn, { weightStart, weightEnd, finalStatus = DatasetStatus.READY, startMessage = "Starting..." } = {}) {
      patch(datasetId, {
        status: DatasetStatus.COMPUTING,
        progress: Math.round(weightStart),
        message: startMessage,
        error: null,
      });

      let stop = null;

      try {
        const { jobId } = await startFn();

        stop = subscribeJobProgress({
          jobId,
          onEvent: (evt) => {
            const overall = lerp(weightStart, weightEnd, evt.overallProgress / 100);
            patch(datasetId, {
              status: evt.done ? finalStatus : DatasetStatus.COMPUTING,
              progress: Math.round(overall),
              message: evt.message || (evt.done ? "Done." : "Working..."),
              jobStage: evt.stage,
              jobStageProgress: evt.stageProgress,
              jobId,
            });
          },
          onError: (err) => {
            patch(datasetId, { status: DatasetStatus.ERROR, message: "Job error.", error: String(err) });
            setError(err);
          },
        });

        return { jobId, stop };
      } catch (e) {
        if (stop) stop();
        patch(datasetId, { status: DatasetStatus.ERROR, message: "Failed to start job.", error: String(e) });
        setError(e);
        throw e;
      }
    },
    async startVectorize(datasetId, params) {
      return actions._startJobGeneric(
        datasetId,
        () => api.startVectorize(datasetId, params),
        { weightStart: 25, weightEnd: 50, finalStatus: DatasetStatus.VECTORS_READY, startMessage: "Starting vectorization..." }
      );
    },

    async startTrain(datasetId, params) {
      return actions._startJobGeneric(
        datasetId,
        () => api.startTrain(datasetId, params),
        { weightStart: 50, weightEnd: 80, finalStatus: DatasetStatus.TRAINED, startMessage: "Starting training..." }
      );
    },

    async startPca(datasetId, params) {
      return actions._startJobGeneric(
        datasetId,
        () => api.startPca(datasetId, params),
        { weightStart: 80, weightEnd: 90, finalStatus: DatasetStatus.PCA_COMPLETED, startMessage: "Starting PCA..." }
      );
    },

    async startTsne(datasetId, params) {
      return actions._startJobGeneric(
        datasetId,
        () => api.startTsne(datasetId, params),
        { weightStart: 90, weightEnd: 100, finalStatus: DatasetStatus.READY, startMessage: "Starting tSNE..." }
      );
    },

    async startPipeline(datasetId, params) {
      return actions._startJobGeneric(
        datasetId,
        () => api.startPipeline(datasetId, params),
        { weightStart: 25, weightEnd: 100, finalStatus: DatasetStatus.READY, startMessage: "Starting pipeline..." }
      );
    },

};

// --- derived helpers ---
function normalizeDataset(d) {
  if (!d) {
    return {
      id: "",
      name: "Untitled",
      createdAt: new Date().toISOString(),
      status: DatasetStatus.EMPTY,
      progress: 0,
      message: "",
    };
  }

  return {
    id: d.id,
    name: d.name || "Untitled",
    createdAt: d.createdAt || new Date().toISOString(),
    status: d.status || DatasetStatus.EMPTY,
    progress: typeof d.progress === "number" ? d.progress : 0,
    message: d.message || "",
    error: d.error,

    // previews
    previewImages: d.previewImages || [],
    previewMeta: d.previewMeta || [],
    matchedPreview: d.matchedPreview || [],

    // job details (optional)
    jobId: d.jobId,
    jobStage: d.jobStage,
    jobStageProgress: d.jobStageProgress,
    imageCount: d.imageCount,
  };
}

function lerp(a, b, t) {
  const tt = Math.max(0, Math.min(1, t));
  return a + (b - a) * tt;
}

export const datasetStore = { state, actions };
