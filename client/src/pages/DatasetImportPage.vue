<template>
  <div class="import">
    <h1 class="import__title">Import dataset</h1>

    <div class="import__top">
      <div class="import__datasetRow">
        <label class="import__label">Dataset</label>
        <select class="import__select" :value="activeId || ''" @change="onDatasetChange">
          <option disabled value="">Select…</option>
          <option v-for="d in datasets" :key="d.id" :value="d.id">{{ d.name }}</option>
        </select>

        <button class="import__btn" type="button" @click="openCreateDialog">Import new dataset</button>

        <DatasetStatusBadge v-if="active" :status="active.status" />

        <button class="import__btnDanger" v-if="active" type="button" @click="deleteDataset">Delete dataset</button>

        <button class="import__btnPrimary" v-if="active" type="button" :disabled="!canGoToAnalogy" @click="goToAnalogy">
          Open in Analogy
        </button>
      </div>

      <DatasetProgress v-if="active" :progress="active.progress" :message="active.message" />

        <!-- Create dataset dialog -->
        <div
          v-if="createDialogOpen"
          class="import__modalBackdrop"
          @click.self="createDialogOpen = false"
        >
          <div class="import__modal">
            <h2 class="import__modalTitle">Create dataset</h2>

            <div class="import__grid">
              <label class="import__label">
                Name
                <input class="import__input" type="text" v-model="newDataset.name" />
              </label>

              <label class="import__label">
                Type
                <select class="import__select" v-model="newDataset.type">
                  <option value="image">Images</option>
                  <option value="latent">Latent space</option>
                </select>
              </label>
            </div>

            <div class="import__modalActions">
              <button class="import__btn" type="button" @click="createDialogOpen = false">Cancel</button>
              <button class="import__btnPrimary" type="button" @click="createFromDialog">Create</button>
            </div>
          </div>
        </div>

      <div class="import__error" v-if="active && active.error">{{ active.error }}</div>
      <div class="import__error" v-else-if="active && store.state.lastError">{{ store.state.lastError }}</div>
    </div>

    <div v-if="!active" class="import__empty">
      Select a dataset (or create one) to begin.
    </div>

    <div v-else class="import__body">
    <div class="import__steps">
      <button type="button" class="import__step" :data-active="step === 1" @click="step = 1">
        1. {{ isLatent ? "Latent space" : "Images (ZIP)" }}
      </button>

      <button v-if="!isLatent" type="button" class="import__step" :disabled="inferredStep < 1"
              :data-active="step === 2" @click="step = 2">
        2. Metadata (CSV)
      </button>

      <button type="button" class="import__step"
        :disabled="isLatent ? !active || active.status !== DatasetStatus.LATENT_UPLOADED : inferredStep < 2"
        :data-active="step === 3"
        @click="step = 3">
          {{ isLatent ? "2. Compute" : "3. Compute" }}
        </button>

    </div>

      <div class="import__panel" v-if="step === 1">
          <h2 v-if="!isLatent">Step 1: Upload raw dataset (images in a ZIP)</h2>
          <h2 v-else>Step 1: Upload latent space (text vectors)</h2>

          <!-- Image import -->
          <div v-if="!isLatent">
            <input type="file" accept=".zip" @change="onZipSelected" />
            <div class="import__hint" v-if="!active.previewImages || !active.previewImages.length">
              After upload, we show the first few images.
            </div>
            <DatasetPreview :images="active.previewImages" />
          </div>

          <!-- Latent import -->
          <div v-else>
            <div class="import__grid">
              <label class="import__label">
                Latent dim
                <input class="import__input" type="number" min="1" v-model.number="latent.latent_dim" />
              </label>

              <label class="import__label">
                Sample %
                <input class="import__input" type="number" min="1" max="100" v-model.number="latent.sample_percentage" />
              </label>
            </div>

            <input type="file" accept=".txt" @change="onLatentSelected" />
            <div class="import__hint">
              Expected format: one row per item: <code>token v1 v2 ...</code>
            </div>
          </div>
        </div>


      <div class="import__panel" v-else-if="step === 2 && !isLatent">
        <h2>Step 2: Upload metadata (e.g., a CSV file)</h2>
        <input type="file" accept=".csv,text/csv" @change="onCsvSelected" />
        <div class="import__hint">After upload, we show the first few metadata rows and a matched preview if available.</div>
        <DatasetPreview :images="active.previewImages" :meta="active.previewMeta" :matched="active.matchedPreview" />
      </div>

        <div class="import__panel" v-else>
          <h2>Step 3: Run pipeline</h2>

          <div v-if="isLatent" class="import__hint">
            This dataset is a precomputed latent space. Only PCA and t-SNE are available.
          </div>

          <div class="import__hint">
            Configure vectorization + training parameters and run steps. Progress updates live.
          </div>

          <div v-if="!isLatent">
          <h3>Vectorization</h3>
          <div class="import__grid">
            <label>Width <input type="number" v-model.number="vec.width" /></label>
            <label>Height <input type="number" v-model.number="vec.height" /></label>
            <label>Train % <input type="number" v-model.number="vec.train_pct" min="1" max="99" /></label>
            <label>Latent dims <input type="text" v-model="vec.latent_dims" placeholder="8,16" /></label>
            <label>Image mode
              <select v-model="vec.img_mode">
                <option>RGB</option>
                <option>RGBA</option>
                <option>L</option>
              </select>
            </label>
          </div>

          <button class="import__btnPrimary" type="button"
            :disabled="vectorizeDisabled"
            @click="startVectorize">
            Run vectorization
          </button>

          <h3>Training</h3>
          <div class="import__grid">
            <label>Epochs <input type="number" v-model.number="train.epochs" min="1" /></label>
          </div>

          <button class="import__btnPrimary" type="button"
            :disabled="trainDisabled"
            @click="startTrain">
            Run training
          </button>

          <h3>All-in-one</h3>
          <button class="import__btnPrimary" type="button"
            :disabled="pipelineDisabled"
            @click="startPipeline">
            Run full pipeline
          </button>

          </div>

          <h3>PCA</h3>
          <button class="import__btnPrimary" type="button" :disabled="pcaDisabled" @click="startPca">
            Run PCA
          </button>

          <h3>t-SNE</h3>
          <div class="import__grid">
            <label>Perplexities
              <input type="text" v-model="tsne.perplexities" placeholder="e.g. 5, 10, 30 (smaller than dataset size)"/>
            </label>
          </div>
          <button class="import__btnPrimary" type="button" :disabled="tsneDisabled" @click="startTsne">
            Run t-SNE
          </button>

          <div class="import__hint" v-if="active.jobStage">
            Stage: <strong>{{ active.jobStage }}</strong>
            <span v-if="typeof active.jobStageProgress === 'number'">
              ({{ Math.round(active.jobStageProgress * 100) }}%)
            </span>
          </div>
        </div>

    </div>
  </div>
</template>

<script>
import { datasetStore } from "@/datasets/datasetStore";
import { DatasetStatus } from "@/datasets/datasetTypes";
import DatasetProgress from "@/components/datasets/DatasetProgress.vue";
import DatasetStatusBadge from "@/components/datasets/DatasetStatusBadge.vue";
import DatasetPreview from "@/components/datasets/DatasetPreview.vue";

export default {
  name: "DatasetImportPage",
  components: { DatasetProgress, DatasetStatusBadge, DatasetPreview },
  data() {
    return {
      DatasetStatus,
      store: datasetStore,
      step: 1,

      // Create dataset dialog
      createDialogOpen: false,
      newDataset: { name: `Dataset ${new Date().toLocaleString()}`, type: "image" },

      latent: { latent_dim: 50, sample_percentage: 100 },

      vec: { width: 64, height: 64, train_pct: 80, latent_dims: "8,16", dataset_name: "", img_mode: "RGB" },
      train: { epochs: 5 },
      tsne: { perplexities: "5, 10" },
    };
  },
  computed: {
    datasets() {
      return this.store.state.datasets || [];
    },
    isLatent() {
      return this.active && this.active.type === "latent";
    },
    activeId() {
      return this.store.state.activeDatasetId;
    },
    active() {
      const id = this.activeId;
      return this.datasets.find((d) => d.id === id) || null;
    },
    computingDisabled() {
      if (!this.active) return true;
      return (
        this.active.status === DatasetStatus.COMPUTING ||
        this.active.status === DatasetStatus.UPLOADING_RAW ||
        this.active.status === DatasetStatus.UPLOADING_CSV
      );
    },
    inferredStep() {
      return this.inferStep();
    },
    vectorizeDisabled() {
      if (!this.active) return true;
      if (this.isLatent) return true;
      if (this.active.status !== DatasetStatus.CSV_UPLOADED) return true;
      return this.computingDisabled;
    },
    trainDisabled() {
      if (this.isLatent) return true;
      if (!this.active) return true;
      if (this.active.status !== DatasetStatus.VECTORS_READY) return true;
      return this.computingDisabled;
    },
    pcaDisabled() {
      if (!this.active) return true;

      if (this.isLatent) {
        // Allow PCA after latent upload (or re-run when READY)
        if (![DatasetStatus.LATENT_UPLOADED, DatasetStatus.READY].includes(this.active.status)) return true;
        return this.computingDisabled;
      }

      if (![DatasetStatus.VECTORS_READY, DatasetStatus.TRAINED].includes(this.active.status)) return true;
      return this.computingDisabled;
    },
    tsneDisabled() {
      if (!this.active) return true;

      console.log("tsne-disabled: " + this.active.status);

      if (this.isLatent) {
        // Simplest: allow once latent is uploaded (tsne can run; it can compute PCA internally or use existing PCA)
        if (![DatasetStatus.LATENT_UPLOADED, DatasetStatus.PCA_COMPLETED, DatasetStatus.READY].includes(this.active.status)) return true;
        return this.computingDisabled;
      }

      if (![DatasetStatus.TRAINED, DatasetStatus.PCA_COMPLETED, DatasetStatus.READY].includes(this.active.status)) return true;
      return this.computingDisabled;
    },
    pipelineDisabled() {
      if (!this.active) return true;
      if (this.isLatent) return true;
      if (this.active.status !== DatasetStatus.CSV_UPLOADED) return true;
      return this.computingDisabled;
    },
    canGoToAnalogy() {
      if (!this.active) return false;
      return this.active.status === DatasetStatus.READY;
    },
  },
  async created() {
    await this.store.actions.fetchDatasets();

    // Heuristic: auto-advance to the first incomplete step
    // this.step = this.inferStep();
  },
  watch: {
    activeId() {
      this.step = this.inferStep();
    },
    "active.status"() {
      this.step = this.inferStep();
    },
  },
  methods: {
    inferStep() {
      let result = 1;

      if (this.active) {
        const s = this.active.status;

        if (this.isLatent) {
          if (s === DatasetStatus.EMPTY || s === DatasetStatus.UPLOADING_LATENT)
            result = 1;
          if (s === DatasetStatus.LATENT_UPLOADED || s === DatasetStatus.COMPUTING ||
              s === DatasetStatus.PCA_COMPLETED || s === DatasetStatus.TSNE_COMPLETED ||
              s === DatasetStatus.READY || s === DatasetStatus.ERROR)
            result = 2;
        } else {
          if (s === DatasetStatus.EMPTY || s === DatasetStatus.UPLOADING_RAW)
            result = 1;
          if (s === DatasetStatus.RAW_UPLOADED || s === DatasetStatus.UPLOADING_CSV)
            result = 2;
          if (s === DatasetStatus.CSV_UPLOADED || s === DatasetStatus.COMPUTING || s === DatasetStatus.VECTORS_READY ||
              s === DatasetStatus.TRAINED || s === DatasetStatus.READY || s === DatasetStatus.PCA_COMPLETED ||
              s === DatasetStatus.READY || s === DatasetStatus.ERROR )
            result = 3;
        }
      }
      console.log("inferStep returns " + result)

      return result;
    },

    async onLatentSelected(e) {
      const file = e.target.files && e.target.files[0];
      if (!file || !this.activeId) return;

      await this.store.actions.uploadLatentText(this.activeId, file, {
        latent_dim: this.latent.latent_dim,
        sample_percentage: this.latent.sample_percentage,
      });

      e.target.value = "";
    },

    onDatasetChange(e) {
      this.store.actions.setActiveDataset(e.target.value);
    },

    openCreateDialog() {
      this.newDataset = { name: `Dataset ${new Date().toLocaleString()}`, type: "image" };
      this.createDialogOpen = true;
    },

    async onLatentSelected(e) {
      const file = e.target.files && e.target.files[0];
      if (!file || !this.activeId) return;
      await this.store.actions.uploadLatentText(this.activeId, file, {
        latent_dim: this.latent.latent_dim,
        sample_percentage: this.latent.sample_percentage,
      });
      e.target.value = "";
    },

    async createFromDialog() {
      const name = (this.newDataset.name || "").trim();
      if (!name) return;

      await this.store.actions.createDataset({ name, type: this.newDataset.type });

      this.createDialogOpen = false;
      this.step = 1;
    },

    async onZipSelected(e) {
      const file = e.target.files && e.target.files[0];
      if (!file || !this.activeId) return;
      const res = await this.store.actions.uploadRawZip(this.activeId, file);

      console.log("uploadRawZip response:", res);
      console.log("active.previewImages after:", this.active.previewImages);

      e.target.value = "";
    },

    async onCsvSelected(e) {
      const file = e.target.files && e.target.files[0];
      if (!file || !this.activeId) return;
      await this.store.actions.uploadCsv(this.activeId, file);
      e.target.value = "";
    },

    async deleteDataset() {
        if (!this.active) return;

        const name = this.active.name || this.active.id;
        const ok = window.confirm(
          `Delete dataset "${name}"?\n\nThis will permanently remove all data and cannot be undone.`
        );
        if (!ok) return;

        try {
          await fetch(`/api/datasets/${this.active.id}`, {
            method: "DELETE"
          });

          // refresh list
          await this.store.actions.fetchDatasets();

          // reset selection
          this.store.actions.setActiveDataset(null);

          // optional: go back to a safe page
          if (this.$router) {
            this.$router.push({ name: "dataset-import" }).catch(() => {});
          }

        } catch (err) {
          alert("Failed to delete dataset: " + err);
        }
      },
        async startVectorize() {
          await this.store.actions.startVectorize(this.activeId, { ...this.vec, dataset_name: this.vec.dataset_name || this.active.name });
        },
        async startTrain() {
          await this.store.actions.startTrain(this.activeId, { ...this.train });
        },
        async startPca() {
          await this.store.actions.startPca(this.activeId, {});
        },
        async startTsne() {
          await this.store.actions.startTsne(this.activeId, { perplexities: this.tsne.perplexities });
        },
        async startPipeline() {
          await this.store.actions.startPipeline(this.activeId, {
            vectorize: { ...this.vec, dataset_name: this.vec.dataset_name || this.active.name },
            train: { ...this.train },
            pca: {},
            tsne: { perplexities: this.tsne.perplexities }
          });
        },
        goToAnalogy() {
            if (!this.activeId) return;

            // Ensure the imported dataset becomes the active dataset globally
            this.store.actions.setActiveDataset(this.activeId);

            // Navigate to Analogy page
            if (this.$router) {
              this.$router.push({ name: "analogy" }).catch(() => {});
            }
          },

  },
};
</script>

<style scoped>
.import {
  height: 100vh;
  display: flex;
  flex-direction: column;
}
.import { padding: 18px; max-width: 980px; margin: 0 auto; }
.import__title { margin: 0 0 14px; font-size: 22px; }
.import__top { display: grid; gap: 10px; padding: 12px; border: 1px solid rgba(0,0,0,0.12); border-radius: 12px; flex: 0 0 auto; }
.import__datasetRow { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.import__label { margin-bottom: 0; font-weight: bold; }
.import__select { min-width: 140px; padding: 6px 10px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.2); }
.import__btn { padding: 6px 10px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.2); background: transparent; cursor: pointer; }
.import__btn:hover { background: rgba(0,0,0,0.04); }
.import__btnPrimary { padding: 8px 12px; border-radius: 10px; border: 1px solid rgba(0,0,0,0.2); background: rgba(0,0,0,0.06); cursor: pointer; }
.import__btnPrimary:disabled { opacity: 0.5; cursor: not-allowed; }
.import__btnDanger { padding: 6px 10px; border-radius: 8px; border: 1px solid #b00020; background: transparent; color: #b00020; cursor: pointer; }
.import__btnDanger:hover { background: rgba(176, 0, 32, 0.08); }
.import__error { color: #b00020; font-size: 13px; }
.import__empty { padding: 16px; opacity: 0.75; }
.import__body {
    margin-top: 14px; display: grid; gap: 12px;
    flex: 1 1 auto;
    display: flex; flex-direction: column; overflow: hidden; min-height: 0; }
.import__steps { display: flex; gap: 8px; flex-wrap: wrap; flex: 0 0 auto;}
.import__step { padding: 6px 10px; border-radius: 999px; border: 1px solid rgba(0,0,0,0.18); background: transparent; cursor: pointer; font-size: 13px; }
.import__step[data-active="true"] { background: rgba(0,0,0,0.06); border-color: rgba(0,0,0,0.28); }
.import__panel {
    padding: 14px; border: 1px solid rgba(0,0,0,0.12); border-radius: 12px;
    flex: 1 1 auto; overflow-y: auto; min-height: 0;
    }
.import__hint { margin-top: 8px; font-size: 13px; opacity: 0.85; }
.import__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin: 10px 0 12px;
}
.import__grid label { display: grid; gap: 4px; font-size: 13px; }
.import__grid input, .import__grid select {
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid rgba(0,0,0,0.2);
}

.import__modalBackdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.35);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 50;
}

.import__modal {
  background: #fff;
  border-radius: 12px;
  max-width: 520px;
  width: 92%;
  padding: 16px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

.import__modalTitle {
  margin: 0 0 12px;
  font-size: 18px;
}

.import__modalActions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 12px;
}


</style>
