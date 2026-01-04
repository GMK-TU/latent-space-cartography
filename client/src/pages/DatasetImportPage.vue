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

        <button class="import__btn" type="button" @click="createNew">Import new dataset</button>

        <DatasetStatusBadge v-if="active" :status="active.status" />

        <button class="import__btnDanger" v-if="active" type="button" @click="deleteDataset">Delete dataset</button>
      </div>

      <DatasetProgress v-if="active" :progress="active.progress" :message="active.message" />
      <div class="import__error" v-if="active && active.error">{{ active.error }}</div>
      <div class="import__error" v-else-if="active && store.state.lastError">{{ store.state.lastError }}</div>
    </div>

    <div v-if="!active" class="import__empty">
      Select a dataset (or create one) to begin.
    </div>

    <div v-else class="import__body">
      <div class="import__steps">
        <button type="button" class="import__step" :data-active="step === 1" @click="step = 1">1. Images (ZIP)</button>
        <button type="button" class="import__step" :disabled="inferredStep < 1" :data-active="step === 2" @click="step = 2">2. Metadata (CSV)</button>
        <button type="button" class="import__step" :disabled="inferredStep < 2" :data-active="step === 3" @click="step = 3">3. Compute</button>
      </div>

      <div class="import__panel" v-if="step === 1">
        <h2>Step 1: Upload raw dataset (e.g., images in a ZIP)</h2>
        <input type="file" accept=".zip" @change="onZipSelected" />
        <div class="import__hint" v-if="!active.previewImages || !active.previewImages.length">
          After upload, we show the first few images.
        </div>
        <DatasetPreview :images="active.previewImages" />
      </div>

      <div class="import__panel" v-else-if="step === 2">
        <h2>Step 2: Upload metadata (e.g., a CSV file)</h2>
        <input type="file" accept=".csv,text/csv" @change="onCsvSelected" />
        <div class="import__hint">After upload, we show the first few metadata rows and a matched preview if available.</div>
        <DatasetPreview :images="active.previewImages" :meta="active.previewMeta" :matched="active.matchedPreview" />
      </div>

      <div class="import__panel" v-else>
        <h2>Step 3: Compute artefacts</h2>
        <div class="import__hint">
          This triggers server-side computations (latent space, PCA, t-SNE, etc.). Progress is shown in real time.
        </div>

        <button class="import__btnPrimary" type="button" :disabled="computingDisabled" @click="startCompute">
          Start computations
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
      store: datasetStore,
      step: 1,
    };
  },
  computed: {
    datasets() {
      return this.store.state.datasets || [];
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
          if (this.active.status === DatasetStatus.EMPTY || this.active.status === DatasetStatus.UPLOADING_RAW)
            result = 1;
          if (this.active.status === DatasetStatus.RAW_UPLOADED || this.active.status === DatasetStatus.UPLOADING_CSV)
            result = 2;
          if (this.active.status === DatasetStatus.CSV_UPLOADED || this.active.status === DatasetStatus.COMPUTING)
            result = 3;
          if (this.active.status === DatasetStatus.READY)
            result = 3;
      }
      console.log("inferStep returns " + result)

      return result;
    },

    onDatasetChange(e) {
      this.store.actions.setActiveDataset(e.target.value);
    },

    async createNew() {
      const name = window.prompt("Dataset name:", `Dataset ${new Date().toLocaleString()}`);
      if (!name) return;
      await this.store.actions.createDataset(name);
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

    async startCompute() {
      if (!this.activeId) return;
      await this.store.actions.startComputations(this.activeId);
      this.step = 3;
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
      }
  },
};
</script>

<style scoped>
.import { padding: 18px; max-width: 980px; margin: 0 auto; }
.import__title { margin: 0 0 14px; font-size: 22px; }
.import__top { display: grid; gap: 10px; padding: 12px; border: 1px solid rgba(0,0,0,0.12); border-radius: 12px; }
.import__datasetRow { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.import__label { margin-bottom: 0; font-weight: bold; }
.import__select { min-width: 280px; padding: 6px 10px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.2); }
.import__btn { padding: 6px 10px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.2); background: transparent; cursor: pointer; }
.import__btn:hover { background: rgba(0,0,0,0.04); }
.import__btnPrimary { padding: 8px 12px; border-radius: 10px; border: 1px solid rgba(0,0,0,0.2); background: rgba(0,0,0,0.06); cursor: pointer; }
.import__btnPrimary:disabled { opacity: 0.5; cursor: not-allowed; }
.import__btnDanger { padding: 6px 10px; border-radius: 8px; border: 1px solid #b00020; background: transparent; color: #b00020; cursor: pointer; }
.import__btnDanger:hover { background: rgba(176, 0, 32, 0.08); }
.import__error { color: #b00020; font-size: 13px; }
.import__empty { padding: 16px; opacity: 0.75; }
.import__body { margin-top: 14px; display: grid; gap: 12px; }
.import__steps { display: flex; gap: 8px; flex-wrap: wrap; }
.import__step { padding: 6px 10px; border-radius: 999px; border: 1px solid rgba(0,0,0,0.18); background: transparent; cursor: pointer; font-size: 13px; }
.import__step[data-active="true"] { background: rgba(0,0,0,0.06); border-color: rgba(0,0,0,0.28); }
.import__panel { padding: 14px; border: 1px solid rgba(0,0,0,0.12); border-radius: 12px; }
.import__hint { margin-top: 8px; font-size: 13px; opacity: 0.85; }
</style>
