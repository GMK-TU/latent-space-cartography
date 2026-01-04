<template>
  <div class="ds-picker">
    <div class="ds-picker__row">
      <label class="ds-picker__label">Dataset</label>

      <select class="ds-picker__select" :value="activeId || ''" @change="onChange">
        <option disabled value="">Select…</option>
        <option v-for="d in datasets" :key="d.id" :value="d.id">
          {{ d.name }} — {{ d.status }} ({{ Math.round(d.progress || 0) }}%)
        </option>
      </select>

      <DatasetStatusBadge v-if="active" :status="active.status" />

      <button class="ds-picker__btn" type="button" @click="goImport">
        Import or Manage datasets
      </button>
    </div>

    <DatasetProgress
      v-if="active"
      :progress="active.progress"
      :message="active.message"
      :show="active.status === 'computing' || active.status === 'uploading_raw' || active.status === 'uploading_csv'"
    />
  </div>
</template>

<script>
import { datasetStore } from "@/datasets/datasetStore";
import DatasetProgress from "./DatasetProgress.vue";
import DatasetStatusBadge from "./DatasetStatusBadge.vue";

export default {
  name: "DatasetPicker",
  components: { DatasetProgress, DatasetStatusBadge },
  data() {
    return { store: datasetStore };
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
  },
  created() {
    // Safe to call repeatedly
    this.store.actions.fetchDatasets();
  },
  methods: {
    onChange(e) {
      const id = e.target.value;
      this.store.actions.setActiveDataset(id);
      this.$emit("changed", id);
    },
    goImport() {
      // Works if vue-router is present
      if (this.$router) this.$router.push({ name: "dataset-import" }).catch(() => {});
    },
  },
};
</script>

<style scoped>
.ds-picker { display: grid; gap: 8px; }
.ds-picker__row { display: flex; align-items: end; gap: 10px; flex-wrap: wrap; }
.ds-picker__select { min-width: 240px; padding: 6px 10px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.2); }
.ds-picker__btn {
    color:inherit; padding: 6px 10px;
    border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.2);
    background: transparent; cursor: pointer;
    background-color: color-mix(in srgb, currentColor 12%, transparent);
    }
.ds-picker__btn:hover {
    background-color: color-mix(in srgb, currentColor 20%, transparent);
    border-color: color-mix(in srgb, currentColor 45%, transparent);
    }
</style>
