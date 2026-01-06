<template>
  <span class="ds-badge" :data-status="status">
    Status: {{ label }}
  </span>
</template>

<script>
import { DatasetStatus } from "@/datasets/datasetTypes";

export default {
  name: "DatasetStatusBadge",
  props: {
    status: { type: String, required: true },
  },
  computed: {
    label() {
      switch (this.status) {
        case DatasetStatus.EMPTY: return "Empty";
        case DatasetStatus.UPLOADING_RAW: return "Uploading (ZIP)";
        case DatasetStatus.RAW_UPLOADED: return "Images uploaded";
        case DatasetStatus.UPLOADING_CSV: return "Uploading (CSV)";
        case DatasetStatus.CSV_UPLOADED: return "Metadata uploaded";
        case DatasetStatus.COMPUTING: return "Computing";
        case DatasetStatus.VECTORS_READY: return "Vectors ready";
        case DatasetStatus.TRAINED: return "Trained";
        case DatasetStatus.PCA_COMPLETED: return "PCA completed";
        case DatasetStatus.READY: return "Ready";
        case DatasetStatus.ERROR: return "Error";
        default: return this.status;
      }
    },
  },
};
</script>

<style scoped>
.ds-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
  line-height: 18px;
  border: 1px solid rgba(0,0,0,0.15);
}
.ds-badge[data-status="ready"] { border-color: rgba(0,0,0,0.35); }
.ds-badge[data-status="error"] { border-color: rgba(255,0,0,0.5); }
</style>
