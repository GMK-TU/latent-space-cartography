// Dataset domain types / enums (JS-friendly)
export const DatasetType = Object.freeze({
  IMAGE: "image",
  LATENT: "latent",
});

export const DatasetStatus = Object.freeze({
  EMPTY: "empty",
  UPLOADING_RAW: "uploading_raw",
  RAW_UPLOADED: "raw_uploaded",
  UPLOADING_LATENT: "uploading_latent",
  LATENT_UPLOADED: "latent_uploaded",
  UPLOADING_CSV: "uploading_csv",
  CSV_UPLOADED: "csv_uploaded",
  COMPUTING: "computing",
  VECTORS_READY: "vectors_ready",
  TRAINED: "trained",
  PCA_COMPLETED: "pca_completed",
  READY: "ready",
  ERROR: "error",
});

export function isReady(status) {
  return status === DatasetStatus.READY;
}

export function isBusy(status) {
  return (
    status === DatasetStatus.UPLOADING_RAW ||
    status === DatasetStatus.UPLOADING_CSV ||
    status === DatasetStatus.COMPUTING
  );
}
