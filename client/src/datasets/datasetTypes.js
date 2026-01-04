// Dataset domain types / enums (JS-friendly)
export const DatasetStatus = Object.freeze({
  EMPTY: "empty",
  UPLOADING_RAW: "uploading_raw",
  RAW_UPLOADED: "raw_uploaded",
  UPLOADING_CSV: "uploading_csv",
  CSV_UPLOADED: "csv_uploaded",
  COMPUTING: "computing",
  VECTORS_READY: "vectors_ready",
  TRAINED: "trained",
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
