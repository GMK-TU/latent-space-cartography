/**
 * Subscribe to server-side job progress.
 *
 * Preferred: Server-Sent Events (SSE) for live updates.
 * Fallback: polling.
 *
 * Call signature:
 *   const stop = subscribeJobProgress({
 *     jobId,
 *     onEvent: (evt) => {},
 *     onError: (err) => {},
 *     pollingMs: 1500,
 *   });
 *   stop(); // unsubscribe
 */

import { getJobProgress, jobEventsUrl } from "./datasetApi";

/**
 * Subscribe to job progress:
 * - tries SSE first
 * - falls back to polling
 */
export function subscribeJobProgress({ jobId, onEvent, onError, pollingMs = 1500 }) {
  let stopped = false;
  let es = null;
  let timer = null;

  function stop() {
    stopped = true;
    if (timer) clearInterval(timer);
    timer = null;
    if (es) es.close();
    es = null;
  }

  function emit(rawEvt) {
    if (stopped) return;
    try {
      onEvent && onEvent(normalizeProgressEvent(rawEvt));
    } catch (e) {
      onError && onError(e);
    }
  }

  async function pollOnce() {
    try {
      const raw = await getJobProgress(jobId);
      emit(raw);
      if (raw.done === true || raw.status === "done") stop();
    } catch (e) {
      onError && onError(e);
      // keep polling; transient failures happen
    }
  }

  function startPolling() {
    pollOnce();
    timer = setInterval(pollOnce, pollingMs);
  }

  // Try SSE first (backend may not implement it; fallback on error)
  try {
    es = new EventSource(jobEventsUrl(jobId), { withCredentials: true });
    es.onmessage = (msg) => {
      try {
        const raw = JSON.parse(msg.data);
        emit(raw);
        if (raw.done === true || raw.status === "done") stop();
      } catch (e) {
        onError && onError(e);
      }
    };
    es.onerror = () => {
      // SSE not available / server closed; fallback
      if (!stopped) {
        try { es.close(); } catch (_) {}
        es = null;
        startPolling();
      }
    };
  } catch (e) {
    startPolling();
  }

  return stop;
}

export function normalizeProgressEvent(evt) {
  const overallProgress =
    typeof evt.progress === "number"
      ? evt.progress
      : typeof evt.overallProgress === "number"
      ? evt.overallProgress
      : 0;

  return {
    overallProgress: clamp(overallProgress, 0, 100),
    message: evt.message || "",
    stage: evt.stage || "",
    stageProgress:
      typeof evt.stageProgress === "number" ? clamp(evt.stageProgress, 0, 1) : undefined,
    done: evt.done === true || overallProgress >= 100 || evt.status === "done",
    raw: evt,
  };
}

function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}
