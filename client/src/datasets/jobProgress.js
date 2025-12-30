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

export function subscribeJobProgress({ jobId, onEvent, onError, pollingMs = 1500 }) {
  let stopped = false;

  // Try SSE first
  if (typeof window !== "undefined" && "EventSource" in window) {
    try {
      const es = new EventSource(jobEventsUrl(jobId), { withCredentials: true });

      es.onmessage = (msg) => {
        if (stopped) return;
        try {
          const data = JSON.parse(msg.data);
          onEvent && onEvent(normalizeEvent(data));
        } catch (e) {
          // Ignore malformed events
        }
      };

      es.onerror = (err) => {
        if (stopped) return;
        // If SSE fails (server/proxy), fall back to polling
        try { es.close(); } catch (_) {}
        poll();
      };

      return () => {
        stopped = true;
        try { es.close(); } catch (_) {}
      };
    } catch (e) {
      // Fall through to polling
    }
  }

  // Fallback: polling
  let timer = null;

  async function poll() {
    if (stopped) return;
    try {
      const data = await getJobProgress(jobId);
      onEvent && onEvent(normalizeEvent(data));
      if (data && (data.done === true || data.overallProgress >= 100)) {
        stop();
        return;
      }
    } catch (e) {
      onError && onError(e);
    }
    timer = setTimeout(poll, pollingMs);
  }

  function stop() {
    stopped = true;
    if (timer) clearTimeout(timer);
    timer = null;
  }

  poll();
  return stop;
}

function normalizeEvent(evt) {
  // Normalize to a consistent shape for the store.
  // Supports either {overallProgress} or {progress}.
  const overallProgress =
    typeof evt.overallProgress === "number"
      ? evt.overallProgress
      : typeof evt.progress === "number"
      ? evt.progress
      : 0;

  return {
    overallProgress: clamp(overallProgress, 0, 100),
    message: evt.message || "",
    stage: evt.stage || "",
    stageProgress: typeof evt.stageProgress === "number" ? clamp(evt.stageProgress, 0, 1) : undefined,
    done: evt.done === true || overallProgress >= 100,
    raw: evt,
  };
}

function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}
