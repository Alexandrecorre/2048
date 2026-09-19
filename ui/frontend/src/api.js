const BASE_URL = "http://127.0.0.1:8000";

async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  experiments: () => get("/api/experiments"),
  experimentSummary: (name) => get(`/api/experiments/${encodeURIComponent(name)}/summary`),
  experimentScores: (name) => get(`/api/experiments/${encodeURIComponent(name)}/scores`),
  experimentReplays: (name) => get(`/api/experiments/${encodeURIComponent(name)}/replays`),
  experimentReplay: (name, seed) =>
    get(`/api/experiments/${encodeURIComponent(name)}/replay/${seed}`),
  ablationMatrix: () => get("/api/ablation/matrix"),
  ablationLearningCurves: () => get("/api/ablation/learning-curves"),
  searchExperiment: () => get("/api/search-experiment"),
  dqnHistory: (representation) => get(`/api/dqn/${representation}`),
};
