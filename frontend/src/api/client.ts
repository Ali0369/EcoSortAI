import type { CategoriesResponse, CompareResponse, DatasetSummary, ExplainResponse, Health, MetricsPayload, ModelInfo, Prediction, SampleInfo } from "./types";

const BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) { super(message); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "network", "Can't reach the EcoSort server. Check that the backend is running and try again.");
  }
  if (!res.ok) {
    let code = "http_error", message = `The server answered with an error (${res.status}).`;
    try { const j = await res.json(); if (j?.error) { code = j.error.code; message = j.error.message; } } catch { /* keep defaults */ }
    throw new ApiError(res.status, code, message);
  }
  return res.json() as Promise<T>;
}

function form(file: File, extra: Record<string, string | undefined> = {}) {
  const f = new FormData();
  f.append("file", file);
  Object.entries(extra).forEach(([k, v]) => { if (v) f.append(k, v); });
  return f;
}

export const api = {
  health: () => request<Health>("/api/health"),
  models: () => request<ModelInfo[]>("/api/models"),
  metrics: () => request<MetricsPayload>("/api/models/compare"),
  categories: () => request<CategoriesResponse>("/api/categories"),
  dataset: () => request<DatasetSummary>("/api/dataset"),
  samples: () => request<{ samples: SampleInfo[] }>("/api/samples"),
  sampleUrl: (name: string) => `${BASE}/api/samples/${encodeURIComponent(name)}`,
  predict: (file: File, modelId?: string) => request<Prediction>("/api/predict", { method: "POST", body: form(file, { model_id: modelId }) }),
  compare: (file: File, modelIds?: string[]) => request<CompareResponse>("/api/predict/compare", { method: "POST", body: form(file, { model_ids: modelIds?.join(",") }) }),
  explain: (file: File, modelId?: string) => request<ExplainResponse>("/api/explain", { method: "POST", body: form(file, { model_id: modelId }) }),
};
