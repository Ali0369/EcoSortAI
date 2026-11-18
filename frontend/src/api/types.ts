export interface TopPrediction { class_name: string; probability: number }
export interface Prediction {
  model_id: string; predicted_class: string; confidence: number; is_uncertain: boolean; uncertainty_threshold: number;
  top3: TopPrediction[]; probabilities: Record<string, number>; inference_ms: number;
}
export interface CompareResponse { predictions: Prediction[]; all_agree: boolean; image_size: number[] }
export interface ModelInfo {
  id: string; code: string; name: string; role: string; description: string; is_proposed: boolean; is_default: boolean;
  uncertainty_threshold: number; loaded: boolean;
}
export interface Health {
  status: "ok" | "degraded"; models_loaded: number; models_expected: number; artifacts_ok: boolean; smoke_test: boolean;
  runtime: string | null; problems: string[]; warnings?: string[];
}
export interface CategoryInfo {
  id: string; label: string; recyclable: boolean; headline: string; guidance: string; summary: string;
  steps: { title: string; text: string }[];
}
export interface CategoriesResponse { categories: CategoryInfo[]; station_order: string[]; card_order: string[]; disclaimer: string }

export interface ModelMetrics {
  code: string; name: string; seed: number; train_accuracy: number; val_accuracy: number; test_accuracy: number;
  macro_precision: number; macro_recall: number; macro_f1: number; confusion_matrix: number[][];
  per_class: Record<string, { precision: number; recall: number; "f1-score": number; support: number }>;
  training_seconds: number; epochs_run: number; inference_ms_cpu: number | null; model_size_mb: number; params_total: number;
  uncertainty: { threshold: number; test?: { coverage: number; selective_accuracy: number | null } };
}
export interface MetricsPayload {
  metrics: {
    smoke_test?: boolean; class_names: string[]; generated_utc: string; models: Record<string, ModelMetrics>;
    statistics?: { mcnemar_e2_vs_e3?: { a_right_b_wrong: number; a_wrong_b_right: number; p_value: number } };
    protocol?: string;
  };
  models: { id: string; code: string; name: string; role: string; description: string; is_proposed: boolean }[];
  default_model_id: string;
}
export interface DatasetSummary {
  name: string; source: string; license: string; classes: string[]; class_counts: Record<string, number>; total_images: number;
  splits: Record<string, { n: number; fraction: number }>; smoke_test?: boolean;
}
export interface ExplainResponse {
  model_id: string; explained_class: string; confidence: number; original_png_b64: string; heatmap_png_b64: string;
  overlay_png_b64: string; peak_region: string; high_attention_area_fraction: number; feature_map_size: number[];
}
export interface SampleInfo { file: string; kind: string; true_label: string; note: string }
