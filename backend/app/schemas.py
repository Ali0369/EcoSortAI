"""Pydantic response models (the API contract)."""
from __future__ import annotations

from pydantic import BaseModel


class TopPrediction(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    model_id: str
    predicted_class: str
    confidence: float
    is_uncertain: bool
    uncertainty_threshold: float
    top3: list[TopPrediction]
    probabilities: dict[str, float]
    inference_ms: float


class CompareResponse(BaseModel):
    predictions: list[PredictionResponse]
    all_agree: bool
    image_size: list[int]


class ModelInfo(BaseModel):
    id: str
    code: str
    name: str
    role: str
    description: str
    is_proposed: bool
    is_default: bool
    uncertainty_threshold: float
    loaded: bool


class HealthResponse(BaseModel):
    status: str                    # "ok" | "degraded"
    models_loaded: int
    models_expected: int
    artifacts_ok: bool
    smoke_test: bool
    runtime: str | None = None
    problems: list[str] = []
    warnings: list[str] = []


class ExplainResponse(BaseModel):
    model_id: str
    explained_class: str
    confidence: float
    original_png_b64: str
    heatmap_png_b64: str
    overlay_png_b64: str
    peak_region: str
    high_attention_area_fraction: float
    feature_map_size: list[int]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
