"""HTTP endpoints. All prediction endpoints run the real trained models via the registry."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ..categories import CARD_ORDER, CATEGORIES, STATION_ORDER
from ..schemas import (CompareResponse, ExplainResponse, HealthResponse, ModelInfo, PredictionResponse)
from ..services.image_io import decode_upload

router = APIRouter(prefix="/api")


def _registry(request: Request):
    return request.app.state.registry


async def _read_image(request: Request, file: UploadFile):
    settings = request.app.state.settings
    raw = await file.read(settings.max_upload_bytes + 1)      # never read more than the limit + 1 byte
    return decode_upload(raw, file.content_type, settings)


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    reg = _registry(request)
    expected = len(reg.bundle.models) if reg.bundle else 0
    ok = reg.ready and len(reg.models) == expected and not reg.problems
    return HealthResponse(status="ok" if ok else "degraded", models_loaded=len(reg.models), models_expected=expected,
                          artifacts_ok=reg.bundle is not None, smoke_test=bool(reg.bundle and reg.bundle.smoke_test),
                          runtime=reg.tf_version, problems=reg.problems, warnings=reg.warnings)


@router.get("/models", response_model=list[ModelInfo])
def list_models(request: Request):
    reg = _registry(request)
    if reg.bundle is None:
        raise HTTPException(503, detail={"code": "artifacts_missing", "message": "; ".join(reg.problems)})
    return [ModelInfo(id=m["id"], code=m["code"], name=m["name"], role=m["role"], description=m["description"],
                      is_proposed=m["is_proposed"], is_default=m["id"] == reg.bundle.default_model_id,
                      uncertainty_threshold=reg.bundle.thresholds[m["id"]]["threshold"], loaded=m["id"] in reg.models)
            for m in reg.bundle.models]


@router.get("/models/compare")
def compare_metrics(request: Request):
    """The real experiment metrics exported by the notebook (metrics.json) - nothing is added here."""
    reg = _registry(request)
    if reg.bundle is None:
        raise HTTPException(503, detail={"code": "artifacts_missing", "message": "; ".join(reg.problems)})
    return {"metrics": reg.bundle.metrics, "models": reg.bundle.models, "default_model_id": reg.bundle.default_model_id,
            "thresholds": reg.bundle.thresholds}


@router.get("/categories")
def categories(request: Request):
    reg = _registry(request)
    names = reg.bundle.class_names if reg.bundle else list(CATEGORIES)
    items = [{"id": n, **{k: v for k, v in CATEGORIES[n].items()}, "steps": [{"title": t, "text": x} for t, x in CATEGORIES[n]["steps"]]}
             for n in names if n in CATEGORIES]
    return {"categories": items, "station_order": [n for n in STATION_ORDER if n in names], "card_order": [n for n in CARD_ORDER if n in names],
            "disclaimer": "Recycling rules vary by location. Always check your local collection guidelines."}


@router.get("/dataset")
def dataset(request: Request):
    reg = _registry(request)
    if reg.bundle is None or reg.bundle.dataset_summary is None:
        raise HTTPException(404, detail={"code": "dataset_summary_missing", "message": "dataset_summary.json was not found in the artifacts."})
    return reg.bundle.dataset_summary


@router.get("/samples")
def samples(request: Request):
    reg = _registry(request)
    return {"samples": reg.bundle.samples if reg.bundle else []}


@router.get("/samples/{name}")
def sample_file(name: str, request: Request):
    reg = _registry(request)
    allowed = {s["file"] for s in (reg.bundle.samples if reg.bundle else [])}
    if name not in allowed:                                   # whitelist: no path traversal possible
        raise HTTPException(404, detail={"code": "not_found", "message": "Unknown sample."})
    return FileResponse(reg.bundle.root / "samples" / name)


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...), model_id: str | None = Form(None)):
    reg = _registry(request)
    reg.require_ready()
    image = await _read_image(request, file)
    return reg.predict(model_id, image)


@router.post("/predict/compare", response_model=CompareResponse)
async def predict_compare(request: Request, file: UploadFile = File(...), model_ids: str | None = Form(None)):
    """Run the SAME image through several models. ``model_ids`` = comma-separated ids (default: all loaded)."""
    reg = _registry(request)
    reg.require_ready()
    image = await _read_image(request, file)
    ids = [m.strip() for m in model_ids.split(",") if m.strip()] if model_ids else None
    for i in ids or []:
        reg.resolve(i)
    preds = reg.compare(ids, image)
    return CompareResponse(predictions=preds, all_agree=len({p["predicted_class"] for p in preds}) == 1, image_size=[int(image.shape[1]), int(image.shape[0])])


@router.post("/explain", response_model=ExplainResponse)
async def explain(request: Request, file: UploadFile = File(...), model_id: str | None = Form(None)):
    reg = _registry(request)
    reg.require_ready()
    image = await _read_image(request, file)
    return reg.explain(model_id, image)
