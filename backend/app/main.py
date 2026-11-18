"""FastAPI application factory.  Run:  uvicorn app.main:app --reload   (from the backend/ folder)"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api.routes import router
from .core.artifacts import ArtifactError, load_bundle
from .core.config import Settings
from .services.image_io import ImageValidationError
from .services.model_registry import ModelRegistry, ModelsUnavailable, UnknownModel

log = logging.getLogger("ecosort")


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(settings: Settings | None = None, load_models: bool = True) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            bundle, problems = load_bundle(settings.artifacts_dir, settings.verify_hashes), []
        except ArtifactError as exc:
            bundle, problems = None, exc.problems
            log.error("Artifacts invalid: %s", problems)
        registry = ModelRegistry(settings, bundle, problems)
        if load_models:
            registry.load_all()            # models are loaded ONCE here, never per request
        app.state.registry = registry
        yield

    app = FastAPI(title="EcoSort AI API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"], allow_headers=["*"])
    app.include_router(router)

    @app.exception_handler(ImageValidationError)
    async def _img(_: Request, exc: ImageValidationError):
        return _err(exc.status_code, exc.code, exc.message)

    @app.exception_handler(ModelsUnavailable)
    async def _unavail(_: Request, exc: ModelsUnavailable):
        return _err(503, "models_unavailable", str(exc))

    @app.exception_handler(UnknownModel)
    async def _unknown(_: Request, exc: UnknownModel):
        return _err(404, "unknown_model", str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        d = exc.detail
        if isinstance(d, dict) and "code" in d:
            return _err(exc.status_code, d["code"], d["message"])
        return _err(exc.status_code, "http_error", str(d))

    @app.exception_handler(RequestValidationError)
    async def _val(_: Request, exc: RequestValidationError):
        return _err(422, "bad_request", "The request is missing a required field (for example the image file).")

    @app.exception_handler(Exception)
    async def _any(_: Request, exc: Exception):
        log.exception("unhandled error")
        return _err(500, "internal_error", "Something went wrong while processing the image.")

    return app


app = create_app()
