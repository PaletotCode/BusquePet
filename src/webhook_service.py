"""FastAPI webhook that exposes BusquePet's hybrid matcher."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference import PetMatchingPipeline
from system_utils import configure_logging, ensure_paths_exist

LOGGER = configure_logging(__name__)


class Settings(BaseModel):
    model_path: Path = Field(default=Path("models/contrastive/best_model.pt"))
    faiss_index_path: Path = Field(default=Path("models/faiss_index"))
    metadata_path: Path = Field(default=Path("data/embeddings/embedding_metadata.csv"))
    base_model_name: str = Field(default="ISxOdin/vit-base-oxford-iiit-pets")
    embedding_dim: int = Field(default=768)
    default_top_k: int = Field(default=10)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001)

    @classmethod
    def from_env(cls) -> "Settings":
        fields = cls.model_fields
        return cls(
            model_path=Path(os.getenv("BUSQUEPET_MODEL_PATH", fields["model_path"].default)),
            faiss_index_path=Path(os.getenv("BUSQUEPET_FAISS_INDEX", fields["faiss_index_path"].default)),
            metadata_path=Path(os.getenv("BUSQUEPET_METADATA_PATH", fields["metadata_path"].default)),
            base_model_name=os.getenv("BUSQUEPET_BASE_MODEL", fields["base_model_name"].default),
            embedding_dim=int(os.getenv("BUSQUEPET_EMBED_DIM", fields["embedding_dim"].default)),
            default_top_k=int(os.getenv("BUSQUEPET_DEFAULT_TOP_K", fields["default_top_k"].default)),
            host=os.getenv("BUSQUEPET_WEBHOOK_HOST", fields["host"].default),
            port=int(os.getenv("BUSQUEPET_WEBHOOK_PORT", fields["port"].default)),
        )


class InferenceMetadata(BaseModel):
    raw_path: Optional[str] = None
    original_filename: Optional[str] = None
    content_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    received_at: Optional[str] = None
    phash_source: Optional[str] = None
    normalized_storage: Optional[str] = None
    extra: Dict[str, Any] | None = None


class MatchRequest(BaseModel):
    job_id: str
    image_path: str = Field(..., description="Absolute path of the normalized image produced by the Go API.")
    phash: Optional[str] = Field(default=None, description="Optional perceptual hash pre-calculated by Go.")
    k: Optional[int] = Field(default=None, description="Desired number of matches.")
    use_hybrid: bool = True
    metadata: Optional[InferenceMetadata] = None


class MatchItem(BaseModel):
    rank: int
    image_id: str
    breed: str
    score: float
    distance: Optional[float] = None
    image_path: str
    metadata: Dict[str, Any] | None = None


class MatchResponse(BaseModel):
    job_id: str
    matches: List[MatchItem]
    metadata: Dict[str, Any] | None = None
    duration_ms: int


@lru_cache()
def get_settings() -> Settings:
    settings = Settings.from_env()
    required_paths = (
        settings.model_path,
        settings.faiss_index_path,
        settings.metadata_path,
    )
    ensure_paths_exist(required_paths)
    return settings


@lru_cache()
def get_pipeline() -> PetMatchingPipeline:
    cfg = get_settings()
    LOGGER.info("Loading PetMatchingPipeline (model=%s)", cfg.model_path)
    pipeline = PetMatchingPipeline(
        model_path=str(cfg.model_path),
        faiss_index_path=str(cfg.faiss_index_path),
        metadata_path=str(cfg.metadata_path),
        base_model_name=cfg.base_model_name,
        embedding_dim=cfg.embedding_dim,
    )
    LOGGER.info("Pipeline ready with %d indexed rows", len(pipeline.metadata_df))
    return pipeline


def create_app() -> FastAPI:
    app = FastAPI(title="BusquePet Webhook", version="1.0.0")

    @app.on_event("startup")
    def _startup() -> None:  # pragma: no cover - import-time effect
        get_settings()
        get_pipeline()

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/match", response_model=MatchResponse)
    def match(request: MatchRequest) -> JSONResponse:
        started = time.perf_counter()
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=400, detail=f"Image path does not exist: {image_path}")

        settings = get_settings()
        pipeline = get_pipeline()

        top_k = request.k or settings.default_top_k
        results = pipeline.search(
            query_image_path=str(image_path),
            k=top_k,
            use_hybrid=request.use_hybrid,
            query_phash=request.phash,
        )

        matches = []
        for item in results:
            matches.append(
                MatchItem(
                    rank=int(item.get("rank", 0)),
                    image_id=str(item.get("image_id", "")),
                    breed=str(item.get("breed", "")),
                    score=float(item.get("score", item.get("embedding_score", 0.0))),
                    distance=float(item.get("distance", item.get("embedding_distance", 0.0))),
                    image_path=str(item.get("image_path", "")),
                    metadata={
                        "phash_score": item.get("phash_score"),
                        "embedding_score": item.get("embedding_score"),
                    },
                )
            )

        duration = int((time.perf_counter() - started) * 1000)
        response = MatchResponse(
            job_id=request.job_id,
            matches=matches,
            metadata=request.metadata.model_dump(exclude_unset=True) if request.metadata else None,
            duration_ms=duration,
        )
        return JSONResponse(content=response.model_dump())

    return app


app = create_app()


if __name__ == "__main__":
    cfg = get_settings()
    import uvicorn

    uvicorn.run("webhook_service:app", host=cfg.host, port=cfg.port, reload=False)
