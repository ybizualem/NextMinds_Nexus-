"""Admin routes — trigger re-crawl, view stats, health check."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.models import IngestResponse, StatsResponse, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/ingest", response_model=IngestResponse)
async def api_ingest():
    """Trigger a full re-crawl and ingestion pipeline."""
    try:
        from src.ingest import run_full_ingestion
        summary = run_full_ingestion(triggered_by="api")
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    if "error" in summary:
        return IngestResponse(status="error")

    return IngestResponse(
        status="completed",
        total_crawled=summary.get("total_crawled", 0),
        added=summary.get("added", 0),
        updated=summary.get("updated", 0),
        removed=summary.get("removed", 0),
        errors=summary.get("errors", 0),
    )


@router.get("/stats", response_model=StatsResponse)
async def api_stats():
    """Get catalog statistics."""
    try:
        from src.db.operations import get_activity_stats
        stats = get_activity_stats()
    except Exception as e:
        logger.exception("Stats query failed")
        raise HTTPException(status_code=500, detail="Could not retrieve stats.")

    return StatsResponse(
        total=stats["total"],
        active=stats["active"],
        grade_bands=stats["grade_bands"],
        stages=stats["stages"],
        oldest_crawl=str(stats.get("oldest_crawl", "")),
        newest_crawl=str(stats.get("newest_crawl", "")),
        by_grade_band=stats.get("by_grade_band", {}),
        by_stage=stats.get("by_stage", {}),
    )


@router.get("/health", response_model=HealthResponse)
async def api_health():
    """Check API health: database connectivity and embedding model."""
    db_status = "unknown"
    model_status = "unknown"

    # Check database
    try:
        from src.db.operations import get_connection
        with get_connection() as conn:
            conn.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    # Check embedding model
    try:
        from src.embeddings.embedder import get_model
        get_model()
        model_status = "loaded"
    except Exception as e:
        model_status = f"error: {e}"

    overall = "healthy" if db_status == "connected" and model_status == "loaded" else "degraded"

    return HealthResponse(
        status=overall,
        database=db_status,
        embedding_model=model_status,
    )
