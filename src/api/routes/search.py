"""Search route — exposes the existing pgvector search as an HTTP endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.embeddings.embedder import embed_text
from src.db.operations import search_activities
from src.api.models import SearchRequest, SearchResponse, ActivityResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def api_search(req: SearchRequest):
    """Semantic search over the curriculum catalog."""
    try:
        query_embedding = embed_text(req.query)
        rows = search_activities(
            query_embedding=query_embedding,
            grade_band=req.grade_band,
            stage=req.stage,
            max_time=req.max_time,
            limit=req.limit,
        )
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail="Search failed — check database connection.")

    results = [
        ActivityResult(
            activity_name=r["activity_name"],
            grade_band=r["grade_band"],
            stage=r["stage"],
            resource_url=r["resource_url"],
            resource_type=r["resource_type"],
            drive_id=r.get("drive_id"),
            similarity=r.get("similarity"),
        )
        for r in rows
    ]

    return SearchResponse(query=req.query, results=results, count=len(results))
