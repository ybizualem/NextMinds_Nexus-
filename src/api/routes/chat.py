"""Chat route — conversational Gemini endpoint with function calling."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.models import ChatRequest, ChatResponse, ActivityResult
from src.api.chat_engine import chat, clear_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    """Send a message to the CTIC Curriculum Assistant."""
    try:
        reply, session_id, activities = chat(
            message=req.message,
            session_id=req.session_id,
        )
    except RuntimeError as e:
        # Missing API key, config errors
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail="Chat request failed.")

    activity_results = [
        ActivityResult(
            activity_name=a["activity_name"],
            grade_band=a["grade_band"],
            stage=a["stage"],
            resource_url=a["resource_url"],
            resource_type=a["resource_type"],
            drive_id=a.get("drive_id"),
            similarity=a.get("similarity"),
        )
        for a in activities
    ]

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        activities=activity_results,
    )


@router.delete("/chat/{session_id}")
async def api_clear_chat(session_id: str):
    """Clear conversation history for a session."""
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
