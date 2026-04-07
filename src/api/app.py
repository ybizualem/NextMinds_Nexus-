"""
FastAPI application for the CTIC Curriculum Recommendation Engine.

Part 2: Backend API
- POST /api/search   — Semantic search with optional filters
- POST /api/chat     — Conversational Gemini assistant with function calling
- DELETE /api/chat/{session_id} — Clear chat session
- POST /api/admin/ingest  — Trigger re-crawl + ingestion
- GET  /api/admin/stats   — Catalog statistics
- GET  /api/admin/health  — System health check

Run with:
    python -m src.api.app
    # or
    uvicorn src.api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)

try:
    from src.api.routes.search import router as search_router
    from src.api.routes.chat import router as chat_router
    from src.api.routes.admin import router as admin_router
    logging.info("Routes imported successfully")
except Exception as e:
    logging.exception("Failed to import routes")
    raise

app = FastAPI(
    title="CTIC Curriculum Engine API",
    description="Conversational recommendation engine for CTIC curriculum activities. "
                "Helps teachers discover and share invention-based learning resources.",
    version="0.2.0",
)

# CORS — allow the React frontend (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
try:
    app.include_router(search_router)
    app.include_router(chat_router)
    app.include_router(admin_router)
    logging.info("Routers registered successfully")
except Exception as e:
    logging.exception("Failed to register routers")
    raise


@app.get("/")
async def root():
    return {
        "name": "CTIC Curriculum Engine API",
        "version": "0.2.0",
        "docs": "/docs",
        "endpoints": {
            "search": "POST /api/search",
            "chat": "POST /api/chat",
            "stats": "GET /api/admin/stats",
            "health": "GET /api/admin/health",
            "ingest": "POST /api/admin/ingest",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
