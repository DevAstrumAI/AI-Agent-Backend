"""
RAG Backend — api/livekit_router.py
=====================================
Generates LiveKit access tokens for the frontend.

GET /livekit/token?room=room-abc&identity=user-123
  → returns { token: "eyJ..." }

Add to app.py:
    from api.livekit_router import router as livekit_router
    app.include_router(livekit_router)

Requirements:
    pip install livekit-api
"""

import os
import json
from fastapi import APIRouter, HTTPException, Query
from livekit.api import AccessToken, VideoGrants

router = APIRouter(prefix="/livekit", tags=["LiveKit"])

LIVEKIT_API_KEY    = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")


@router.get("/token")
async def get_token(
    room:     str = Query(..., description="LiveKit room name"),
    identity: str = Query(..., description="Participant identity (e.g. user-abc123)"),
    mode:     str = Query(default="rag", description="Session mode: rag|booking"),
):
    """
    Generate a short-lived LiveKit access token for a participant.
    Called by the React frontend before connecting to a voice session.
    """
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(
            status_code=500,
            detail="LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in .env"
        )

    mode_norm = (mode or "rag").strip().lower()
    if mode_norm not in ("rag", "booking"):
        raise HTTPException(status_code=422, detail="mode must be 'rag' or 'booking'")

    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_metadata(json.dumps({"mode": mode_norm}))
        .with_grants(VideoGrants(
            room_join=True,
            room=room,
            can_publish=True,       # user can send audio
            can_subscribe=True,     # user can hear agent
            can_publish_data=True,  # DataChannel messages
        ))
        .to_jwt()
    )

    return {"token": token, "room": room, "identity": identity}