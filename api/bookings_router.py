"""
RAG Backend — api/bookings_router.py
========================================
Full CRUD for bookings.

POST   /bookings/              — create booking (called by agent)
GET    /bookings/              — list all bookings (optional ?status=confirmed)
GET    /bookings/{id}          — get single booking by ID or confirmation number
PATCH  /bookings/{id}          — update booking fields (status, patient_name, etc.)
DELETE /bookings/{id}          — hard delete booking
POST   /bookings/{id}/cancel   — soft cancel (sets status=cancelled)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from database.models import (
    save_booking,
    get_all_bookings,
    get_booking_by_id,
    update_booking,
    delete_booking,
    cancel_booking_db,
)

router = APIRouter(prefix="/bookings", tags=["Bookings"])


# ── Pydantic models ───────────────────────────────────────────

class BookingCreate(BaseModel):
    service_name:    str
    doctor_name:     str
    patient_name:    str
    slot_id:         Optional[str] = None
    language:        str = "en"
    session_summary: Optional[str] = None


class BookingUpdate(BaseModel):
    status:          Optional[str] = None   # confirmed | cancelled | no_show
    patient_name:    Optional[str] = None
    language:        Optional[str] = None
    session_summary: Optional[str] = None
    slot_date:       Optional[str] = None
    slot_time:       Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/", status_code=201)
async def create_booking(payload: BookingCreate):
    """
    Save a confirmed booking. Called by the agent after user says YES.
    Marks the slot unavailable if slot_id is provided.
    """
    try:
        record = await save_booking(
            service_name    = payload.service_name,
            doctor_name     = payload.doctor_name,
            patient_name    = payload.patient_name,
            slot_id         = payload.slot_id,
            language        = payload.language,
            session_summary = payload.session_summary,
        )
        return {"success": True, "booking": record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_bookings(status: Optional[str] = None):
    """
    List all bookings. Filter by ?status=confirmed|cancelled|no_show
    """
    bookings = await get_all_bookings(status=status)
    return {"bookings": bookings, "count": len(bookings)}


@router.get("/{booking_id}")
async def get_booking(booking_id: str):
    """
    Get a single booking.
    booking_id can be an integer ID (e.g. 1, 2, 3)
    or a confirmation number (e.g. FM-2026-AB1234).
    """
    booking = await get_booking_by_id(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    return booking


@router.patch("/{booking_id}")
async def patch_booking(booking_id: str, payload: BookingUpdate):
    """
    Update booking fields. booking_id accepts integer or confirmation number.
    """
    booking = await get_booking_by_id(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    updated = await update_booking(booking_id, **payload.model_dump(exclude_none=True))
    return {"success": True, "booking": updated}


@router.post("/{booking_id}/cancel")
async def cancel_booking(booking_id: str):
    """Soft cancel — sets status to 'cancelled'. booking_id accepts integer or confirmation number."""
    booking = await get_booking_by_id(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    updated = await cancel_booking_db(booking_id)
    return {"success": True, "booking": updated}


@router.delete("/{booking_id}")
async def delete_booking_endpoint(booking_id: str):
    """Hard delete. booking_id accepts integer or confirmation number."""
    deleted = await delete_booking(booking_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Booking not found")
    return {"success": True, "deleted_id": booking_id}