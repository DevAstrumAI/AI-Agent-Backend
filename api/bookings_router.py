"""
RAG Backend — api/bookings_router.py
========================================
Full CRUD for bookings with advanced filtering.

POST   /bookings/              — create booking (called by agent)
GET    /bookings/              — list all bookings with optional filters
GET    /bookings/{id}          — get single booking by ID or confirmation number
PATCH  /bookings/{id}          — update booking fields
DELETE /bookings/{id}          — hard delete booking
POST   /bookings/{id}/cancel   — soft cancel (sets status=cancelled)
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from database.models import (
    save_booking,
    get_all_bookings,
    get_bookings_filtered,
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
    status:          Optional[str] = None
    patient_name:    Optional[str] = None
    language:        Optional[str] = None
    session_summary: Optional[str] = None
    slot_date:       Optional[str] = None
    slot_time:       Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/", status_code=201)
async def create_booking(payload: BookingCreate):
    import time
    import logging
    log = logging.getLogger(__name__)
    t0 = time.perf_counter()
    try:
        record = await save_booking(
            service_name    = payload.service_name,
            doctor_name     = payload.doctor_name,
            patient_name    = payload.patient_name,
            slot_id         = payload.slot_id,
            language        = payload.language,
            session_summary = payload.session_summary,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        msg = f"[HTTP][bookings] POST /bookings/ completed in {dt_ms:.1f}ms"
        print(msg, flush=True)
        log.info(msg)
        return {"success": True, "booking": record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_bookings(
    status:       Optional[str] = Query(default=None, description="Filter by status: confirmed|cancelled|no_show"),
    doctor_name:  Optional[str] = Query(default=None, description="Filter by doctor name (partial, case-insensitive)"),
    service_name: Optional[str] = Query(default=None, description="Filter by service name (partial, case-insensitive)"),
    date_from:    Optional[str] = Query(default=None, description="Filter from date YYYY-MM-DD"),
    date_to:      Optional[str] = Query(default=None, description="Filter to date YYYY-MM-DD"),
    time_from:    Optional[str] = Query(default=None, description="Filter from time HH:MM"),
    time_to:      Optional[str] = Query(default=None, description="Filter to time HH:MM"),
    search:       Optional[str] = Query(default=None, description="Search patient name or confirmation number"),
):
    """
    List bookings with optional filters.
    All params are optional and combinable.
    """
    has_filters = any(v is not None for v in [
        status, doctor_name, service_name,
        date_from, date_to, time_from, time_to, search
    ])

    if has_filters:
        bookings = await get_bookings_filtered(
            status       = status,
            doctor_name  = doctor_name,
            service_name = service_name,
            date_from    = date_from,
            date_to      = date_to,
            time_from    = time_from,
            time_to      = time_to,
            search       = search,
        )
    else:
        bookings = await get_all_bookings()

    return {"bookings": bookings, "count": len(bookings)}


@router.get("/{booking_id}")
async def get_booking(booking_id: str):
    booking = await get_booking_by_id(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    return booking


@router.patch("/{booking_id}")
async def patch_booking(booking_id: str, payload: BookingUpdate):
    booking = await get_booking_by_id(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    updated = await update_booking(booking_id, **payload.model_dump(exclude_none=True))
    return {"success": True, "booking": updated}


@router.post("/{booking_id}/cancel")
async def cancel_booking(booking_id: str):
    booking = await get_booking_by_id(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    updated = await cancel_booking_db(booking_id)
    return {"success": True, "booking": updated}


@router.delete("/{booking_id}")
async def delete_booking_endpoint(booking_id: str):
    deleted = await delete_booking(booking_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Booking not found")
    return {"success": True, "deleted_id": booking_id}