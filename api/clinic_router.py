"""
RAG Backend — api/clinic_router.py
=====================================
Full CRUD for services, doctors, and slots.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import os
import threading
import logging
import time

from database.models import (
    get_all_services, get_service_by_id, create_service, update_service, delete_service,
    get_services_filtered,
    get_all_doctors, get_doctor_by_id, create_doctor, update_doctor, delete_doctor,
    get_services_for_doctor, assign_service_to_doctor, remove_service_from_doctor,
    get_doctors_for_service_db, get_doctors_filtered,
    get_all_slots, get_slot_by_id, create_slot, update_slot, delete_slot,
    get_available_slots,
)

router = APIRouter(prefix="/clinic", tags=["Clinic"])
log = logging.getLogger(__name__)

# In-memory TTL caches for the agent/booking read-path.
# These endpoints are hit frequently by new LiveKit workers, so caching them
# eliminates repeated DB latency.
_SERVICES_CACHE_TTL_S = float(os.getenv("CLINIC_SERVICES_CACHE_TTL_S", "60"))
_DOCTORS_CACHE_TTL_S  = float(os.getenv("CLINIC_DOCTORS_CACHE_TTL_S", "60"))
_DOCTOR_SERVICES_CACHE_TTL_S = float(os.getenv("CLINIC_DOCTOR_SERVICES_CACHE_TTL_S", "60"))
_DOCTORS_ALL_CACHE_TTL_S = float(os.getenv("CLINIC_DOCTORS_ALL_CACHE_TTL_S", "60"))

_services_cache: dict[str, object] = {"value": None, "expires_at": 0.0}
_doctors_cache: dict[str, dict[str, object]] = {}
_doctor_services_cache: dict[str, dict[str, object]] = {}
_doctors_all_cache: dict[str, object] = {"value": None, "expires_at": 0.0}
_cache_lock = threading.Lock()


def _cache_get(entry: dict[str, object]) -> object | None:
    expires_at = float(entry.get("expires_at") or 0.0)
    if time.monotonic() < expires_at:
        return entry.get("value")
    return None


def _cache_set(entry: dict[str, object], value: object, ttl_s: float) -> None:
    entry["value"] = value
    entry["expires_at"] = time.monotonic() + ttl_s


def _invalidate_clinic_caches() -> None:
    with _cache_lock:
        _services_cache["value"] = None
        _services_cache["expires_at"] = 0.0
        _doctors_cache.clear()
        _doctor_services_cache.clear()
        _doctors_all_cache["value"] = None
        _doctors_all_cache["expires_at"] = 0.0


# ─────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────

class ServiceCreate(BaseModel):
    name:             str
    description:      Optional[str] = ""
    duration_minutes: Optional[int] = 60


class ServiceUpdate(BaseModel):
    name:             Optional[str] = None
    description:      Optional[str] = None
    duration_minutes: Optional[int] = None
    active:           Optional[int] = None


class DoctorCreate(BaseModel):
    full_name:  str
    title:      str
    department: str
    bio:        str


class DoctorUpdate(BaseModel):
    full_name:  Optional[str] = None
    title:      Optional[str] = None
    department: Optional[str] = None
    bio:        Optional[str] = None
    active:     Optional[int] = None


class AssignService(BaseModel):
    service_id: str


class SlotCreate(BaseModel):
    doctor_id:  str
    service_id: str
    slot_date:  str
    slot_time:  str


class SlotUpdate(BaseModel):
    slot_date:  Optional[str] = None
    slot_time:  Optional[str] = None
    available:  Optional[int] = None
    doctor_id:  Optional[str] = None
    service_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# SERVICES
# ─────────────────────────────────────────────────────────────

@router.get("/services")
async def list_services(
    search:    Optional[str] = Query(default=None, description="Search by name"),
    doctor_id: Optional[str] = Query(default=None, description="Filter by doctor UUID"),
    duration:  Optional[int] = Query(default=None, description="Filter by exact duration in minutes"),
    active:    Optional[int] = Query(default=None, description="Filter by status: 1=active, 0=inactive"),
):
    if any(v is not None for v in [search, doctor_id, duration, active]):
        services = await get_services_filtered(
            search    = search,
            doctor_id = doctor_id,
            duration  = duration,
            active    = active,
        )
    else:
        # Cache only the unfiltered "agent booking" request.
        with _cache_lock:
            cached = _cache_get(_services_cache)
        if cached is None:
            t0 = time.perf_counter()
            services = await get_all_services()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with _cache_lock:
                _cache_set(_services_cache, services, _SERVICES_CACHE_TTL_S)
            msg = f"[HTTP][clinic][services] all returned={len(services)} in {dt_ms:.1f}ms"
            print(msg, flush=True)
            log.info(msg)
        else:
            services = cached  # type: ignore[assignment]
            msg = f"[HTTP][clinic][services] all returned={len(services)} (cache hit)"
            print(msg, flush=True)
            log.info(msg)

    return {
        "services": services,
        "names":    [s["name"] for s in services if s.get("active", 1)],
        "count":    len(services),
    }


@router.post("/services", status_code=201)
async def add_service(payload: ServiceCreate):
    try:
        service = await create_service(
            name             = payload.name,
            description      = payload.description or "",
            duration_minutes = payload.duration_minutes or 60,
        )
        _invalidate_clinic_caches()
        return {"success": True, "service": service}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/services/{service_id}")
async def get_service(service_id: str):
    service = await get_service_by_id(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    return service


@router.patch("/services/{service_id}")
async def patch_service(service_id: str, payload: ServiceUpdate):
    service = await get_service_by_id(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    updated = await update_service(service_id, **payload.model_dump(exclude_none=True))
    _invalidate_clinic_caches()
    return {"success": True, "service": updated}


@router.delete("/services/{service_id}")
async def remove_service(service_id: str):
    deleted = await delete_service(service_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Service not found")
    _invalidate_clinic_caches()
    return {"success": True, "deleted_id": service_id}


# ─────────────────────────────────────────────────────────────
# DOCTORS
# ─────────────────────────────────────────────────────────────

@router.get("/doctors")
async def list_doctors(
    service:    Optional[str] = Query(default=None, description="Filter by service name (agent use)"),
    search:     Optional[str] = Query(default=None, description="Search by name"),
    department: Optional[str] = Query(default=None, description="Filter by department"),
    service_id: Optional[str] = Query(default=None, description="Filter by service UUID"),
    active:     Optional[int] = Query(default=None, description="Filter by status: 1=active, 0=inactive"),
):
    if service:
        key = (service or "").strip()
        with _cache_lock:
            entry = _doctors_cache.get(key)
            cached = _cache_get(entry) if entry else None
        if cached is None:
            t0 = time.perf_counter()
            doctors = await get_doctors_for_service_db(service)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with _cache_lock:
                _doctors_cache[key] = {"value": doctors, "expires_at": time.monotonic() + _DOCTORS_CACHE_TTL_S}
        else:
            doctors = cached  # type: ignore[assignment]
            dt_ms = -1.0
        if dt_ms >= 0:
            msg = f"[HTTP][clinic][doctors] service={service!r} returned={len(doctors)} in {dt_ms:.1f}ms"
        else:
            msg = f"[HTTP][clinic][doctors] service={service!r} returned={len(doctors)} (cache hit)"
        print(msg, flush=True)
        log.info(msg)
        return {
            "doctors": doctors,
            "names":   [d["full_name"] for d in doctors],
            "count":   len(doctors),
        }

    # Cache unfiltered GET /clinic/doctors (admin dashboard calls this a lot).
    if search is None and department is None and service_id is None and active is None:
        with _cache_lock:
            cached = _cache_get(_doctors_all_cache)
        if cached is None:
            t0 = time.perf_counter()
            doctors = await get_doctors_filtered(
                search=search,
                department=department,
                service_id=service_id,
                active=active,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with _cache_lock:
                _doctors_all_cache["value"] = doctors
                _doctors_all_cache["expires_at"] = time.monotonic() + _DOCTORS_ALL_CACHE_TTL_S
            print(f"[HTTP][clinic][doctors] all returned={len(doctors)} in {dt_ms:.1f}ms", flush=True)
        else:
            doctors = cached  # type: ignore[assignment]
            print(f"[HTTP][clinic][doctors] all returned={len(doctors)} (cache hit)", flush=True)
        return {"doctors": doctors, "count": len(doctors)}

    doctors = await get_doctors_filtered(
        search=search,
        department=department,
        service_id=service_id,
        active=active,
    )
    return {"doctors": doctors, "count": len(doctors)}


@router.post("/doctors", status_code=201)
async def add_doctor(payload: DoctorCreate):
    try:
        doctor = await create_doctor(
            full_name  = payload.full_name,
            title      = payload.title,
            department = payload.department,
            bio        = payload.bio,
        )
        _invalidate_clinic_caches()
        return {"success": True, "doctor": doctor}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/doctors/{doctor_id}")
async def get_doctor(doctor_id: str):
    doctor = await get_doctor_by_id(doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    return doctor


@router.patch("/doctors/{doctor_id}")
async def patch_doctor(doctor_id: str, payload: DoctorUpdate):
    doctor = await get_doctor_by_id(doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    updated = await update_doctor(doctor_id, **payload.model_dump(exclude_none=True))
    _invalidate_clinic_caches()
    return {"success": True, "doctor": updated}


@router.delete("/doctors/{doctor_id}")
async def remove_doctor(doctor_id: str):
    deleted = await delete_doctor(doctor_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Doctor not found")
    _invalidate_clinic_caches()
    return {"success": True, "deleted_id": doctor_id}


@router.get("/doctors/{doctor_id}/services")
async def get_doctor_services(doctor_id: str):
    doctor = await get_doctor_by_id(doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    with _cache_lock:
        entry = _doctor_services_cache.get(doctor_id)
        cached = _cache_get(entry) if entry else None
    if cached is None:
        t0 = time.perf_counter()
        services = await get_services_for_doctor(doctor_id)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        with _cache_lock:
            _doctor_services_cache[doctor_id] = {"value": services, "expires_at": time.monotonic() + _DOCTOR_SERVICES_CACHE_TTL_S}
        print(f"[HTTP][clinic][doctor-services] doctor_id={doctor_id!r} returned={len(services)} in {dt_ms:.1f}ms", flush=True)
    else:
        services = cached  # type: ignore[assignment]
        print(f"[HTTP][clinic][doctor-services] doctor_id={doctor_id!r} returned={len(services)} (cache hit)", flush=True)
    return {"doctor_id": doctor_id, "services": services, "count": len(services)}


@router.post("/doctors/{doctor_id}/services", status_code=201)
async def assign_doctor_service(doctor_id: str, payload: AssignService):
    await assign_service_to_doctor(doctor_id, payload.service_id)
    _invalidate_clinic_caches()
    services = await get_services_for_doctor(doctor_id)
    return {"success": True, "doctor_id": doctor_id, "services": services}


@router.delete("/doctors/{doctor_id}/services/{service_id}")
async def unassign_doctor_service(doctor_id: str, service_id: str):
    removed = await remove_service_from_doctor(doctor_id, service_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Assignment not found")
    _invalidate_clinic_caches()
    return {"success": True, "doctor_id": doctor_id, "removed_service_id": service_id}


# ─────────────────────────────────────────────────────────────
# SLOTS
# ─────────────────────────────────────────────────────────────

@router.get("/slots")
async def list_slots(
    service:        Optional[str] = Query(default=None, description="Service name (agent use)"),
    doctor:         Optional[str] = Query(default=None, description="Doctor name (agent use)"),
    doctor_id:      Optional[str] = Query(default=None, description="Filter by doctor UUID"),
    service_id:     Optional[str] = Query(default=None, description="Filter by service UUID"),
    available_only: bool          = Query(default=False, description="Only available slots"),
    date_from:      Optional[str] = Query(default=None, description="Start date filter YYYY-MM-DD"),
    date_to:        Optional[str] = Query(default=None, description="End date filter YYYY-MM-DD"),
    time_from:      Optional[str] = Query(default=None, description="Start time filter HH:MM"),
    time_to:        Optional[str] = Query(default=None, description="End time filter HH:MM"),
    limit:          int           = Query(default=6, ge=1, le=100),
):
    # Agent path — filter by service and doctor names
    if service and doctor:
        t0 = time.perf_counter()
        slots = await get_available_slots(
            service_name = service,
            doctor_name  = doctor,
            limit        = limit,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        msg = f"[HTTP][clinic][slots] service={service!r} doctor={doctor!r} returned={len(slots)} in {dt_ms:.1f}ms"
        print(msg, flush=True)
        log.info(msg)
        if not slots:
            return {"slots": [], "spoken_options": [], "count": 0,
                    "message": f"No available slots for {doctor} — {service}"}
        spoken = [f"Option {i}: {_format_slot(s['slot_date'], s['slot_time'])}"
                  for i, s in enumerate(slots, 1)]
        return {"slots": slots, "spoken_options": spoken, "count": len(slots)}

    # Admin path — full filtering including date and time range
    slots = await get_all_slots(
        doctor_id      = doctor_id,
        service_id     = service_id,
        available_only = available_only,
        date_from      = date_from,
        date_to        = date_to,
        time_from      = time_from,
        time_to        = time_to,
    )
    return {"slots": slots, "count": len(slots)}


@router.post("/slots", status_code=201)
async def add_slot(payload: SlotCreate):
    try:
        datetime.strptime(payload.slot_date, "%Y-%m-%d")
        datetime.strptime(payload.slot_time, "%H:%M")
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="slot_date must be YYYY-MM-DD and slot_time must be HH:MM"
        )
    try:
        slot = await create_slot(
            doctor_id  = payload.doctor_id,
            service_id = payload.service_id,
            slot_date  = payload.slot_date,
            slot_time  = payload.slot_time,
        )
        return {"success": True, "slot": slot}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/slots/{slot_id}")
async def get_slot(slot_id: str):
    slot = await get_slot_by_id(slot_id)
    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found")
    return slot


@router.patch("/slots/{slot_id}")
async def patch_slot(slot_id: str, payload: SlotUpdate):
    slot = await get_slot_by_id(slot_id)
    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found")
    updated = await update_slot(slot_id, **payload.model_dump(exclude_none=True))
    return {"success": True, "slot": updated}


@router.delete("/slots/{slot_id}")
async def remove_slot(slot_id: str):
    deleted = await delete_slot(slot_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Slot not found")
    return {"success": True, "deleted_id": slot_id}


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _format_slot(date_str: str, time_str: str) -> str:
    try:
        dt    = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        day   = dt.strftime("%A")
        month = dt.strftime("%B")
        dom   = _ordinal(dt.day)
        hour  = dt.strftime("%I:%M %p").lstrip("0")
        return f"{day} {month} {dom} at {hour}"
    except Exception:
        return f"{date_str} at {time_str}"


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"