"""
RAG Backend — api/clinic_router.py
=====================================
Full CRUD for services, doctors, and slots.

SERVICES
  GET    /clinic/services              — list all services
  POST   /clinic/services              — create service
  GET    /clinic/services/{id}         — get service by ID
  PATCH  /clinic/services/{id}         — update service
  DELETE /clinic/services/{id}         — delete service

DOCTORS
  GET    /clinic/doctors               — list all doctors
  POST   /clinic/doctors               — create doctor
  GET    /clinic/doctors/{id}          — get doctor by ID
  PATCH  /clinic/doctors/{id}          — update doctor
  DELETE /clinic/doctors/{id}          — delete doctor
  GET    /clinic/doctors/{id}/services — services assigned to doctor
  POST   /clinic/doctors/{id}/services — assign service to doctor
  DELETE /clinic/doctors/{id}/services/{service_id} — remove service from doctor

SLOTS
  GET    /clinic/slots                 — list slots (filters: ?doctor_id=&service_id=&available_only=true)
  POST   /clinic/slots                 — create slot
  GET    /clinic/slots/{id}            — get slot by ID
  PATCH  /clinic/slots/{id}            — update slot
  DELETE /clinic/slots/{id}            — delete slot

AGENT ENDPOINTS (unchanged)
  GET    /clinic/services              → names list for agent
  GET    /clinic/doctors?service=name  → doctors for a service
  GET    /clinic/slots?service=&doctor= → available slots for agent
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from database.models import (
    # Services
    get_all_services, get_service_by_id, create_service, update_service, delete_service,
    # Doctors
    get_all_doctors, get_doctor_by_id, create_doctor, update_doctor, delete_doctor,
    get_services_for_doctor, assign_service_to_doctor, remove_service_from_doctor,
    get_doctors_for_service_db,
    # Slots
    get_all_slots, get_slot_by_id, create_slot, update_slot, delete_slot,
    get_available_slots,
)

router = APIRouter(prefix="/clinic", tags=["Clinic"])


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
    active:           Optional[int] = None   # 1 = active, 0 = inactive


class DoctorCreate(BaseModel):
    full_name: str
    title:     Optional[str] = ""
    bio:       Optional[str] = ""


class DoctorUpdate(BaseModel):
    full_name: Optional[str] = None
    title:     Optional[str] = None
    bio:       Optional[str] = None
    active:    Optional[int] = None   # 1 = active, 0 = inactive


class AssignService(BaseModel):
    service_id: str


class SlotCreate(BaseModel):
    doctor_id:  str
    service_id: str
    slot_date:  str   # "YYYY-MM-DD"
    slot_time:  str   # "HH:MM"


class SlotUpdate(BaseModel):
    slot_date:  Optional[str] = None
    slot_time:  Optional[str] = None
    available:  Optional[int] = None   # 1 = available, 0 = booked
    doctor_id:  Optional[str] = None
    service_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# SERVICES
# ─────────────────────────────────────────────────────────────

@router.get("/services")
async def list_services():
    """List all services. Also returns flat names list for the agent."""
    services = await get_all_services()
    return {
        "services": services,
        "names":    [s["name"] for s in services if s.get("active", 1)],
        "count":    len(services),
    }


@router.post("/services", status_code=201)
async def add_service(payload: ServiceCreate):
    """Create a new service."""
    try:
        service = await create_service(
            name             = payload.name,
            description      = payload.description or "",
            duration_minutes = payload.duration_minutes or 60,
        )
        return {"success": True, "service": service}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/services/{service_id}")
async def get_service(service_id: str):
    """Get a single service by ID."""
    service = await get_service_by_id(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    return service


@router.patch("/services/{service_id}")
async def patch_service(service_id: str, payload: ServiceUpdate):
    """Update service fields."""
    service = await get_service_by_id(service_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    updated = await update_service(service_id, **payload.model_dump(exclude_none=True))
    return {"success": True, "service": updated}


@router.delete("/services/{service_id}")
async def remove_service(service_id: str):
    """
    Delete a service. Also removes doctor_services mappings and slots
    linked to this service (CASCADE).
    """
    deleted = await delete_service(service_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Service not found")
    return {"success": True, "deleted_id": service_id}


# ─────────────────────────────────────────────────────────────
# DOCTORS
# ─────────────────────────────────────────────────────────────

@router.get("/doctors")
async def list_doctors(
    service: Optional[str] = Query(default=None, description="Filter by service name (for agent)")
):
    """
    List all doctors.
    Pass ?service=IV Therapy to get doctors for a specific service (agent use).
    """
    if service:
        # Agent path — filter by service name
        doctors = await get_doctors_for_service_db(service)
        return {
            "doctors": doctors,
            "names":   [d["full_name"] for d in doctors],
            "count":   len(doctors),
        }
    doctors = await get_all_doctors()
    return {"doctors": doctors, "count": len(doctors)}


@router.post("/doctors", status_code=201)
async def add_doctor(payload: DoctorCreate):
    """Create a new doctor."""
    try:
        doctor = await create_doctor(
            full_name = payload.full_name,
            title     = payload.title or "",
            bio       = payload.bio or "",
        )
        return {"success": True, "doctor": doctor}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/doctors/{doctor_id}")
async def get_doctor(doctor_id: str):
    """Get a single doctor by ID."""
    doctor = await get_doctor_by_id(doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    return doctor


@router.patch("/doctors/{doctor_id}")
async def patch_doctor(doctor_id: str, payload: DoctorUpdate):
    """Update doctor fields."""
    doctor = await get_doctor_by_id(doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    updated = await update_doctor(doctor_id, **payload.model_dump(exclude_none=True))
    return {"success": True, "doctor": updated}


@router.delete("/doctors/{doctor_id}")
async def remove_doctor(doctor_id: str):
    """
    Delete a doctor. Also removes doctor_services mappings and slots (CASCADE).
    """
    deleted = await delete_doctor(doctor_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Doctor not found")
    return {"success": True, "deleted_id": doctor_id}


@router.get("/doctors/{doctor_id}/services")
async def get_doctor_services(doctor_id: str):
    """Get all services assigned to a doctor."""
    doctor = await get_doctor_by_id(doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    services = await get_services_for_doctor(doctor_id)
    return {"doctor_id": doctor_id, "services": services, "count": len(services)}


@router.post("/doctors/{doctor_id}/services", status_code=201)
async def assign_doctor_service(doctor_id: str, payload: AssignService):
    """Assign a service to a doctor."""
    await assign_service_to_doctor(doctor_id, payload.service_id)
    services = await get_services_for_doctor(doctor_id)
    return {"success": True, "doctor_id": doctor_id, "services": services}


@router.delete("/doctors/{doctor_id}/services/{service_id}")
async def unassign_doctor_service(doctor_id: str, service_id: str):
    """Remove a service assignment from a doctor."""
    removed = await remove_service_from_doctor(doctor_id, service_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Assignment not found")
    return {"success": True, "doctor_id": doctor_id, "removed_service_id": service_id}


# ─────────────────────────────────────────────────────────────
# SLOTS
# ─────────────────────────────────────────────────────────────

@router.get("/slots")
async def list_slots(
    service:        Optional[str]  = Query(default=None, description="Service name (agent use)"),
    doctor:         Optional[str]  = Query(default=None, description="Doctor name (agent use)"),
    doctor_id:      Optional[str]  = Query(default=None, description="Doctor UUID (admin use)"),
    service_id:     Optional[str]  = Query(default=None, description="Service UUID (admin use)"),
    available_only: bool           = Query(default=False),
    limit:          int            = Query(default=6, ge=1, le=100),
):
    """
    List slots.

    Agent use:   ?service=IV Therapy&doctor=Dr. Stefan Koch
    Admin use:   ?doctor_id=doc-2&service_id=svc-10&available_only=true
    """
    # Agent path — filter by name, return spoken options
    if service and doctor:
        slots = await get_available_slots(
            service_name = service,
            doctor_name  = doctor,
            limit        = limit,
        )
        if not slots:
            return {"slots": [], "spoken_options": [], "count": 0,
                    "message": f"No available slots for {doctor} — {service}"}
        spoken = [f"Option {i}: {_format_slot(s['slot_date'], s['slot_time'])}"
                  for i, s in enumerate(slots, 1)]
        return {"slots": slots, "spoken_options": spoken, "count": len(slots)}

    # Admin path — full list with optional filters
    slots = await get_all_slots(
        doctor_id      = doctor_id,
        service_id     = service_id,
        available_only = available_only,
    )
    return {"slots": slots, "count": len(slots)}


@router.post("/slots", status_code=201)
async def add_slot(payload: SlotCreate):
    """
    Create a new appointment slot.
    slot_date format: YYYY-MM-DD
    slot_time format: HH:MM  (e.g. 09:00)
    """
    # Validate date/time format
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
    """Get a single slot by ID."""
    slot = await get_slot_by_id(slot_id)
    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found")
    return slot


@router.patch("/slots/{slot_id}")
async def patch_slot(slot_id: str, payload: SlotUpdate):
    """
    Update slot fields. Use available=0 to manually mark as booked,
    available=1 to reopen a slot.
    """
    slot = await get_slot_by_id(slot_id)
    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found")
    updated = await update_slot(slot_id, **payload.model_dump(exclude_none=True))
    return {"success": True, "slot": updated}


@router.delete("/slots/{slot_id}")
async def remove_slot(slot_id: str):
    """Delete a slot permanently."""
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