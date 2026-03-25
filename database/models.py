"""
RAG Backend — database/models.py
==================================
PostgreSQL schema + full CRUD for:
  - services
  - doctors
  - doctor_services  (many-to-many)
  - slots
  - bookings
"""

import asyncio
import asyncpg
import logging
import os
import re
import time
import uuid
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "")

_pool: asyncpg.Pool | None = None

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Pool management
# ─────────────────────────────────────────────────────────────

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db() first.")
    return _pool


async def init_db():
    global _pool

    if not DATABASE_URL:
        raise EnvironmentError("DATABASE_URL not set. Add it to your environment.")

    url = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    _pool = await asyncpg.create_pool(url, min_size=2, max_size=10)

    async with _pool.acquire() as conn:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS services ("
            "  id               TEXT PRIMARY KEY,"
            "  name             TEXT NOT NULL UNIQUE,"
            "  description      TEXT DEFAULT '',"
            "  duration_minutes INTEGER DEFAULT 60,"
            "  active           INTEGER DEFAULT 1"
            ")"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS doctors ("
            "  id         TEXT PRIMARY KEY,"
            "  full_name  TEXT NOT NULL,"
            "  title      TEXT DEFAULT '',"
            "  department TEXT DEFAULT '',"
            "  bio        TEXT DEFAULT '',"
            "  active     INTEGER DEFAULT 1"
            ")"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS doctor_services ("
            "  doctor_id  TEXT NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,"
            "  service_id TEXT NOT NULL REFERENCES services(id) ON DELETE CASCADE,"
            "  PRIMARY KEY (doctor_id, service_id)"
            ")"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS slots ("
            "  id         TEXT PRIMARY KEY,"
            "  doctor_id  TEXT NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,"
            "  service_id TEXT NOT NULL REFERENCES services(id) ON DELETE CASCADE,"
            "  slot_date  TEXT NOT NULL,"
            "  slot_time  TEXT NOT NULL,"
            "  available  INTEGER DEFAULT 1,"
            "  UNIQUE (doctor_id, service_id, slot_date, slot_time)"
            ")"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS bookings ("
            "  id                  SERIAL PRIMARY KEY,"
            "  confirmation_number TEXT NOT NULL UNIQUE,"
            "  slot_id             TEXT REFERENCES slots(id),"
            "  service_id          TEXT REFERENCES services(id),"
            "  doctor_id           TEXT REFERENCES doctors(id),"
            "  service_name        TEXT NOT NULL,"
            "  doctor_name         TEXT NOT NULL,"
            "  patient_name        TEXT NOT NULL,"
            "  slot_date           TEXT,"
            "  slot_time           TEXT,"
            "  language            TEXT DEFAULT 'en',"
            "  status              TEXT DEFAULT 'confirmed',"
            "  booked_at           TEXT DEFAULT (to_char(NOW(), 'YYYY-MM-DD HH24:MI:SS')),"
            "  session_summary     TEXT"
            ")"
        )

        # Migrations
        await conn.execute(
            "ALTER TABLE doctors ADD COLUMN IF NOT EXISTS department TEXT DEFAULT ''"
        )

        # Backfill older rows: NULL should behave as available (1)
        await conn.execute("UPDATE slots SET available=1 WHERE available IS NULL")

        # Unique constraint on slots
        await conn.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'slots_unique_booking'
                ) THEN
                    ALTER TABLE slots ADD CONSTRAINT slots_unique_booking
                    UNIQUE (doctor_id, service_id, slot_date, slot_time);
                END IF;
            END$$;
        """)

        # Performance indexes for agent lookups (name normalization + join keys).
        # Note: these indexes are safe to re-run.
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_services_name_norm
            ON services (LOWER(BTRIM(name)))
            WHERE active=1;
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_services_name_norm_all
            ON services (LOWER(BTRIM(name)));
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doctors_fullname_norm
            ON doctors (LOWER(BTRIM(full_name)))
            WHERE active=1;
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doctors_fullname_norm_all
            ON doctors (LOWER(BTRIM(full_name)));
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doctors_active_fullname
            ON doctors (full_name)
            WHERE active=1;
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doctor_services_service_doctor
            ON doctor_services (service_id, doctor_id);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_slots_doctor_service_date_available
            ON slots (doctor_id, service_id, slot_date, available);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_slots_doctor_service_date_time_available
            ON slots (doctor_id, service_id, slot_date, slot_time, available);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_slots_slot_date
            ON slots (slot_date);
        """)

    print("✅ PostgreSQL database ready")


async def close_db():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ─────────────────────────────────────────────────────────────
# Name helpers
# ─────────────────────────────────────────────────────────────

_TITLE_PREFIX = re.compile(
    r"^\s*(dr\.?|prof\.?|mr\.?|mrs\.?|ms\.?|dipl\.?[-\s]?\w*\.?)\s+",
    re.IGNORECASE,
)


def _strip_title(name: str) -> str:
    """
    Remove honorific prefix so names like 'Dr. Hina Aslam' match the DB
    column value 'Hina Aslam'.  Safe to call on already-bare names.
    """
    return _TITLE_PREFIX.sub("", (name or "")).strip()


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _generate_confirmation() -> str:
    year  = datetime.now().year
    short = str(uuid.uuid4())[:6].upper()
    return f"FM-{year}-{short}"


# ─────────────────────────────────────────────────────────────
# SERVICES CRUD
# ─────────────────────────────────────────────────────────────

async def get_all_services() -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM services ORDER BY name")
        return [dict(r) for r in rows]


async def get_service_by_id(service_id: str) -> dict | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM services WHERE id=$1", service_id)
        return dict(row) if row else None


async def get_service_by_name(name: str) -> dict | None:
    """
    Exact-match lookup (case-insensitive, whitespace-trimmed).
    Services are not prefixed with titles so exact match is fine here.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM services"
            " WHERE LOWER(BTRIM(name)) = LOWER(BTRIM($1)) AND active=1",
            name,
        )
        return dict(row) if row else None


async def create_service(name: str, description: str = "", duration_minutes: int = 60) -> dict:
    sid  = str(uuid.uuid4())
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO services (id, name, description, duration_minutes) VALUES ($1,$2,$3,$4)",
            sid, name, description, duration_minutes,
        )
    return await get_service_by_id(sid)


async def update_service(service_id: str, **fields) -> dict | None:
    allowed = {"name", "description", "duration_minutes", "active"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return await get_service_by_id(service_id)
    set_parts  = [f"{k}=${i+1}" for i, k in enumerate(updates.keys())]
    set_clause = ", ".join(set_parts)
    values     = list(updates.values()) + [service_id]
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE services SET {set_clause} WHERE id=${len(values)}",
            *values,
        )
    return await get_service_by_id(service_id)


async def delete_service(service_id: str) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM services WHERE id=$1", service_id)
        return result.split()[-1] != "0"


async def get_services_filtered(
    search:    str | None = None,
    doctor_id: str | None = None,
    duration:  int | None = None,
    active:    int | None = None,
) -> list[dict]:
    conditions = []
    params     = []
    i          = 1

    if search:
        conditions.append(f"LOWER(s.name) LIKE LOWER(${i})")
        params.append(f"%{search}%"); i += 1
    if duration is not None:
        conditions.append(f"s.duration_minutes = ${i}")
        params.append(duration); i += 1
    if active is not None:
        conditions.append(f"s.active = ${i}")
        params.append(active); i += 1
    if doctor_id:
        conditions.append(
            f"EXISTS (SELECT 1 FROM doctor_services ds"
            f" WHERE ds.service_id = s.id AND ds.doctor_id = ${i})"
        )
        params.append(doctor_id); i += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"SELECT * FROM services s {where} ORDER BY s.name", *params)
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
# DOCTORS CRUD
# ─────────────────────────────────────────────────────────────

async def get_all_doctors() -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM doctors ORDER BY full_name")
        return [dict(r) for r in rows]


async def get_doctor_by_id(doctor_id: str) -> dict | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM doctors WHERE id=$1", doctor_id)
        return dict(row) if row else None


async def get_doctor_by_name(name: str) -> dict | None:
    """
    Look up a doctor by name.

    Strategy (most-specific to least):
      1. Exact match on the bare name (title stripped).
      2. ILIKE partial match as a fallback so minor spacing/prefix
         differences don't silently produce a NULL doctor_id on bookings.
    """
    bare = _strip_title(name)
    pool = await get_pool()
    async with pool.acquire() as conn:
        # 1. Exact match (preferred — avoids ambiguity when multiple doctors
        #    share a partial name, e.g. "Ali" matching "Dr. Ali Khan" AND
        #    "Dr. Zainab Ali").
        row = await conn.fetchrow(
            "SELECT * FROM doctors"
            " WHERE LOWER(BTRIM(full_name)) = LOWER(BTRIM($1)) AND active=1",
            bare,
        )
        if row:
            return dict(row)

        # 2. Partial ILIKE fallback — catches residual prefix/suffix noise.
        row = await conn.fetchrow(
            "SELECT * FROM doctors"
            " WHERE full_name ILIKE $1 AND active=1"
            " ORDER BY full_name LIMIT 1",
            f"%{bare}%",
        )
        return dict(row) if row else None


async def create_doctor(
    full_name:  str,
    title:      str = "",
    department: str = "",
    bio:        str = "",
) -> dict:
    did  = str(uuid.uuid4())
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO doctors (id, full_name, title, department, bio) VALUES ($1,$2,$3,$4,$5)",
            did, full_name, title, department, bio,
        )
    return await get_doctor_by_id(did)


async def update_doctor(doctor_id: str, **fields) -> dict | None:
    allowed = {"full_name", "title", "department", "bio", "active"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return await get_doctor_by_id(doctor_id)
    set_parts  = [f"{k}=${i+1}" for i, k in enumerate(updates.keys())]
    set_clause = ", ".join(set_parts)
    values     = list(updates.values()) + [doctor_id]
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE doctors SET {set_clause} WHERE id=${len(values)}",
            *values,
        )
    return await get_doctor_by_id(doctor_id)


async def delete_doctor(doctor_id: str) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM doctors WHERE id=$1", doctor_id)
        return result.split()[-1] != "0"


async def get_services_for_doctor(doctor_id: str) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT s.* FROM services s"
            " JOIN doctor_services ds ON ds.service_id = s.id"
            " WHERE ds.doctor_id = $1 ORDER BY s.name",
            doctor_id,
        )
        return [dict(r) for r in rows]


async def assign_service_to_doctor(doctor_id: str, service_id: str) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO doctor_services (doctor_id, service_id) VALUES ($1,$2)"
            " ON CONFLICT DO NOTHING",
            doctor_id, service_id,
        )
    return True


async def remove_service_from_doctor(doctor_id: str, service_id: str) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM doctor_services WHERE doctor_id=$1 AND service_id=$2",
            doctor_id, service_id,
        )
        return result.split()[-1] != "0"


async def get_doctors_for_service_db(service_name: str) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        t0 = time.perf_counter()
        rows = await conn.fetch(
            "SELECT d.id, d.full_name, d.title, d.department, d.bio"
            " FROM doctors d"
            " JOIN doctor_services ds ON ds.doctor_id = d.id"
            " JOIN services s ON s.id = ds.service_id"
            " WHERE LOWER(BTRIM(s.name)) = LOWER(BTRIM($1)) AND d.active=1 AND s.active=1"
            " ORDER BY d.full_name",
            service_name,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        msg = (
            f"[db][get_doctors_for_service_db] service={service_name!r} "
            f"returned={len(rows)} in {dt_ms:.1f}ms"
        )
        print(msg, flush=True)
        log.info(msg)
        return [dict(r) for r in rows]


async def get_doctors_filtered(
    search:     str | None = None,
    department: str | None = None,
    service_id: str | None = None,
    active:     int | None = None,
) -> list[dict]:
    conditions = []
    params     = []
    i          = 1

    if search:
        conditions.append(f"LOWER(d.full_name) LIKE LOWER(${i})")
        params.append(f"%{search}%"); i += 1
    if department:
        conditions.append(f"LOWER(d.department) = LOWER(${i})")
        params.append(department); i += 1
    if active is not None:
        conditions.append(f"d.active = ${i}")
        params.append(active); i += 1
    if service_id:
        conditions.append(
            f"EXISTS (SELECT 1 FROM doctor_services ds"
            f" WHERE ds.doctor_id = d.id AND ds.service_id = ${i})"
        )
        params.append(service_id); i += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"SELECT * FROM doctors d {where} ORDER BY d.full_name", *params)
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
# SLOTS CRUD
# ─────────────────────────────────────────────────────────────

async def get_all_slots(
    doctor_id:      str  | None = None,
    service_id:     str  | None = None,
    available_only: bool        = False,
    date_from:      str  | None = None,
    date_to:        str  | None = None,
    time_from:      str  | None = None,
    time_to:        str  | None = None,
) -> list[dict]:
    conditions = []
    params     = []
    i          = 1

    if doctor_id:
        conditions.append(f"sl.doctor_id=${i}"); params.append(doctor_id); i += 1
    if service_id:
        conditions.append(f"sl.service_id=${i}"); params.append(service_id); i += 1
    if available_only:
        conditions.append("sl.available=1")
    if date_from:
        conditions.append(f"sl.slot_date >= ${i}"); params.append(date_from); i += 1
    if date_to:
        conditions.append(f"sl.slot_date <= ${i}"); params.append(date_to); i += 1
    if time_from:
        conditions.append(f"sl.slot_time >= ${i}"); params.append(time_from); i += 1
    if time_to:
        conditions.append(f"sl.slot_time <= ${i}"); params.append(time_to); i += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT sl.*, d.full_name AS doctor_name, s.name AS service_name"
            f" FROM slots sl"
            f" JOIN doctors d ON d.id = sl.doctor_id"
            f" JOIN services s ON s.id = sl.service_id"
            f" {where}"
            f" ORDER BY sl.slot_date, sl.slot_time",
            *params,
        )
        return [dict(r) for r in rows]


async def get_slot_by_id(slot_id: str) -> dict | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM slots WHERE id=$1", slot_id)
        return dict(row) if row else None


async def create_slot(doctor_id: str, service_id: str, slot_date: str, slot_time: str) -> dict:
    sid  = str(uuid.uuid4())
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            await conn.execute(
                "INSERT INTO slots (id, doctor_id, service_id, slot_date, slot_time, available)"
                " VALUES ($1,$2,$3,$4,$5,1)",
                sid, doctor_id, service_id, slot_date, slot_time,
            )
        except asyncpg.UniqueViolationError:
            raise ValueError(
                f"A slot already exists for this doctor on {slot_date} at {slot_time}"
            )
    return await get_slot_by_id(sid)


async def update_slot(slot_id: str, **fields) -> dict | None:
    allowed = {"slot_date", "slot_time", "available", "doctor_id", "service_id"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return await get_slot_by_id(slot_id)
    set_parts  = [f"{k}=${i+1}" for i, k in enumerate(updates.keys())]
    set_clause = ", ".join(set_parts)
    values     = list(updates.values()) + [slot_id]
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            await conn.execute(
                f"UPDATE slots SET {set_clause} WHERE id=${len(values)}",
                *values,
            )
        except asyncpg.UniqueViolationError:
            raise ValueError(
                "A slot already exists for this doctor at the new date/time"
            )
    return await get_slot_by_id(slot_id)


async def delete_slot(slot_id: str) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM slots WHERE id=$1", slot_id)
        return result.split()[-1] != "0"


async def mark_slot_unavailable(slot_id: str, conn: asyncpg.Connection | None = None) -> bool:
    """
    Mark a slot as unavailable (available=0).
    Returns True if we successfully changed it from available → unavailable.
    Returns False if slot doesn't exist or was already unavailable.
    """
    pool = await get_pool()
    if conn is None:
        async with pool.acquire() as _conn:
            return await mark_slot_unavailable(slot_id, conn=_conn)

    result = await conn.execute(
        "UPDATE slots SET available=0"
        " WHERE id=$1 AND (available IS NULL OR available=1)",
        slot_id,
    )
    try:
        changed = int(result.split()[-1])
    except Exception:
        changed = 0
    return changed == 1


async def get_available_slots(
    service_name: str,
    doctor_name: str,
    limit: int = 6,
) -> list[dict]:
    """
    Return upcoming available slots for a given doctor and service.

    Matching strategy (most-specific to least):
      1. Exact match on stripped bare name  → no ambiguity risk.
      2. ILIKE partial match as fallback    → catches any residual
         prefix/suffix noise ("Dr." etc.) that slipped past the
         client/router stripping layers.
    """
    bare_name = _strip_title(doctor_name)
    today     = datetime.now().date().strftime("%Y-%m-%d")
    pool      = await get_pool()

    base_select = (
        "SELECT sl.id, sl.slot_date, sl.slot_time,"
        "       d.full_name AS doctor_name, s.name AS service_name,"
        "       s.duration_minutes"
        " FROM slots sl"
        " JOIN doctors d ON d.id = sl.doctor_id"
        " JOIN services s ON s.id = sl.service_id"
        " WHERE (sl.available IS NULL OR sl.available=1)"
        "   AND sl.slot_date >= $3"
        "   AND LOWER(BTRIM(s.name)) = LOWER(BTRIM($2))"
    )

    async with pool.acquire() as conn:
        # 1. Exact match on bare name
        t_exact0 = time.perf_counter()
        rows = await conn.fetch(
            base_select
            + "   AND LOWER(BTRIM(d.full_name)) = LOWER(BTRIM($1))"
              " ORDER BY sl.slot_date, sl.slot_time LIMIT $4",
            bare_name, service_name, today, limit,
        )
        dt_exact_ms = (time.perf_counter() - t_exact0) * 1000.0
        msg = (
            f"[db][get_available_slots:exact] service={service_name!r} "
            f"doctor_bare={bare_name!r} returned={len(rows)} in {dt_exact_ms:.1f}ms"
        )
        print(msg, flush=True)
        log.info(msg)

        if not rows:
            # 2. ILIKE partial fallback — handles edge-cases where the DB
            #    value has extra whitespace or a Unicode variant of the name.
            t_ilike0 = time.perf_counter()
            rows = await conn.fetch(
                base_select
                + "   AND d.full_name ILIKE $1"
                  " ORDER BY sl.slot_date, sl.slot_time LIMIT $4",
                f"%{bare_name}%", service_name, today, limit,
            )
            dt_ilike_ms = (time.perf_counter() - t_ilike0) * 1000.0
            msg = (
                f"[db][get_available_slots:ilike] service={service_name!r} "
                f"doctor_bare={bare_name!r} returned={len(rows)} in {dt_ilike_ms:.1f}ms"
            )
            print(msg, flush=True)
            log.info(msg)

        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
# BOOKINGS CRUD
# ─────────────────────────────────────────────────────────────

async def get_all_bookings(status: str = None) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        if status:
            rows = await conn.fetch(
                "SELECT * FROM bookings WHERE status=$1 ORDER BY booked_at DESC", status
            )
        else:
            rows = await conn.fetch("SELECT * FROM bookings ORDER BY booked_at DESC")
        return [dict(r) for r in rows]


async def get_bookings_filtered(
    status:       str | None = None,
    doctor_name:  str | None = None,
    service_name: str | None = None,
    date_from:    str | None = None,
    date_to:      str | None = None,
    time_from:    str | None = None,
    time_to:      str | None = None,
    search:       str | None = None,
) -> list[dict]:
    conditions = []
    params     = []
    i          = 1

    if status:
        conditions.append(f"status = ${i}")
        params.append(status); i += 1
    if doctor_name:
        conditions.append(f"LOWER(doctor_name) LIKE LOWER(${i})")
        params.append(f"%{doctor_name}%"); i += 1
    if service_name:
        conditions.append(f"LOWER(service_name) LIKE LOWER(${i})")
        params.append(f"%{service_name}%"); i += 1
    if date_from:
        conditions.append(f"slot_date >= ${i}")
        params.append(date_from); i += 1
    if date_to:
        conditions.append(f"slot_date <= ${i}")
        params.append(date_to); i += 1
    if time_from:
        conditions.append(f"slot_time >= ${i}")
        params.append(time_from); i += 1
    if time_to:
        conditions.append(f"slot_time <= ${i}")
        params.append(time_to); i += 1
    if search:
        conditions.append(
            f"(LOWER(patient_name) LIKE LOWER(${i})"
            f" OR LOWER(confirmation_number) LIKE LOWER(${i}))"
        )
        params.append(f"%{search}%"); i += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM bookings {where} ORDER BY booked_at DESC",
            *params,
        )
        return [dict(r) for r in rows]


async def get_booking_by_id(booking_id: int | str) -> dict | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            int_id = int(booking_id)
            row    = await conn.fetchrow("SELECT * FROM bookings WHERE id=$1", int_id)
        except (ValueError, TypeError):
            row = await conn.fetchrow(
                "SELECT * FROM bookings WHERE confirmation_number=$1", booking_id
            )
        return dict(row) if row else None


async def save_booking(
    service_name: str,
    doctor_name: str,
    patient_name: str,
    slot_id: str | None = None,
    language: str = "en",
    session_summary: str | None = None,
) -> dict:
    t_total0 = time.perf_counter()
    confirmation = _generate_confirmation()

    # Strip title before resolving IDs — DB stores bare names.
    bare_doctor = _strip_title(doctor_name)

    # If slot_id is provided, derive doctor_id/service_id directly from the locked
    # slot row. This avoids slow name-based lookups and is faster/consistent.
    service_id: str | None = None
    doctor_id: str | None = None
    slot_date = None
    slot_time = None
    dt_resolve_ms = 0.0
    new_id = None

    pool = await get_pool()
    t_acquire0 = time.perf_counter()
    async with pool.acquire() as conn:
        dt_acquire_ms = (time.perf_counter() - t_acquire0) * 1000.0
        async with conn.transaction():
            t_tx0 = time.perf_counter()
            dt_slot_ms = 0.0
            dt_insert_ms = 0.0
            if slot_id:
                # Consume slot + insert booking in ONE statement to avoid
                # a second DB round-trip.
                t_slot0 = time.perf_counter()
                row = await conn.fetchrow(
                    """
                    WITH consumed AS (
                      UPDATE slots
                         SET available=0
                       WHERE id=$1
                         AND (available IS NULL OR available=1)
                       RETURNING slot_date, slot_time, doctor_id, service_id
                    )
                    INSERT INTO bookings
                      (confirmation_number, slot_id, service_id, doctor_id,
                       service_name, doctor_name, patient_name,
                       slot_date, slot_time, language, session_summary)
                    SELECT
                      $2, $1, c.service_id, c.doctor_id,
                      $3, $4, $5,
                      c.slot_date, c.slot_time, $6, $7
                    FROM consumed c
                    RETURNING id, slot_date, slot_time
                    """,
                    slot_id,           # $1
                    confirmation,     # $2
                    service_name,     # $3
                    bare_doctor,      # $4
                    patient_name,     # $5
                    language,         # $6
                    session_summary,  # $7
                )

                if not row:
                    raise ValueError(
                        "Selected slot is no longer available. Please choose another slot."
                    )

                slot_date = row["slot_date"]
                slot_time = row["slot_time"]
                new_id = row["id"] if row else None

                dt_slot_ms = (time.perf_counter() - t_slot0) * 1000.0
                print(
                    f"[save_booking] slot consume+insert {dt_slot_ms:.1f}ms (slot_id={slot_id!r})",
                    flush=True,
                )

            # Fallback: if slot_id missing, resolve by name + insert.
            if slot_id is None and (not doctor_id or not service_id):
                t_resolve0 = time.perf_counter()
                service, doctor = await asyncio.gather(
                    get_service_by_name(service_name),
                    get_doctor_by_name(bare_doctor),
                )
                dt_resolve_ms = (time.perf_counter() - t_resolve0) * 1000.0
                service_id = service["id"] if service else None
                doctor_id = doctor["id"] if doctor else None

                if not doctor_id:
                    log.warning(
                        "[save_booking] Could not resolve doctor_id for name=%r (bare=%r)",
                        doctor_name,
                        bare_doctor,
                    )

            dt_tx_ms = (time.perf_counter() - t_tx0) * 1000.0
            if slot_id is None:
                t_insert0 = time.perf_counter()
                row = await conn.fetchrow(
                    "INSERT INTO bookings"
                    "  (confirmation_number, slot_id, service_id, doctor_id,"
                    "   service_name, doctor_name, patient_name,"
                    "   slot_date, slot_time, language, session_summary)"
                    " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)"
                    " RETURNING id",
                    confirmation, slot_id, service_id, doctor_id,
                    service_name, bare_doctor, patient_name,
                    slot_date, slot_time, language, session_summary,
                )
                dt_insert_ms = (time.perf_counter() - t_insert0) * 1000.0
                new_id = row["id"] if row else None

    dt_total_ms = (time.perf_counter() - t_total0) * 1000.0
    print(
        f"[save_booking] completed total {dt_total_ms:.1f}ms "
        f"(pool_acquire {dt_acquire_ms:.1f}ms, tx {dt_tx_ms:.1f}ms, insert {dt_insert_ms:.1f}ms, "
        f"slot_consume {dt_slot_ms:.1f}ms, resolve_ids {dt_resolve_ms:.1f}ms)",
        flush=True,
    )
    return {
        "id":                  new_id,
        "confirmation_number": confirmation,
        "service_name":        service_name,
        "doctor_name":         bare_doctor,
        "patient_name":        patient_name,
        "slot_date":           slot_date,
        "slot_time":           slot_time,
        "status":              "confirmed",
    }


async def update_booking(booking_id: int | str, **fields) -> dict | None:
    allowed = {"status", "patient_name", "language", "session_summary", "slot_date", "slot_time"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return await get_booking_by_id(booking_id)
    set_parts  = [f"{k}=${i+1}" for i, k in enumerate(updates.keys())]
    set_clause = ", ".join(set_parts)
    values     = list(updates.values())
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            int_id = int(booking_id)
            await conn.execute(
                f"UPDATE bookings SET {set_clause} WHERE id=${len(values)+1}",
                *values, int_id,
            )
        except (ValueError, TypeError):
            await conn.execute(
                f"UPDATE bookings SET {set_clause} WHERE confirmation_number=${len(values)+1}",
                *values, booking_id,
            )
    return await get_booking_by_id(booking_id)


async def delete_booking(booking_id: int | str) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            if isinstance(booking_id, int):
                booking = await conn.fetchrow(
                    "SELECT id, slot_id FROM bookings WHERE id=$1", booking_id
                )
            else:
                try:
                    int_id  = int(booking_id)
                    booking = await conn.fetchrow(
                        "SELECT id, slot_id FROM bookings WHERE id=$1", int_id
                    )
                except (ValueError, TypeError):
                    booking = await conn.fetchrow(
                        "SELECT id, slot_id FROM bookings WHERE confirmation_number=$1",
                        booking_id,
                    )

            if not booking:
                return False

            slot_id = booking["slot_id"]
            deleted = await conn.execute("DELETE FROM bookings WHERE id=$1", booking["id"])
            was_deleted = deleted.split()[-1] != "0"
            if not was_deleted:
                return False

            if slot_id:
                still_referenced = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM bookings WHERE slot_id=$1)",
                    slot_id,
                )
                if not still_referenced:
                    await conn.execute("DELETE FROM slots WHERE id=$1", slot_id)

            return True


async def cancel_booking_db(booking_id: int | str) -> dict | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            booking = await get_booking_by_id(booking_id)
            if not booking:
                return None

            updated = await update_booking(booking_id, status="cancelled")
            slot_id = booking.get("slot_id")
            if slot_id:
                still_used = await conn.fetchval(
                    "SELECT EXISTS("
                    "  SELECT 1 FROM bookings"
                    "  WHERE slot_id=$1 AND status='confirmed'"
                    ")",
                    slot_id,
                )
                if not still_used:
                    await conn.execute("UPDATE slots SET available=1 WHERE id=$1", slot_id)

            return updated