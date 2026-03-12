"""
RAG Backend — database/models.py
==================================
SQLite schema + full CRUD for:
  - services
  - doctors
  - doctor_services  (many-to-many)
  - slots
  - bookings
"""

import aiosqlite
import os
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "data/functiomed.db")


# ─────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────

async def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")

        await db.execute("""
            CREATE TABLE IF NOT EXISTS services (
                id               TEXT PRIMARY KEY,
                name             TEXT NOT NULL UNIQUE,
                description      TEXT,
                duration_minutes INTEGER DEFAULT 60,
                active           INTEGER DEFAULT 1
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id        TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                title     TEXT DEFAULT '',
                bio       TEXT,
                active    INTEGER DEFAULT 1
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS doctor_services (
                doctor_id  TEXT NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
                service_id TEXT NOT NULL REFERENCES services(id) ON DELETE CASCADE,
                PRIMARY KEY (doctor_id, service_id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS slots (
                id         TEXT PRIMARY KEY,
                doctor_id  TEXT NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
                service_id TEXT NOT NULL REFERENCES services(id) ON DELETE CASCADE,
                slot_date  TEXT NOT NULL,
                slot_time  TEXT NOT NULL,
                available  INTEGER DEFAULT 1
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                confirmation_number TEXT NOT NULL UNIQUE,
                slot_id             TEXT REFERENCES slots(id),
                service_id          TEXT REFERENCES services(id),
                doctor_id           TEXT REFERENCES doctors(id),
                service_name        TEXT NOT NULL,
                doctor_name         TEXT NOT NULL,
                patient_name        TEXT NOT NULL,
                slot_date           TEXT,
                slot_time           TEXT,
                language            TEXT DEFAULT 'en',
                status              TEXT DEFAULT 'confirmed',
                booked_at           TEXT DEFAULT (datetime('now')),
                session_summary     TEXT
            )
        """)
        await db.commit()
    print(f"✅ Database ready: {DB_PATH}")


# ─────────────────────────────────────────────────────────────
# SERVICES CRUD
# ─────────────────────────────────────────────────────────────

async def get_all_services() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM services ORDER BY name")
        return [dict(r) for r in await cur.fetchall()]


async def get_service_by_id(service_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM services WHERE id=?", (service_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def get_service_by_name(name: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM services WHERE LOWER(name)=LOWER(?) AND active=1", (name,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def create_service(name: str, description: str = "", duration_minutes: int = 60) -> dict:
    sid = str(uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO services (id, name, description, duration_minutes) VALUES (?,?,?,?)",
            (sid, name, description, duration_minutes),
        )
        await db.commit()
    return await get_service_by_id(sid)


async def update_service(service_id: str, **fields) -> dict | None:
    allowed = {"name", "description", "duration_minutes", "active"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return await get_service_by_id(service_id)
    set_clause = ", ".join(f"{k}=?" for k in updates)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE services SET {set_clause} WHERE id=?",
            (*updates.values(), service_id),
        )
        await db.commit()
    return await get_service_by_id(service_id)


async def delete_service(service_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        cur = await db.execute("DELETE FROM services WHERE id=?", (service_id,))
        await db.commit()
        return cur.rowcount > 0


# ─────────────────────────────────────────────────────────────
# DOCTORS CRUD
# ─────────────────────────────────────────────────────────────

async def get_all_doctors() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM doctors ORDER BY full_name")
        return [dict(r) for r in await cur.fetchall()]


async def get_doctor_by_id(doctor_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM doctors WHERE id=?", (doctor_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def get_doctor_by_name(name: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM doctors WHERE LOWER(full_name)=LOWER(?) AND active=1", (name,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def create_doctor(full_name: str, title: str = "", bio: str = "") -> dict:
    did = str(uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO doctors (id, full_name, title, bio) VALUES (?,?,?,?)",
            (did, full_name, title, bio),
        )
        await db.commit()
    return await get_doctor_by_id(did)


async def update_doctor(doctor_id: str, **fields) -> dict | None:
    allowed = {"full_name", "title", "bio", "active"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return await get_doctor_by_id(doctor_id)
    set_clause = ", ".join(f"{k}=?" for k in updates)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE doctors SET {set_clause} WHERE id=?",
            (*updates.values(), doctor_id),
        )
        await db.commit()
    return await get_doctor_by_id(doctor_id)


async def delete_doctor(doctor_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        cur = await db.execute("DELETE FROM doctors WHERE id=?", (doctor_id,))
        await db.commit()
        return cur.rowcount > 0


async def get_services_for_doctor(doctor_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """
            SELECT s.* FROM services s
            JOIN doctor_services ds ON ds.service_id = s.id
            WHERE ds.doctor_id = ?
            ORDER BY s.name
            """,
            (doctor_id,),
        )
        return [dict(r) for r in await cur.fetchall()]


async def assign_service_to_doctor(doctor_id: str, service_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute(
            "INSERT OR IGNORE INTO doctor_services (doctor_id, service_id) VALUES (?,?)",
            (doctor_id, service_id),
        )
        await db.commit()
    return True


async def remove_service_from_doctor(doctor_id: str, service_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "DELETE FROM doctor_services WHERE doctor_id=? AND service_id=?",
            (doctor_id, service_id),
        )
        await db.commit()
        return cur.rowcount > 0


async def get_doctors_for_service_db(service_name: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """
            SELECT d.id, d.full_name, d.title, d.bio
            FROM doctors d
            JOIN doctor_services ds ON ds.doctor_id = d.id
            JOIN services s ON s.id = ds.service_id
            WHERE LOWER(s.name) = LOWER(?) AND d.active=1 AND s.active=1
            ORDER BY d.full_name
            """,
            (service_name,),
        )
        return [dict(r) for r in await cur.fetchall()]


# ─────────────────────────────────────────────────────────────
# SLOTS CRUD
# ─────────────────────────────────────────────────────────────

async def get_all_slots(doctor_id: str = None, service_id: str = None, available_only: bool = False) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        conditions = []
        params     = []
        if doctor_id:
            conditions.append("sl.doctor_id=?")
            params.append(doctor_id)
        if service_id:
            conditions.append("sl.service_id=?")
            params.append(service_id)
        if available_only:
            conditions.append("sl.available=1")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        cur = await db.execute(
            f"""
            SELECT sl.*, d.full_name AS doctor_name, s.name AS service_name
            FROM slots sl
            JOIN doctors d  ON d.id = sl.doctor_id
            JOIN services s ON s.id = sl.service_id
            {where}
            ORDER BY sl.slot_date, sl.slot_time
            """,
            params,
        )
        return [dict(r) for r in await cur.fetchall()]


async def get_slot_by_id(slot_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM slots WHERE id=?", (slot_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def create_slot(doctor_id: str, service_id: str, slot_date: str, slot_time: str) -> dict:
    sid = str(uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute(
            "INSERT INTO slots (id, doctor_id, service_id, slot_date, slot_time) VALUES (?,?,?,?,?)",
            (sid, doctor_id, service_id, slot_date, slot_time),
        )
        await db.commit()
    return await get_slot_by_id(sid)


async def update_slot(slot_id: str, **fields) -> dict | None:
    allowed = {"slot_date", "slot_time", "available", "doctor_id", "service_id"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return await get_slot_by_id(slot_id)
    set_clause = ", ".join(f"{k}=?" for k in updates)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE slots SET {set_clause} WHERE id=?",
            (*updates.values(), slot_id),
        )
        await db.commit()
    return await get_slot_by_id(slot_id)


async def delete_slot(slot_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("DELETE FROM slots WHERE id=?", (slot_id,))
        await db.commit()
        return cur.rowcount > 0


async def mark_slot_unavailable(slot_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE slots SET available=0 WHERE id=?", (slot_id,))
        await db.commit()


async def get_available_slots(service_name: str, doctor_name: str, limit: int = 6) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        today = datetime.now().date().strftime("%Y-%m-%d")
        cur = await db.execute(
            """
            SELECT sl.id, sl.slot_date, sl.slot_time,
                   d.full_name AS doctor_name, s.name AS service_name, s.duration_minutes
            FROM slots sl
            JOIN doctors  d ON d.id = sl.doctor_id
            JOIN services s ON s.id = sl.service_id
            WHERE LOWER(d.full_name) = LOWER(?)
              AND LOWER(s.name)      = LOWER(?)
              AND sl.available       = 1
              AND sl.slot_date       >= ?
            ORDER BY sl.slot_date, sl.slot_time
            LIMIT ?
            """,
            (doctor_name, service_name, today, limit),
        )
        return [dict(r) for r in await cur.fetchall()]


# ─────────────────────────────────────────────────────────────
# BOOKINGS CRUD
# ─────────────────────────────────────────────────────────────

async def get_all_bookings(status: str = None) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if status:
            cur = await db.execute(
                "SELECT * FROM bookings WHERE status=? ORDER BY booked_at DESC", (status,)
            )
        else:
            cur = await db.execute("SELECT * FROM bookings ORDER BY booked_at DESC")
        return [dict(r) for r in await cur.fetchall()]


async def get_booking_by_id(booking_id: int | str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        try:
            int_id = int(booking_id)
            cur = await db.execute("SELECT * FROM bookings WHERE id=?", (int_id,))
        except (ValueError, TypeError):
            cur = await db.execute(
                "SELECT * FROM bookings WHERE confirmation_number=?", (booking_id,)
            )
        row = await cur.fetchone()
        return dict(row) if row else None


async def save_booking(
    service_name: str,
    doctor_name: str,
    patient_name: str,
    slot_id: str | None = None,
    language: str = "en",
    session_summary: str | None = None,
) -> dict:
    confirmation = _generate_confirmation()

    service    = await get_service_by_name(service_name)
    doctor     = await get_doctor_by_name(doctor_name)
    service_id = service["id"] if service else None
    doctor_id  = doctor["id"]  if doctor  else None

    slot_date = None
    slot_time = None

    if slot_id:
        slot = await get_slot_by_id(slot_id)
        if slot:
            slot_date = slot["slot_date"]
            slot_time = slot["slot_time"]
            await mark_slot_unavailable(slot_id)
        else:
            print(f"⚠️  slot_id {slot_id} not found, booking without slot reference")
            slot_id = None

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(
            """
            INSERT INTO bookings
              (confirmation_number, slot_id, service_id, doctor_id,
               service_name, doctor_name, patient_name,
               slot_date, slot_time, language, session_summary)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (confirmation, slot_id, service_id, doctor_id,
             service_name, doctor_name, patient_name,
             slot_date, slot_time, language, session_summary),
        )
        await db.commit()
        row = await db.execute("SELECT last_insert_rowid()")
        result = await row.fetchone()
        new_id = result[0] if result else None

    return {
        "id":                  new_id,
        "confirmation_number": confirmation,
        "service_name":        service_name,
        "doctor_name":         doctor_name,
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
    set_clause = ", ".join(f"{k}=?" for k in updates)
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            int_id = int(booking_id)
            await db.execute(
                f"UPDATE bookings SET {set_clause} WHERE id=?",
                (*updates.values(), int_id),
            )
        except (ValueError, TypeError):
            await db.execute(
                f"UPDATE bookings SET {set_clause} WHERE confirmation_number=?",
                (*updates.values(), booking_id),
            )
        await db.commit()
    return await get_booking_by_id(booking_id)


async def delete_booking(booking_id: int | str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            int_id = int(booking_id)
            cur = await db.execute("DELETE FROM bookings WHERE id=?", (int_id,))
        except (ValueError, TypeError):
            cur = await db.execute(
                "DELETE FROM bookings WHERE confirmation_number=?", (booking_id,)
            )
        await db.commit()
        return cur.rowcount > 0


async def cancel_booking_db(booking_id: int | str) -> dict | None:
    return await update_booking(booking_id, status="cancelled")


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _generate_confirmation() -> str:
    year  = datetime.now().year
    short = str(uuid.uuid4())[:6].upper()
    return f"FM-{year}-{short}"