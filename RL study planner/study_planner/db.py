from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from werkzeug.security import check_password_hash, generate_password_hash


def _connect(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS completions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                plan_id INTEGER,
                subject TEXT NOT NULL,
                time_slot TEXT NOT NULL,
                completed INTEGER NOT NULL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(plan_id) REFERENCES plans(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS progress_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                plan_id INTEGER,
                subject TEXT NOT NULL,
                time_slot TEXT NOT NULL,
                status TEXT NOT NULL,
                scheduled_date TEXT,
                note TEXT,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(plan_id) REFERENCES plans(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS notification_prefs (
                user_id INTEGER PRIMARY KEY,
                in_app_enabled INTEGER NOT NULL DEFAULT 1,
                daily_reminder_enabled INTEGER NOT NULL DEFAULT 1,
                streak_alert_enabled INTEGER NOT NULL DEFAULT 1,
                missed_slot_alert_enabled INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                message TEXT NOT NULL,
                is_read INTEGER NOT NULL DEFAULT 0,
                meta_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.commit()


def create_user(db_path: str, username: str, password: str) -> tuple[bool, str]:
    cleaned_username = username.strip().lower()
    if len(cleaned_username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    with _connect(db_path) as connection:
        existing = connection.execute("SELECT id FROM users WHERE username = ?", (cleaned_username,)).fetchone()
        if existing is not None:
            return False, "Username already exists."

        connection.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (cleaned_username, generate_password_hash(password), datetime.utcnow().isoformat()),
        )
        connection.commit()

    return True, "Account created successfully."


def authenticate_user(db_path: str, username: str, password: str) -> Optional[Dict[str, Any]]:
    cleaned_username = username.strip().lower()
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (cleaned_username,),
        ).fetchone()
        if row is None:
            return None

        if not check_password_hash(row["password_hash"], password):
            return None

        return {
            "id": row["id"],
            "username": row["username"],
            "created_at": row["created_at"],
        }


def get_user_by_id(db_path: str, user_id: int) -> Optional[Dict[str, Any]]:
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "username": row["username"], "created_at": row["created_at"]}


def save_plan(db_path: str, user_id: int, payload: Dict[str, Any]) -> int:
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO plans (user_id, created_at, payload_json) VALUES (?, ?, ?)",
            (user_id, datetime.utcnow().isoformat(), json.dumps(payload)),
        )
        connection.commit()
        return int(cursor.lastrowid)


def list_user_plans(db_path: str, user_id: int) -> List[Dict[str, Any]]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id, created_at, payload_json FROM plans WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()

    plans: List[Dict[str, Any]] = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        plans.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "payload": payload,
            }
        )
    return plans


def get_user_plan(db_path: str, user_id: int, plan_id: int) -> Optional[Dict[str, Any]]:
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT id, created_at, payload_json FROM plans WHERE user_id = ? AND id = ?",
            (user_id, plan_id),
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "payload": json.loads(row["payload_json"]),
    }


def record_completion(
    db_path: str, user_id: int, plan_id: Optional[int], subject: str, time_slot: str, completed: bool
) -> None:
    """Record whether a subject was completed in a time slot."""
    with _connect(db_path) as connection:
        connection.execute(
            "INSERT INTO completions (user_id, plan_id, subject, time_slot, completed, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, plan_id, subject, time_slot, 1 if completed else 0, datetime.utcnow().isoformat()),
        )
        connection.commit()


def record_progress_event(
    db_path: str,
    user_id: int,
    plan_id: Optional[int],
    subject: str,
    time_slot: str,
    status: str,
    scheduled_date: Optional[str] = None,
    note: str = "",
) -> None:
    """Record detailed progress event for real-time tracking."""
    normalized_status = status.strip().lower()
    if normalized_status not in {"completed", "missed", "rescheduled"}:
        normalized_status = "completed"

    with _connect(db_path) as connection:
        connection.execute(
            "INSERT INTO progress_events (user_id, plan_id, subject, time_slot, status, scheduled_date, note, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                plan_id,
                subject,
                time_slot,
                normalized_status,
                scheduled_date,
                note.strip() if note else "",
                datetime.utcnow().isoformat(),
            ),
        )
        connection.commit()


def ensure_notification_prefs(db_path: str, user_id: int) -> Dict[str, Any]:
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT user_id, in_app_enabled, daily_reminder_enabled, streak_alert_enabled, missed_slot_alert_enabled, updated_at FROM notification_prefs WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if row is None:
            now = datetime.utcnow().isoformat()
            connection.execute(
                "INSERT INTO notification_prefs (user_id, in_app_enabled, daily_reminder_enabled, streak_alert_enabled, missed_slot_alert_enabled, updated_at) VALUES (?, 1, 1, 1, 1, ?)",
                (user_id, now),
            )
            connection.commit()
            return {
                "user_id": user_id,
                "in_app_enabled": True,
                "daily_reminder_enabled": True,
                "streak_alert_enabled": True,
                "missed_slot_alert_enabled": True,
            }

        return {
            "user_id": row["user_id"],
            "in_app_enabled": bool(row["in_app_enabled"]),
            "daily_reminder_enabled": bool(row["daily_reminder_enabled"]),
            "streak_alert_enabled": bool(row["streak_alert_enabled"]),
            "missed_slot_alert_enabled": bool(row["missed_slot_alert_enabled"]),
        }


def update_notification_prefs(db_path: str, user_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    current = ensure_notification_prefs(db_path, user_id)
    merged = {
        "in_app_enabled": bool(updates.get("in_app_enabled", current["in_app_enabled"])),
        "daily_reminder_enabled": bool(updates.get("daily_reminder_enabled", current["daily_reminder_enabled"])),
        "streak_alert_enabled": bool(updates.get("streak_alert_enabled", current["streak_alert_enabled"])),
        "missed_slot_alert_enabled": bool(updates.get("missed_slot_alert_enabled", current["missed_slot_alert_enabled"])),
    }

    with _connect(db_path) as connection:
        connection.execute(
            "UPDATE notification_prefs SET in_app_enabled = ?, daily_reminder_enabled = ?, streak_alert_enabled = ?, missed_slot_alert_enabled = ?, updated_at = ? WHERE user_id = ?",
            (
                1 if merged["in_app_enabled"] else 0,
                1 if merged["daily_reminder_enabled"] else 0,
                1 if merged["streak_alert_enabled"] else 0,
                1 if merged["missed_slot_alert_enabled"] else 0,
                datetime.utcnow().isoformat(),
                user_id,
            ),
        )
        connection.commit()

    return {"user_id": user_id, **merged}


def _notification_exists_today(connection: sqlite3.Connection, user_id: int, kind: str, message: str) -> bool:
    row = connection.execute(
        "SELECT id FROM notifications WHERE user_id = ? AND kind = ? AND message = ? AND DATE(created_at) = DATE('now') LIMIT 1",
        (user_id, kind, message),
    ).fetchone()
    return row is not None


def create_notification(
    db_path: str,
    user_id: int,
    kind: str,
    message: str,
    meta: Optional[Dict[str, Any]] = None,
    dedupe_daily: bool = True,
) -> Optional[int]:
    with _connect(db_path) as connection:
        if dedupe_daily and _notification_exists_today(connection, user_id, kind, message):
            return None

        cursor = connection.execute(
            "INSERT INTO notifications (user_id, kind, message, is_read, meta_json, created_at) VALUES (?, ?, ?, 0, ?, ?)",
            (
                user_id,
                kind,
                message,
                json.dumps(meta or {}),
                datetime.utcnow().isoformat(),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def list_notifications(db_path: str, user_id: int, unread_only: bool = False, limit: int = 20) -> List[Dict[str, Any]]:
    query = "SELECT id, kind, message, is_read, meta_json, created_at FROM notifications WHERE user_id = ?"
    params: List[Any] = [user_id]
    if unread_only:
        query += " AND is_read = 0"
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _connect(db_path) as connection:
        rows = connection.execute(query, tuple(params)).fetchall()

    notifications: List[Dict[str, Any]] = []
    for row in rows:
        notifications.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "message": row["message"],
                "is_read": bool(row["is_read"]),
                "meta": json.loads(row["meta_json"] or "{}"),
                "created_at": row["created_at"],
            }
        )
    return notifications


def get_unread_notification_count(db_path: str, user_id: int) -> int:
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) as count FROM notifications WHERE user_id = ? AND is_read = 0",
            (user_id,),
        ).fetchone()
    return int(row["count"] if row else 0)


def mark_notification_read(db_path: str, user_id: int, notification_id: int) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            "UPDATE notifications SET is_read = 1 WHERE user_id = ? AND id = ?",
            (user_id, notification_id),
        )
        connection.commit()


def get_user_analytics(db_path: str, user_id: int) -> Dict[str, Any]:
    """Get analytics for a user: completion rate, streaks, consistency."""
    with _connect(db_path) as connection:
        # Use detailed progress events when available.
        total_row = connection.execute(
            "SELECT COUNT(*) as count FROM progress_events WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        total_recordings = total_row["count"] if total_row else 0

        completed_row = connection.execute(
            "SELECT COUNT(*) as count FROM progress_events WHERE user_id = ? AND status = 'completed'",
            (user_id,),
        ).fetchone()
        completed_count = completed_row["count"] if completed_row else 0

        missed_row = connection.execute(
            "SELECT COUNT(*) as count FROM progress_events WHERE user_id = ? AND status = 'missed'",
            (user_id,),
        ).fetchone()
        missed_count = missed_row["count"] if missed_row else 0

        # Backward compatibility fallback.
        if total_recordings == 0:
            total_row = connection.execute(
                "SELECT COUNT(*) as count FROM completions WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            total_recordings = total_row["count"] if total_row else 0
            completed_row = connection.execute(
                "SELECT COUNT(*) as count FROM completions WHERE user_id = ? AND completed = 1",
                (user_id,),
            ).fetchone()
            completed_count = completed_row["count"] if completed_row else 0
            missed_count = max(0, total_recordings - completed_count)

        subject_rows = connection.execute(
            """
            SELECT
                subject,
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as done
            FROM progress_events
            WHERE user_id = ?
            GROUP BY subject
            ORDER BY total DESC
            """,
            (user_id,),
        ).fetchall()

        if not subject_rows:
            subject_rows = connection.execute(
                "SELECT subject, COUNT(*) as total, SUM(completed) as done FROM completions WHERE user_id = ? GROUP BY subject ORDER BY total DESC",
                (user_id,),
            ).fetchall()

        subject_stats = []
        for row in subject_rows:
            done = row["done"] or 0
            completion_rate = (done / row["total"] * 100) if row["total"] > 0 else 0
            subject_stats.append({
                "subject": row["subject"],
                "total": row["total"],
                "completed": done,
                "rate": completion_rate,
            })

        daily_rows = connection.execute(
            """
            SELECT
                DATE(recorded_at) as day,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'missed' THEN 1 ELSE 0 END) as missed
            FROM progress_events
            WHERE user_id = ?
            GROUP BY day
            ORDER BY day DESC
            """,
            (user_id,),
        ).fetchall()

        if not daily_rows:
            daily_rows = connection.execute(
                """
                SELECT
                    DATE(recorded_at) as day,
                    SUM(CASE WHEN completed = 1 THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN completed = 0 THEN 1 ELSE 0 END) as missed
                FROM completions
                WHERE user_id = ?
                GROUP BY day
                ORDER BY day DESC
                """,
                (user_id,),
            ).fetchall()

        completed_days = {
            datetime.strptime(row["day"], "%Y-%m-%d").date()
            for row in daily_rows
            if (row["completed"] or 0) > 0
        }
        current_streak = 0
        check_day = date.today()
        while check_day in completed_days:
            current_streak += 1
            check_day -= timedelta(days=1)

        recent_activity: List[Dict[str, Any]] = []
        daily_map = {
            row["day"]: {
                "completed": int(row["completed"] or 0),
                "missed": int(row["missed"] or 0),
            }
            for row in daily_rows
        }
        for offset in range(13, -1, -1):
            day_value = date.today() - timedelta(days=offset)
            day_key = day_value.isoformat()
            day_stats = daily_map.get(day_key, {"completed": 0, "missed": 0})
            recent_activity.append(
                {
                    "date": day_key,
                    "label": day_value.strftime("%a"),
                    "completed": day_stats["completed"],
                    "missed": day_stats["missed"],
                }
            )

        total_recent_completed = sum(item["completed"] for item in recent_activity)
        total_recent_missed = sum(item["missed"] for item in recent_activity)
        total_recent_activity = total_recent_completed + total_recent_missed
        recent_completion_rate = (
            (total_recent_completed / total_recent_activity * 100)
            if total_recent_activity > 0
            else 0.0
        )

        plans_row = connection.execute(
            "SELECT COUNT(*) as count FROM plans WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        total_plans = plans_row["count"] if plans_row else 0

    completion_rate = (completed_count / total_recordings * 100) if total_recordings > 0 else 0

    return {
        "total_recordings": total_recordings,
        "completed_count": completed_count,
        "missed_count": missed_count,
        "completion_rate": completion_rate,
        "current_streak": current_streak,
        "total_plans": total_plans,
        "subject_stats": subject_stats,
        "recent_activity": recent_activity,
        "recent_activity_total": total_recent_activity,
        "recent_completion_rate": recent_completion_rate,
    }

