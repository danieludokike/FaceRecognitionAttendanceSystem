import os
import sqlite3
import uuid
import numpy as np
from datetime import datetime

_CONN = None
_DB_PATH = None

def init_db(db_path: str):
    global _CONN, _DB_PATH
    first = not os.path.exists(db_path)
    _DB_PATH = db_path
    _CONN = sqlite3.connect(db_path, check_same_thread=False)
    _CONN.execute('PRAGMA foreign_keys = ON;')
    _CONN.row_factory = sqlite3.Row

    _CONN.executescript(
        '''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS admins (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            image_path TEXT,
            embed_dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            matric TEXT UNIQUE,
            image_path TEXT,
            embed_dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        );
        '''
    )

    # defaults
    if get_setting('admin_pin') is None:
        set_setting('admin_pin', '1234')
    if get_setting('cosine_threshold') is None:
        set_setting('cosine_threshold', '0.40')  # slightly stricter default
    if get_setting('consecutive_frames') is None:
        set_setting('consecutive_frames', '2')

def conn():
    return _CONN

# ---------- Settings ----------
def get_setting(key: str, default: str | None = None) -> str | None:
    cur = _CONN.execute('SELECT value FROM settings WHERE key=?', (key,))
    row = cur.fetchone()
    return row['value'] if row else default

def set_setting(key: str, value: str):
    _CONN.execute(
        'INSERT INTO settings(key, value) VALUES(?, ?) '
        'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
        (key, value)
    )
    _CONN.commit()

# ---------- Admins ----------
def add_admin(name: str, embedding: np.ndarray, image_path: str | None = None) -> str:
    admin_id = str(uuid.uuid4())
    emb = np.asarray(embedding, dtype=np.float32)
    _CONN.execute(
        'INSERT INTO admins(id, name, image_path, embed_dim, embedding, created_at) VALUES(?, ?, ?, ?, ?, ?)',
        (admin_id, name, image_path, int(emb.shape[0]), emb.tobytes(), datetime.utcnow().isoformat())
    )
    _CONN.commit()
    return admin_id

def all_admin_embeddings() -> list[tuple[str, str, np.ndarray]]:
    cur = _CONN.execute('SELECT id, name, embed_dim, embedding FROM admins')
    out = []
    for r in cur.fetchall():
        emb = np.frombuffer(r['embedding'], dtype=np.float32)[: r['embed_dim']]
        out.append((r['id'], r['name'], emb))
    return out

# ---------- Students ----------
def add_student(full_name: str, matric: str | None, embedding: np.ndarray, image_path: str | None = None) -> str:
    sid = str(uuid.uuid4())
    emb = np.asarray(embedding, dtype=np.float32)
    _CONN.execute(
        'INSERT INTO students(id, full_name, matric, image_path, embed_dim, embedding, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)',
        (sid, full_name, matric, image_path, int(emb.shape[0]), emb.tobytes(), datetime.utcnow().isoformat())
    )
    _CONN.commit()
    return sid

def all_students():
    cur = _CONN.execute('SELECT id, full_name, matric, image_path, embed_dim FROM students ORDER BY full_name')
    return cur.fetchall()

def all_student_embeddings() -> list[tuple[str, str, str | None, np.ndarray]]:
    cur = _CONN.execute('SELECT id, full_name, matric, embed_dim, embedding FROM students')
    out = []
    for r in cur.fetchall():
        emb = np.frombuffer(r['embedding'], dtype=np.float32)[: r['embed_dim']]
        out.append((r['id'], r['full_name'], r['matric'], emb))
    return out

def delete_student(student_id: str):
    _CONN.execute('DELETE FROM students WHERE id=?', (student_id,))
    _CONN.commit()
