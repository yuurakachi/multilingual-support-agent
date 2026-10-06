"""SQLite connection and schema for the simulated store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "store.db"

ORDER_STATUSES = ("processing", "shipped", "delivered", "cancelled")

# Money is stored as integer cents to avoid floating point rounding.
# Order dates are ISO dates (YYYY-MM-DD); created_at columns are ISO datetimes in UTC.
SCHEMA = """
CREATE TABLE customers (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL,
    email   TEXT NOT NULL UNIQUE COLLATE NOCASE,
    address TEXT NOT NULL
);

CREATE TABLE products (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    price_cents INTEGER NOT NULL CHECK (price_cents > 0)
);

CREATE TABLE orders (
    id               TEXT PRIMARY KEY,
    customer_id      INTEGER NOT NULL REFERENCES customers(id),
    status           TEXT NOT NULL
                     CHECK (status IN ('processing', 'shipped', 'delivered', 'cancelled')),
    shipping_address TEXT NOT NULL,
    tracking_number  TEXT,
    ordered_on       TEXT NOT NULL,
    shipped_on       TEXT,
    delivered_on     TEXT,
    cancelled_on     TEXT
);

CREATE TABLE order_items (
    order_id         TEXT NOT NULL REFERENCES orders(id),
    product_id       INTEGER NOT NULL REFERENCES products(id),
    quantity         INTEGER NOT NULL CHECK (quantity > 0),
    unit_price_cents INTEGER NOT NULL,
    PRIMARY KEY (order_id, product_id)
);

CREATE TABLE refunds (
    id           INTEGER PRIMARY KEY,
    order_id     TEXT NOT NULL UNIQUE REFERENCES orders(id),
    reason       TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'requested',
    created_at   TEXT NOT NULL
);

CREATE TABLE support_tickets (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    email       TEXT NOT NULL,
    summary     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  TEXT NOT NULL
);

CREATE TABLE escalations (
    id         INTEGER PRIMARY KEY,
    reason     TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL
);
"""


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection with dict-like rows and foreign keys enforced."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
