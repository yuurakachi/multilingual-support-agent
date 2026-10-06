"""Shared fixtures: a tiny hand-written store with one order per interesting case."""

from datetime import datetime, timedelta, timezone

import pytest

from support_agent.db import connect, create_schema

# Tools receive this as "now", so tests do not depend on the real date.
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)

ALICE = "alice@example.com"
YUKI = "yuki@example.jp"

ALICE_ADDRESS = "1 Main Street, Springfield, USA"
YUKI_ADDRESS = "〒150-0001 東京都渋谷区神宮前1-2-3"


def days_ago(days: int) -> str:
    return (NOW.date() - timedelta(days=days)).isoformat()


@pytest.fixture
def conn():
    conn = connect(":memory:")
    create_schema(conn)

    conn.executemany(
        "INSERT INTO customers (id, name, email, address) VALUES (?, ?, ?, ?)",
        [(1, "Alice Smith", ALICE, ALICE_ADDRESS), (2, "田中 由紀", YUKI, YUKI_ADDRESS)],
    )
    conn.executemany(
        "INSERT INTO products (id, name, price_cents) VALUES (?, ?, ?)",
        [(1, "Keyboard", 10000), (2, "Mouse", 2550)],
    )
    conn.executemany(
        """
        INSERT INTO orders (id, customer_id, status, shipping_address, tracking_number,
                            ordered_on, shipped_on, delivered_on, cancelled_on)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("ORD-1", 1, "processing", ALICE_ADDRESS, None, days_ago(1), None, None, None),
            ("ORD-2", 1, "shipped", ALICE_ADDRESS, "TRK000000002", days_ago(5), days_ago(3), None, None),
            # Delivered 10 days ago: inside the refund window.
            ("ORD-3", 1, "delivered", ALICE_ADDRESS, "TRK000000003", days_ago(15), days_ago(13), days_ago(10), None),
            # Delivered exactly 30 days ago: last day of the refund window.
            ("ORD-4", 2, "delivered", YUKI_ADDRESS, "TRK000000004", days_ago(35), days_ago(33), days_ago(30), None),
            # Delivered 31 days ago: one day too late.
            ("ORD-5", 2, "delivered", YUKI_ADDRESS, "TRK000000005", days_ago(36), days_ago(34), days_ago(31), None),
            ("ORD-6", 2, "cancelled", YUKI_ADDRESS, None, days_ago(8), None, None, days_ago(7)),
        ],
    )
    conn.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, unit_price_cents) "
        "VALUES (?, ?, ?, ?)",
        [
            ("ORD-1", 1, 1, 10000),
            ("ORD-1", 2, 2, 2550),
            ("ORD-2", 1, 1, 10000),
            ("ORD-3", 1, 1, 10000),
            ("ORD-4", 2, 1, 2550),
            ("ORD-5", 2, 1, 2550),
            ("ORD-6", 1, 1, 10000),
        ],
    )
    conn.commit()
    yield conn
    conn.close()
