"""Tests for the simulated store data."""

from datetime import date

import pytest

from support_agent.db import ORDER_STATUSES, connect, create_schema
from support_agent.seed import build_database, seed

TODAY = date(2026, 10, 1)


def _seeded_connection():
    conn = connect(":memory:")
    create_schema(conn)
    seed(conn, TODAY)
    return conn


@pytest.fixture
def conn():
    conn = _seeded_connection()
    yield conn
    conn.close()


def _days_since(iso_day):
    return (TODAY - date.fromisoformat(iso_day)).days


def test_row_counts(conn):
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 20
    assert conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 12
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 50


def test_order_ids_are_sequential(conn):
    ids = [row["id"] for row in conn.execute("SELECT id FROM orders ORDER BY id")]
    assert ids == [f"ORD-{number}" for number in range(1001, 1051)]


def test_every_status_is_present(conn):
    statuses = {row["status"] for row in conn.execute("SELECT DISTINCT status FROM orders")}
    assert statuses == set(ORDER_STATUSES)


def test_includes_japanese_customers(conn):
    names = [row["name"] for row in conn.execute("SELECT name FROM customers")]
    japanese = [name for name in names if any("぀" <= ch <= "鿿" for ch in name)]
    assert len(japanese) >= 5


def test_every_customer_has_an_order(conn):
    without_orders = conn.execute(
        "SELECT COUNT(*) FROM customers c "
        "WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)"
    ).fetchone()[0]
    assert without_orders == 0


def test_every_order_has_items(conn):
    without_items = conn.execute(
        "SELECT COUNT(*) FROM orders o "
        "WHERE NOT EXISTS (SELECT 1 FROM order_items i WHERE i.order_id = o.id)"
    ).fetchone()[0]
    assert without_items == 0


def test_dates_match_status(conn):
    for order in conn.execute("SELECT * FROM orders"):
        status = order["status"]
        assert order["ordered_on"] <= TODAY.isoformat()

        if status == "processing":
            assert order["shipped_on"] is None
            assert order["delivered_on"] is None
            assert order["cancelled_on"] is None
            assert order["tracking_number"] is None
        elif status == "shipped":
            assert order["ordered_on"] <= order["shipped_on"] <= TODAY.isoformat()
            assert order["delivered_on"] is None
            assert order["tracking_number"]
        elif status == "delivered":
            assert order["ordered_on"] <= order["shipped_on"] <= order["delivered_on"]
            assert order["delivered_on"] <= TODAY.isoformat()
            assert order["tracking_number"]
        elif status == "cancelled":
            assert order["ordered_on"] <= order["cancelled_on"] <= TODAY.isoformat()
            assert order["shipped_on"] is None
            assert order["delivered_on"] is None


def test_delivered_orders_fall_on_both_sides_of_the_refund_window(conn):
    ages = [
        _days_since(row["delivered_on"])
        for row in conn.execute("SELECT delivered_on FROM orders WHERE status = 'delivered'")
    ]
    assert any(age <= 30 for age in ages)
    assert any(age > 30 for age in ages)


def test_anchor_orders_have_a_known_state(conn):
    def order(order_id):
        return conn.execute(
            "SELECT o.*, c.email FROM orders o JOIN customers c ON c.id = o.customer_id "
            "WHERE o.id = ?",
            (order_id,),
        ).fetchone()

    processing = order("ORD-1001")
    assert (processing["status"], processing["email"]) == ("processing", "yuki.tanaka@example.jp")

    assert order("ORD-1002")["status"] == "shipped"

    refundable = order("ORD-1003")
    assert refundable["status"] == "delivered"
    assert _days_since(refundable["delivered_on"]) == 10

    too_old = order("ORD-1004")
    assert too_old["status"] == "delivered"
    assert _days_since(too_old["delivered_on"]) == 45

    assert order("ORD-1005")["status"] == "cancelled"


def test_seed_is_deterministic():
    def dump(conn):
        return [tuple(row) for row in conn.execute("SELECT * FROM orders ORDER BY id")] + [
            tuple(row)
            for row in conn.execute("SELECT * FROM order_items ORDER BY order_id, product_id")
        ]

    first, second = _seeded_connection(), _seeded_connection()
    try:
        assert dump(first) == dump(second)
    finally:
        first.close()
        second.close()


def test_build_database_replaces_existing_file(tmp_path):
    db_path = tmp_path / "store.db"
    build_database(db_path, TODAY)
    build_database(db_path, TODAY)

    conn = connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 50
    finally:
        conn.close()
