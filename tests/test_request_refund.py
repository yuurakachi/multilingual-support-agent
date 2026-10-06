import pytest
from conftest import ALICE, NOW, YUKI, days_ago

from support_agent.tools import get_order_status, request_refund

REASON = "The keyboard arrived with a broken key."


def _refund_count(conn):
    return conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0]


def test_creates_refund_inside_the_window(conn):
    result = request_refund(conn, "ORD-3", ALICE, REASON, now=NOW)

    assert result == {
        "ok": True,
        "refund_id": "RF-0001",
        "order_id": "ORD-3",
        "status": "requested",
        "amount": "100.00",
        "currency": "USD",
        "days_since_delivery": 10,
    }
    stored = conn.execute("SELECT * FROM refunds").fetchone()
    assert stored["order_id"] == "ORD-3"
    assert stored["reason"] == REASON
    assert stored["amount_cents"] == 10000
    assert stored["created_at"] == NOW.isoformat()


def test_order_status_shows_the_refund_afterwards(conn):
    request_refund(conn, "ORD-3", ALICE, REASON, now=NOW)

    assert get_order_status(conn, "ORD-3", ALICE)["refund_status"] == "requested"


def test_day_30_is_still_inside_the_window(conn):
    result = request_refund(conn, "ORD-4", YUKI, "サイズが合いませんでした。", now=NOW)

    assert result["ok"] is True
    assert result["days_since_delivery"] == 30


def test_day_31_is_outside_the_window(conn):
    result = request_refund(conn, "ORD-5", YUKI, REASON, now=NOW)

    assert result["ok"] is False
    assert result["error_code"] == "refund_window_expired"
    assert result["days_since_delivery"] == 31
    assert result["delivered_on"] == days_ago(31)
    assert result["refund_deadline"] == days_ago(1)
    assert _refund_count(conn) == 0


@pytest.mark.parametrize(
    "order_id, email, status",
    [("ORD-1", ALICE, "processing"), ("ORD-2", ALICE, "shipped"), ("ORD-6", YUKI, "cancelled")],
)
def test_rejects_order_that_was_not_delivered(conn, order_id, email, status):
    result = request_refund(conn, order_id, email, REASON, now=NOW)

    assert result["ok"] is False
    assert result["error_code"] == "order_not_delivered"
    assert result["status"] == status
    assert _refund_count(conn) == 0


def test_rejects_second_refund_for_the_same_order(conn):
    request_refund(conn, "ORD-3", ALICE, REASON, now=NOW)

    result = request_refund(conn, "ORD-3", ALICE, "Asking again.", now=NOW)

    assert result["ok"] is False
    assert result["error_code"] == "refund_already_requested"
    assert result["refund_id"] == "RF-0001"
    assert _refund_count(conn) == 1


def test_email_of_another_customer_creates_nothing(conn):
    result = request_refund(conn, "ORD-3", YUKI, REASON, now=NOW)

    assert result["ok"] is False
    assert result["error_code"] == "email_mismatch"
    assert _refund_count(conn) == 0


def test_unknown_order(conn):
    result = request_refund(conn, "ORD-999", ALICE, REASON, now=NOW)

    assert result["ok"] is False
    assert result["error_code"] == "order_not_found"


@pytest.mark.parametrize("bad_reason", ["", "   ", None])
def test_requires_a_reason(conn, bad_reason):
    result = request_refund(conn, "ORD-3", ALICE, bad_reason, now=NOW)

    assert result["ok"] is False
    assert result["error_code"] == "invalid_input"
    assert _refund_count(conn) == 0


def test_uses_the_current_date_by_default(conn):
    # The fixture dates are relative to NOW (2026-10-01); with the real clock,
    # ORD-3 can only get older, so it is either refundable or expired, never an exception.
    result = request_refund(conn, "ORD-3", ALICE, REASON)

    assert result["ok"] is True or result["error_code"] == "refund_window_expired"
