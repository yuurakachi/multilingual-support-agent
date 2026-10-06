import json

from conftest import ALICE, YUKI, YUKI_ADDRESS, days_ago

from support_agent.db import connect
from support_agent.tools import get_order_status


def test_returns_order_details(conn):
    result = get_order_status(conn, "ORD-1", ALICE)

    assert result["ok"] is True
    assert result["order_id"] == "ORD-1"
    assert result["customer_name"] == "Alice Smith"
    assert result["status"] == "processing"
    assert result["ordered_on"] == days_ago(1)
    assert result["shipped_on"] is None
    assert result["tracking_number"] is None
    assert result["items"] == [
        {"product": "Keyboard", "quantity": 1, "unit_price": "100.00"},
        {"product": "Mouse", "quantity": 2, "unit_price": "25.50"},
    ]
    assert result["total"] == "151.00"
    assert result["currency"] == "USD"
    assert result["refund_status"] is None


def test_shipped_order_includes_tracking(conn):
    result = get_order_status(conn, "ORD-2", ALICE)

    assert result["status"] == "shipped"
    assert result["shipped_on"] == days_ago(3)
    assert result["tracking_number"] == "TRK000000002"


def test_result_is_json_serialisable_with_japanese_text(conn):
    result = get_order_status(conn, "ORD-4", YUKI)

    assert result["customer_name"] == "田中 由紀"
    assert result["shipping_address"] == YUKI_ADDRESS
    assert json.loads(json.dumps(result, ensure_ascii=False)) == result


def test_order_id_and_email_ignore_case_and_spaces(conn):
    result = get_order_status(conn, "  ord-1 ", "  Alice@Example.COM ")

    assert result["ok"] is True
    assert result["order_id"] == "ORD-1"


def test_unknown_order(conn):
    result = get_order_status(conn, "ORD-999", ALICE)

    assert result["ok"] is False
    assert result["error_code"] == "order_not_found"


def test_email_of_another_customer_reveals_nothing(conn):
    result = get_order_status(conn, "ORD-1", YUKI)

    assert result["ok"] is False
    assert result["error_code"] == "email_mismatch"
    assert set(result) == {"ok", "error_code", "message"}
    assert "Alice" not in result["message"]


def test_missing_arguments(conn):
    for order_id, email in [("", ALICE), ("ORD-1", ""), ("ORD-1", None), (None, ALICE), (1, ALICE)]:
        result = get_order_status(conn, order_id, email)
        assert result["ok"] is False
        assert result["error_code"] == "invalid_input"


def test_database_failure_is_an_error_result_not_an_exception():
    broken = connect(":memory:")  # no schema: every query fails
    try:
        result = get_order_status(broken, "ORD-1", ALICE)
    finally:
        broken.close()

    assert result["ok"] is False
    assert result["error_code"] == "internal_error"
