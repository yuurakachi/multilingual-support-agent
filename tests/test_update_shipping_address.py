import pytest
from conftest import ALICE, ALICE_ADDRESS, YUKI, YUKI_ADDRESS

from support_agent.tools import MAX_ADDRESS_LENGTH, update_shipping_address

NEW_ADDRESS = "99 Ocean Avenue, Santa Monica, CA 90401, USA"


def _stored_address(conn, order_id):
    return conn.execute(
        "SELECT shipping_address FROM orders WHERE id = ?", (order_id,)
    ).fetchone()["shipping_address"]


def test_updates_address_of_processing_order(conn):
    result = update_shipping_address(conn, "ORD-1", ALICE, NEW_ADDRESS)

    assert result == {
        "ok": True,
        "order_id": "ORD-1",
        "previous_address": ALICE_ADDRESS,
        "shipping_address": NEW_ADDRESS,
    }
    assert _stored_address(conn, "ORD-1") == NEW_ADDRESS


def test_accepts_japanese_address(conn):
    new_address = "〒530-0001 大阪府大阪市北区梅田2-4-9"

    result = update_shipping_address(conn, "ORD-1", ALICE, f"  {new_address}  ")

    assert result["ok"] is True
    assert _stored_address(conn, "ORD-1") == new_address


@pytest.mark.parametrize("order_id, email", [("ORD-2", ALICE), ("ORD-3", ALICE)])
def test_rejects_order_that_already_shipped(conn, order_id, email):
    result = update_shipping_address(conn, order_id, email, NEW_ADDRESS)

    assert result["ok"] is False
    assert result["error_code"] == "order_already_shipped"
    assert _stored_address(conn, order_id) == ALICE_ADDRESS


def test_rejects_cancelled_order(conn):
    result = update_shipping_address(conn, "ORD-6", YUKI, NEW_ADDRESS)

    assert result["ok"] is False
    assert result["error_code"] == "order_cancelled"
    assert _stored_address(conn, "ORD-6") == YUKI_ADDRESS


def test_email_of_another_customer_changes_nothing(conn):
    result = update_shipping_address(conn, "ORD-1", YUKI, NEW_ADDRESS)

    assert result["ok"] is False
    assert result["error_code"] == "email_mismatch"
    assert _stored_address(conn, "ORD-1") == ALICE_ADDRESS


def test_unknown_order(conn):
    result = update_shipping_address(conn, "ORD-999", ALICE, NEW_ADDRESS)

    assert result["ok"] is False
    assert result["error_code"] == "order_not_found"


@pytest.mark.parametrize("bad_address", ["", "   ", None, "x" * (MAX_ADDRESS_LENGTH + 1)])
def test_rejects_invalid_address(conn, bad_address):
    result = update_shipping_address(conn, "ORD-1", ALICE, bad_address)

    assert result["ok"] is False
    assert result["error_code"] == "invalid_input"
    assert _stored_address(conn, "ORD-1") == ALICE_ADDRESS
