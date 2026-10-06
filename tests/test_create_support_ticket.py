import pytest
from conftest import ALICE, NOW

from support_agent.tools import MAX_TEXT_LENGTH, create_support_ticket

SUMMARY = "Customer received the wrong colour and wants an exchange."


def _tickets(conn):
    return conn.execute("SELECT * FROM support_tickets ORDER BY id").fetchall()


def test_creates_ticket_linked_to_known_customer(conn):
    result = create_support_ticket(conn, "  Alice@Example.com ", SUMMARY, now=NOW)

    assert result == {"ok": True, "ticket_id": "TKT-0001", "status": "open", "email": ALICE}
    (ticket,) = _tickets(conn)
    assert ticket["customer_id"] == 1
    assert ticket["email"] == ALICE
    assert ticket["summary"] == SUMMARY
    assert ticket["status"] == "open"
    assert ticket["created_at"] == NOW.isoformat()


def test_creates_ticket_for_unknown_email(conn):
    result = create_support_ticket(conn, "visitor@example.org", "配送について質問があります。", now=NOW)

    assert result["ok"] is True
    (ticket,) = _tickets(conn)
    assert ticket["customer_id"] is None
    assert ticket["summary"] == "配送について質問があります。"


def test_ticket_ids_increase(conn):
    first = create_support_ticket(conn, ALICE, SUMMARY, now=NOW)
    second = create_support_ticket(conn, ALICE, "Another problem.", now=NOW)

    assert (first["ticket_id"], second["ticket_id"]) == ("TKT-0001", "TKT-0002")


@pytest.mark.parametrize("bad_email", ["", "   ", None, "not-an-email", "a@b", "two words@example.com"])
def test_rejects_invalid_email(conn, bad_email):
    result = create_support_ticket(conn, bad_email, SUMMARY, now=NOW)

    assert result["ok"] is False
    assert result["error_code"] == "invalid_input"
    assert _tickets(conn) == []


@pytest.mark.parametrize("bad_summary", ["", "   ", None, "x" * (MAX_TEXT_LENGTH + 1)])
def test_rejects_invalid_summary(conn, bad_summary):
    result = create_support_ticket(conn, ALICE, bad_summary, now=NOW)

    assert result["ok"] is False
    assert result["error_code"] == "invalid_input"
    assert _tickets(conn) == []
