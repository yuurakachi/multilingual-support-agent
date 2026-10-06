import pytest
from conftest import NOW

from support_agent.tools import MAX_TEXT_LENGTH, escalate_to_human

REASON = "Customer asked to speak with a human about a damaged package."


def _escalations(conn):
    return conn.execute("SELECT * FROM escalations ORDER BY id").fetchall()


def test_records_escalation(conn):
    result = escalate_to_human(conn, REASON, now=NOW)

    assert result == {"ok": True, "escalation_id": "ESC-0001", "status": "queued"}
    (escalation,) = _escalations(conn)
    assert escalation["reason"] == REASON
    assert escalation["status"] == "queued"
    assert escalation["created_at"] == NOW.isoformat()


def test_escalation_ids_increase(conn):
    escalate_to_human(conn, REASON, now=NOW)
    second = escalate_to_human(conn, "El cliente está muy molesto.", now=NOW)

    assert second["escalation_id"] == "ESC-0002"


@pytest.mark.parametrize("bad_reason", ["", "   ", None, "x" * (MAX_TEXT_LENGTH + 1)])
def test_requires_a_valid_reason(conn, bad_reason):
    result = escalate_to_human(conn, bad_reason, now=NOW)

    assert result["ok"] is False
    assert result["error_code"] == "invalid_input"
    assert _escalations(conn) == []
