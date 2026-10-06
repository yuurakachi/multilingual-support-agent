"""Support tools the agent can call.

Every tool takes an open SQLite connection plus plain arguments and returns a
JSON-serialisable dict:

    success: {"ok": True, ...data}
    failure: {"ok": False, "error_code": "<stable_code>", "message": "<explanation>"}

Tools never raise for expected problems (unknown order, wrong email, request
outside policy). They return an error result instead, so the model can explain
the problem to the customer and the logs record exactly what happened.
"""

from __future__ import annotations

import functools
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone

CURRENCY = "USD"
MAX_ADDRESS_LENGTH = 300
REFUND_WINDOW_DAYS = 30
MAX_TEXT_LENGTH = 2000

_EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def _error(code: str, message: str, **details) -> dict:
    return {"ok": False, "error_code": code, "message": message, **details}


def _is_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _money(cents: int) -> str:
    return f"{cents / 100:.2f}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _public_id(prefix: str, row_id: int) -> str:
    """Customer-facing reference such as RF-0001 or TKT-0012."""
    return f"{prefix}-{row_id:04d}"


def _tool(func):
    """Turn database failures into an error result instead of an exception."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except sqlite3.Error:
            return _error(
                "internal_error",
                "The store system could not complete the request. Nothing was changed.",
            )

    return wrapper


def _find_verified_order(conn: sqlite3.Connection, order_id, email):
    """Identity check shared by every order tool.

    Returns (order_row, None) when the order exists and belongs to that email,
    otherwise (None, error_result).
    """
    if not _is_text(order_id) or not _is_text(email):
        return None, _error(
            "invalid_input", "Both the order number and the email address are required."
        )

    order = conn.execute(
        """
        SELECT o.*, c.name AS customer_name, c.email AS customer_email
        FROM orders o JOIN customers c ON c.id = o.customer_id
        WHERE o.id = ?
        """,
        (order_id.strip().upper(),),
    ).fetchone()

    if order is None:
        return None, _error("order_not_found", f"No order with number {order_id.strip()} exists.")
    if order["customer_email"].lower() != email.strip().lower():
        return None, _error(
            "email_mismatch",
            "The email address does not match the one on this order. "
            "No order details can be shared.",
        )
    return order, None


def _order_total_cents(conn: sqlite3.Connection, order_id: str) -> int:
    return conn.execute(
        "SELECT SUM(quantity * unit_price_cents) FROM order_items WHERE order_id = ?",
        (order_id,),
    ).fetchone()[0]


@_tool
def get_order_status(conn: sqlite3.Connection, order_id: str, email: str) -> dict:
    """Look up an order. The email must match the customer who placed it."""
    order, error = _find_verified_order(conn, order_id, email)
    if error:
        return error

    items = conn.execute(
        """
        SELECT p.name, i.quantity, i.unit_price_cents
        FROM order_items i JOIN products p ON p.id = i.product_id
        WHERE i.order_id = ?
        ORDER BY p.name
        """,
        (order["id"],),
    ).fetchall()
    refund = conn.execute(
        "SELECT status FROM refunds WHERE order_id = ?", (order["id"],)
    ).fetchone()

    return {
        "ok": True,
        "order_id": order["id"],
        "customer_name": order["customer_name"],
        "status": order["status"],
        "ordered_on": order["ordered_on"],
        "shipped_on": order["shipped_on"],
        "delivered_on": order["delivered_on"],
        "cancelled_on": order["cancelled_on"],
        "tracking_number": order["tracking_number"],
        "shipping_address": order["shipping_address"],
        "items": [
            {
                "product": item["name"],
                "quantity": item["quantity"],
                "unit_price": _money(item["unit_price_cents"]),
            }
            for item in items
        ],
        "total": _money(_order_total_cents(conn, order["id"])),
        "currency": CURRENCY,
        "refund_status": refund["status"] if refund else None,
    }


@_tool
def update_shipping_address(
    conn: sqlite3.Connection, order_id: str, email: str, new_address: str
) -> dict:
    """Change where an order is sent. Only allowed before the order ships."""
    order, error = _find_verified_order(conn, order_id, email)
    if error:
        return error

    if not _is_text(new_address):
        return _error("invalid_input", "A new shipping address is required.")
    new_address = new_address.strip()
    if len(new_address) > MAX_ADDRESS_LENGTH:
        return _error(
            "invalid_input",
            f"The shipping address is too long (maximum {MAX_ADDRESS_LENGTH} characters).",
        )

    status = order["status"]
    if status == "cancelled":
        return _error(
            "order_cancelled",
            "This order was cancelled, so its shipping address cannot be changed.",
            status=status,
        )
    if status != "processing":
        return _error(
            "order_already_shipped",
            f"This order is already {status}. The shipping address can only be changed "
            "before the order ships.",
            status=status,
        )

    with conn:
        conn.execute(
            "UPDATE orders SET shipping_address = ? WHERE id = ?", (new_address, order["id"])
        )
    return {
        "ok": True,
        "order_id": order["id"],
        "previous_address": order["shipping_address"],
        "shipping_address": new_address,
    }


@_tool
def request_refund(
    conn: sqlite3.Connection,
    order_id: str,
    email: str,
    reason: str,
    *,
    now: datetime | None = None,
) -> dict:
    """Request a refund. Only allowed for delivered orders, up to 30 days after delivery."""
    order, error = _find_verified_order(conn, order_id, email)
    if error:
        return error

    if not _is_text(reason):
        return _error("invalid_input", "A reason for the refund is required.")

    status = order["status"]
    if status != "delivered":
        return _error(
            "order_not_delivered",
            f"Refunds can only be requested after delivery. This order is {status}.",
            status=status,
        )

    existing = conn.execute(
        "SELECT id, status FROM refunds WHERE order_id = ?", (order["id"],)
    ).fetchone()
    if existing:
        return _error(
            "refund_already_requested",
            "A refund has already been requested for this order.",
            refund_id=_public_id("RF", existing["id"]),
            refund_status=existing["status"],
        )

    now = now or _utc_now()
    delivered_on = date.fromisoformat(order["delivered_on"])
    days_since_delivery = (now.date() - delivered_on).days
    if days_since_delivery > REFUND_WINDOW_DAYS:
        return _error(
            "refund_window_expired",
            f"Refunds must be requested within {REFUND_WINDOW_DAYS} days of delivery. "
            f"This order was delivered {days_since_delivery} days ago.",
            delivered_on=order["delivered_on"],
            days_since_delivery=days_since_delivery,
            refund_deadline=(delivered_on + timedelta(days=REFUND_WINDOW_DAYS)).isoformat(),
        )

    amount_cents = _order_total_cents(conn, order["id"])
    with conn:
        cursor = conn.execute(
            "INSERT INTO refunds (order_id, reason, amount_cents, created_at) VALUES (?, ?, ?, ?)",
            (order["id"], reason.strip(), amount_cents, now.isoformat()),
        )
    return {
        "ok": True,
        "refund_id": _public_id("RF", cursor.lastrowid),
        "order_id": order["id"],
        "status": "requested",
        "amount": _money(amount_cents),
        "currency": CURRENCY,
        "days_since_delivery": days_since_delivery,
    }


@_tool
def create_support_ticket(
    conn: sqlite3.Connection, email: str, summary: str, *, now: datetime | None = None
) -> dict:
    """Open a ticket for the support team to follow up by email."""
    if not _is_text(email) or not _EMAIL_PATTERN.fullmatch(email.strip()):
        return _error("invalid_input", "A valid email address is required to open a ticket.")
    if not _is_text(summary):
        return _error("invalid_input", "A summary of the problem is required.")
    if len(summary.strip()) > MAX_TEXT_LENGTH:
        return _error(
            "invalid_input", f"The summary is too long (maximum {MAX_TEXT_LENGTH} characters)."
        )

    email = email.strip().lower()
    # Anyone can open a ticket; it is linked to a customer when the email is known.
    customer = conn.execute("SELECT id FROM customers WHERE email = ?", (email,)).fetchone()
    now = now or _utc_now()
    with conn:
        cursor = conn.execute(
            "INSERT INTO support_tickets (customer_id, email, summary, created_at) "
            "VALUES (?, ?, ?, ?)",
            (customer["id"] if customer else None, email, summary.strip(), now.isoformat()),
        )
    return {
        "ok": True,
        "ticket_id": _public_id("TKT", cursor.lastrowid),
        "status": "open",
        "email": email,
    }


@_tool
def escalate_to_human(
    conn: sqlite3.Connection, reason: str, *, now: datetime | None = None
) -> dict:
    """Hand the conversation over to a human agent."""
    if not _is_text(reason):
        return _error("invalid_input", "A reason for the escalation is required.")
    if len(reason.strip()) > MAX_TEXT_LENGTH:
        return _error(
            "invalid_input", f"The reason is too long (maximum {MAX_TEXT_LENGTH} characters)."
        )

    now = now or _utc_now()
    with conn:
        cursor = conn.execute(
            "INSERT INTO escalations (reason, created_at) VALUES (?, ?)",
            (reason.strip(), now.isoformat()),
        )
    return {
        "ok": True,
        "escalation_id": _public_id("ESC", cursor.lastrowid),
        "status": "queued",
    }
