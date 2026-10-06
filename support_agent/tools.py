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
import sqlite3
from datetime import datetime, timezone

CURRENCY = "USD"
MAX_ADDRESS_LENGTH = 300


def _error(code: str, message: str, **details) -> dict:
    return {"ok": False, "error_code": code, "message": message, **details}


def _is_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _money(cents: int) -> str:
    return f"{cents / 100:.2f}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
