"""Bridge between the model and the Python tools.

TOOL_DEFINITIONS is what the model sees: a name, a description of when to use
the tool, and a JSON Schema for its arguments. execute_tool is what runs when
the model asks for one of them.
"""

from __future__ import annotations

import sqlite3

from support_agent import tools


def _definition(name: str, description: str, properties: dict[str, str]) -> dict:
    """Build a tool definition where every argument is a required string."""
    return {
        "name": name,
        "description": description,
        # strict: the API guarantees the arguments match this schema exactly.
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                argument: {"type": "string", "description": text}
                for argument, text in properties.items()
            },
            "required": list(properties),
            "additionalProperties": False,
        },
    }


_ORDER_ID = "The order number the customer gave, for example ORD-1001."
_EMAIL = "The email address the customer gave for this order."

TOOL_DEFINITIONS = [
    _definition(
        "get_order_status",
        "Look up an order: status, dates, tracking number, shipping address, items and total. "
        "Call this whenever the customer asks about an order and has given both the order "
        "number and their email. It also verifies that the email matches the order.",
        {"order_id": _ORDER_ID, "email": _EMAIL},
    ),
    _definition(
        "update_shipping_address",
        "Change the shipping address of an order. Call this when the customer asks to send "
        "an order to a different address and has given the complete new address. It only "
        "succeeds while the order is still processing.",
        {
            "order_id": _ORDER_ID,
            "email": _EMAIL,
            "new_address": "The complete new shipping address, exactly as the customer wrote it.",
        },
    ),
    _definition(
        "request_refund",
        "Open a refund request for an order. Call this when the customer asks for a refund "
        "and has explained why. It only succeeds for delivered orders within 30 days of "
        "delivery, once per order.",
        {
            "order_id": _ORDER_ID,
            "email": _EMAIL,
            "reason": "Why the customer wants a refund, in English.",
        },
    ),
    _definition(
        "create_support_ticket",
        "Open a ticket so the support team follows up by email. Call this for requests the "
        "other tools do not cover, when the customer is fine with an email follow-up.",
        {
            "email": "The email address where the customer wants to be contacted.",
            "summary": "What the customer needs, in English, with any order number and "
            "relevant details.",
        },
    ),
    _definition(
        "escalate_to_human",
        "Hand the conversation to a human agent. Call this when the customer asks for a "
        "person, is very upset, or the case is beyond what the other tools can handle.",
        {
            "reason": "Why a human is needed, in English, with the order number and email if "
            "known and what has already been tried.",
        },
    ),
]

TOOL_FUNCTIONS = {
    "get_order_status": tools.get_order_status,
    "update_shipping_address": tools.update_shipping_address,
    "request_refund": tools.request_refund,
    "create_support_ticket": tools.create_support_ticket,
    "escalate_to_human": tools.escalate_to_human,
}

_EXPECTED_ARGUMENTS = {
    definition["name"]: set(definition["input_schema"]["required"])
    for definition in TOOL_DEFINITIONS
}


def execute_tool(conn: sqlite3.Connection, name: str, tool_input) -> dict:
    """Run the tool the model asked for and always return a result dict.

    A bad request from the model (unknown tool, wrong arguments) or a bug in a
    tool becomes an error result, so one bad call cannot crash the conversation.
    """
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return {
            "ok": False,
            "error_code": "unknown_tool",
            "message": f"There is no tool named {name!r}. "
            f"Available tools: {', '.join(TOOL_FUNCTIONS)}.",
        }

    # Only the arguments in the schema are passed on. This also keeps the model
    # from setting internal parameters such as the injectable clock ("now").
    expected = _EXPECTED_ARGUMENTS[name]
    if not isinstance(tool_input, dict) or set(tool_input) != expected:
        return {
            "ok": False,
            "error_code": "invalid_input",
            "message": f"{name} takes exactly these arguments: {', '.join(sorted(expected))}.",
        }

    try:
        return function(conn, **tool_input)
    except Exception as error:  # noqa: BLE001 - a tool bug must not end the conversation
        return {
            "ok": False,
            "error_code": "internal_error",
            "message": "The store system could not complete the request.",
            # Only the exception type: raw error text is not something to show the model.
            "detail": type(error).__name__,
        }
