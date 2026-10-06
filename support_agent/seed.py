"""Builds the simulated store database.

Dates are relative to the day the seed runs, so the data never goes stale:
an order "delivered 10 days ago" is always 10 days old in a fresh database.
Everything else is deterministic (fixed random seed), so two fresh databases
built on the same day are identical.

Usage:
    python -m support_agent.seed
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from support_agent.db import DEFAULT_DB_PATH, connect, create_schema

RANDOM_SEED = 42
FIRST_ORDER_NUMBER = 1001

# (name, email, default shipping address)
CUSTOMERS = [
    ("田中 由紀", "yuki.tanaka@example.jp", "〒150-0001 東京都渋谷区神宮前1-2-3"),
    ("佐藤 健", "ken.sato@example.jp", "〒530-0001 大阪府大阪市北区梅田2-4-9"),
    ("鈴木 さくら", "sakura.suzuki@example.jp", "〒604-8005 京都府京都市中京区恵比須町5-1"),
    ("高橋 大輔", "daisuke.takahashi@example.jp", "〒060-0001 北海道札幌市中央区北一条西3-7"),
    ("渡辺 美咲", "misaki.watanabe@example.jp", "〒812-0011 福岡県福岡市博多区博多駅前1-8-2"),
    ("伊藤 春香", "haruka.ito@example.jp", "〒460-0008 愛知県名古屋市中区栄4-6-15"),
    ("María García", "maria.garcia@example.com", "Calle de Alcalá 45, 3ºB, 28014 Madrid, España"),
    ("Carlos Hernández", "carlos.hernandez@example.com", "Av. Insurgentes Sur 1235, Col. Del Valle, 03100 Ciudad de México, México"),
    ("Lucía Fernández", "lucia.fernandez@example.com", "Av. Corrientes 3120, Piso 5, C1193 Buenos Aires, Argentina"),
    ("Javier Morales", "javier.morales@example.com", "Carrera 7 #71-21, Chapinero, Bogotá, Colombia"),
    ("Valentina Rojas", "valentina.rojas@example.com", "Av. Providencia 1650, Providencia, Santiago, Chile"),
    ("Diego Ramírez", "diego.ramirez@example.com", "Av. Chapultepec 480, Col. Americana, 44160 Guadalajara, México"),
    ("Sofía Castillo", "sofia.castillo@example.com", "Av. Larco 812, Miraflores, Lima 15074, Perú"),
    ("Emily Johnson", "emily.johnson@example.com", "742 Evergreen Terrace, Portland, OR 97205, USA"),
    ("James O'Connor", "james.oconnor@example.com", "18 Baggot Street Lower, Dublin 2, D02 X658, Ireland"),
    ("Olivia Brown", "olivia.brown@example.com", "27 Deansgate, Manchester M3 2BW, United Kingdom"),
    ("Liam Nguyen", "liam.nguyen@example.com", "5/120 Collins Street, Melbourne VIC 3000, Australia"),
    ("Priya Patel", "priya.patel@example.com", "88 Queen Street West, Unit 1402, Toronto, ON M5H 2N2, Canada"),
    ("Noah Müller", "noah.mueller@example.com", "Torstraße 101, 10119 Berlin, Germany"),
    ("Amara Okafor", "amara.okafor@example.com", "14 Admiralty Way, Lekki Phase 1, Lagos, Nigeria"),
]

# (name, price in cents)
PRODUCTS = [
    ("Wireless Earbuds", 7999),
    ("Mechanical Keyboard", 12900),
    ("USB-C Hub", 4550),
    ("Laptop Stand", 3999),
    ("27-inch Monitor", 28900),
    ("1080p Webcam", 5990),
    ("Ergonomic Mouse", 4999),
    ("Noise-Cancelling Headphones", 19900),
    ("Portable SSD 1TB", 10999),
    ("Smartwatch", 14900),
    ("Desk Lamp", 3450),
    ("Phone Case", 1999),
]

# Hand-picked orders with a known state, so tests and scenario scripts can rely on them.
# They become ORD-1001, ORD-1002, ... in this order.
# (customer email, status, days since that status was reached, [(product index, quantity)])
ANCHOR_ORDERS = [
    ("yuki.tanaka@example.jp", "processing", 2, [(1, 1), (6, 1)]),
    ("maria.garcia@example.com", "shipped", 4, [(0, 1)]),
    ("emily.johnson@example.com", "delivered", 10, [(7, 1)]),
    ("ken.sato@example.jp", "delivered", 45, [(4, 1)]),
    ("carlos.hernandez@example.com", "cancelled", 6, [(9, 1)]),
    ("james.oconnor@example.com", "processing", 1, [(2, 2), (3, 1)]),
    ("sakura.suzuki@example.jp", "delivered", 5, [(8, 1), (11, 2)]),
    ("lucia.fernandez@example.com", "delivered", 60, [(10, 1)]),
]

# The remaining orders are generated. Together with the anchors this gives 50.
GENERATED_STATUS_COUNTS = {"processing": 8, "shipped": 10, "delivered": 19, "cancelled": 5}

# Range for "days since the status was reached" of generated orders.
DAYS_AGO_RANGE = {
    "processing": (0, 3),
    "shipped": (1, 6),
    "delivered": (1, 90),
    "cancelled": (1, 40),
}


def _timeline(status: str, days_ago: int, rng: random.Random, today: date) -> dict:
    """Build consistent order dates, working backwards from the latest event."""
    event_day = today - timedelta(days=days_ago)
    dates = {"ordered_on": None, "shipped_on": None, "delivered_on": None, "cancelled_on": None}

    if status == "processing":
        dates["ordered_on"] = event_day
    elif status == "shipped":
        dates["shipped_on"] = event_day
        dates["ordered_on"] = event_day - timedelta(days=rng.randint(1, 2))
    elif status == "delivered":
        dates["delivered_on"] = event_day
        dates["shipped_on"] = event_day - timedelta(days=rng.randint(2, 6))
        dates["ordered_on"] = dates["shipped_on"] - timedelta(days=rng.randint(1, 2))
    elif status == "cancelled":
        dates["cancelled_on"] = event_day
        dates["ordered_on"] = event_day - timedelta(days=rng.randint(0, 1))
    else:
        raise ValueError(f"Unknown order status: {status}")

    return {column: day.isoformat() if day else None for column, day in dates.items()}


def _insert_order(
    conn: sqlite3.Connection,
    rng: random.Random,
    today: date,
    order_number: int,
    customer_id: int,
    status: str,
    days_ago: int,
    items: list[tuple[int, int]],
) -> None:
    order_id = f"ORD-{order_number}"
    dates = _timeline(status, days_ago, rng, today)
    has_shipped = status in ("shipped", "delivered")
    tracking_number = f"TRK{rng.randrange(10**9):09d}" if has_shipped else None
    address = conn.execute(
        "SELECT address FROM customers WHERE id = ?", (customer_id,)
    ).fetchone()["address"]

    conn.execute(
        """
        INSERT INTO orders (id, customer_id, status, shipping_address, tracking_number,
                            ordered_on, shipped_on, delivered_on, cancelled_on)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            order_id,
            customer_id,
            status,
            address,
            tracking_number,
            dates["ordered_on"],
            dates["shipped_on"],
            dates["delivered_on"],
            dates["cancelled_on"],
        ),
    )
    for product_index, quantity in items:
        conn.execute(
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price_cents) "
            "VALUES (?, ?, ?, ?)",
            (order_id, product_index + 1, quantity, PRODUCTS[product_index][1]),
        )


def seed(conn: sqlite3.Connection, today: date | None = None) -> None:
    """Fill an empty database (schema already created) with simulated data."""
    today = today or date.today()
    rng = random.Random(RANDOM_SEED)

    conn.executemany(
        "INSERT INTO customers (id, name, email, address) VALUES (?, ?, ?, ?)",
        [(i, *customer) for i, customer in enumerate(CUSTOMERS, start=1)],
    )
    conn.executemany(
        "INSERT INTO products (id, name, price_cents) VALUES (?, ?, ?)",
        [(i, *product) for i, product in enumerate(PRODUCTS, start=1)],
    )

    customer_id_by_email = {email: i for i, (_, email, _) in enumerate(CUSTOMERS, start=1)}
    order_number = FIRST_ORDER_NUMBER

    for email, status, days_ago, items in ANCHOR_ORDERS:
        _insert_order(
            conn, rng, today, order_number, customer_id_by_email[email], status, days_ago, items
        )
        order_number += 1

    statuses = [s for s, count in GENERATED_STATUS_COUNTS.items() for _ in range(count)]
    rng.shuffle(statuses)
    for i, status in enumerate(statuses):
        # Round-robin over customers so every customer ends up with orders.
        customer_id = i % len(CUSTOMERS) + 1
        days_ago = rng.randint(*DAYS_AGO_RANGE[status])
        product_indexes = rng.sample(range(len(PRODUCTS)), k=rng.randint(1, 3))
        items = [(product_index, rng.randint(1, 2)) for product_index in product_indexes]
        _insert_order(conn, rng, today, order_number, customer_id, status, days_ago, items)
        order_number += 1

    conn.commit()


def build_database(db_path: str | Path = DEFAULT_DB_PATH, today: date | None = None) -> None:
    """Create a fresh database file, replacing any existing one."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    conn = connect(db_path)
    try:
        create_schema(conn)
        seed(conn, today)
    finally:
        conn.close()


def main() -> None:
    build_database()
    conn = connect()
    try:
        customers = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        by_status = conn.execute(
            "SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY status"
        ).fetchall()
    finally:
        conn.close()

    print(f"Database created at {DEFAULT_DB_PATH}")
    print(f"  customers: {customers}")
    print(f"  products:  {products}")
    print(f"  orders:    {sum(count for _, count in by_status)}")
    for status, count in by_status:
        print(f"    {status}: {count}")


if __name__ == "__main__":
    main()
