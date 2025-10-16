# database.py

import sqlite3
import uuid
from typing import Optional, Tuple, List, Dict

DB_NAME = "receipts.db"


def init_db():
    """Create the database and tables if they don't exist."""
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    # Create the Stores table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stores (
        store_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        address TEXT,
        post_address TEXT,
        short_name TEXT,
        phone_number TEXT,
        org_number TEXT
    )
    """)

    # Create the Receipts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS receipts (
        receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        store_id INTEGER,
        date TEXT NOT NULL,
        total REAL,
        FOREIGN KEY (store_id) REFERENCES stores(store_id)
    )
    """)
    
    # Create the Items table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS items (
        item_id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT NOT NULL,
        article_number TEXT,
        price REAL,
        quantity REAL,
        total REAL,
        discount REAL,
        category TEXT,
        store_id INTEGER NOT NULL,
        receipt_id INTEGER,
        date TEXT NOT NULL,
        comparison_price REAL,
        comparison_price_unit TEXT,
        FOREIGN KEY (store_id) REFERENCES stores(store_id),
        FOREIGN KEY (receipt_id) REFERENCES receipts(receipt_id)
    )
    """)

    connection.commit()
    connection.close()

def add_receipt_to_db(
    store_id: Optional[int] = None,
    date: Optional[str] = None,
    total: Optional[float] = None,
    store_existed: Optional[bool] = False,
) -> Tuple[int, bool]:
    """
    Add a new receipt to the database

    Returns:
        tuple: (receipt_id, was_created) where was_created is True if new receipt was added
    """
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()
    
    print(store_existed, date, total)
    if store_existed:
        # Duplicate detection by date + total (only if both provided)
        if date is not None and total is not None:
            cursor.execute(
                "SELECT receipt_id FROM receipts WHERE date = ? AND total = ?",
                (date, total),
            )
            existing_receipt = cursor.fetchone()
            print(existing_receipt)
            if existing_receipt:
                connection.close()
                return existing_receipt[0], True
    
    # No duplicate found — insert new receipt
    cursor.execute(
        """
        INSERT INTO receipts (store_id, date, total)
        VALUES (?, ?, ?)
        """,
        (store_id, date, total),
    )

    connection.commit()
    receipt_id = cursor.lastrowid
    connection.close()

    return receipt_id, False

def add_store_to_db(
    name: Optional[str] = None,
    address: Optional[str] = None,
    post_address: Optional[str] = None,
    short_name: Optional[str] = None,
    phone_number: Optional[str] = None,
    org_number: Optional[str] = None
) -> Tuple[int, bool]:
    """
    Add a new store to the database if it doesn't exist.
    If the store already exists (based on name), return the existing store_id.

    Returns:
        tuple: (store_id, was_created) where was_created is True if new store was added
    """
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    # If name provided, check if name already exists
    if name:
        cursor.execute("SELECT store_id FROM stores WHERE name = ?", (name,))
        existing_store = cursor.fetchone()
        if existing_store:
            connection.close()
            return existing_store[0], True

    # Store doesn't exist, insert it
    cursor.execute(
        """
        INSERT INTO stores (name, address, post_address, short_name, phone_number, org_number)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, address, post_address, short_name, phone_number, org_number),
    )

    connection.commit()
    store_id = cursor.lastrowid
    connection.close()

    return store_id, False


def add_item_to_db(
    description: str,
    article_number: Optional[str],
    price: Optional[float],
    quantity: float,
    total: Optional[float],
    discount: float,
    category: Optional[str],
    store_id: int,
    receipt_id: int,
    purchase_date: str,
    comparison_price: Optional[float] = None,
    comparison_price_unit: Optional[str] = None,
) -> Tuple[int, bool]:
    """
    Add a new item to the database if it doesn't exist.
    If an identical item already exists, return the existing item_id.

    Returns:
        tuple: (item_id, was_created) where was_created is True if new item was added
    """
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    # Check if item already exists (matching all fields)
    cursor.execute(
        """
        SELECT item_id FROM items
        WHERE description = ?
          AND article_number IS ?
          AND price = ?
          AND quantity = ?
          AND total = ?
          AND discount = ?
          AND category IS ?
          AND store_id = ?
          AND receipt_id = ?
          AND date = ?
          AND comparison_price IS ?
          AND comparison_price_unit IS ?
        """,
        (
            description,
            article_number,
            price,
            quantity,
            total,
            discount,
            category,
            store_id,
            receipt_id,
            purchase_date,
            comparison_price,
            comparison_price_unit,
        ),
    )

    existing_item = cursor.fetchone()

    if existing_item:
        # Item already exists, return its ID
        connection.close()
        return existing_item[0], False

    # Item doesn't exist, insert it
    cursor.execute(
        """
        INSERT INTO items (description, article_number, price, quantity, total,
                          discount, category, store_id, receipt_id, date, comparison_price,
                          comparison_price_unit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            description,
            article_number,
            price,
            quantity,
            total,
            discount,
            category,
            store_id,
            receipt_id,
            purchase_date,
            comparison_price,
            comparison_price_unit,
        ),
    )

    connection.commit()
    item_id = cursor.lastrowid
    connection.close()

    return item_id, True


def get_all_receipts_from_db() -> List[Dict]:
    """Retrieve all receipts with store and items.

    Returns a list of receipts where each receipt is a dict:
    {
      "receipt_id": int,
      "store_id": int,
      "store_name": str,
      "date": str,
      "total": float,
      "items": [
        {
          "item_id": int,
          "description": str,
          "article_number": str,
          "price": float,
          "quantity": float,
          "total": float,
          "discount": float
        }, ...
      ]
    }
    """
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
    SELECT
        r.receipt_id,
        r.store_id,
        s.name AS store_name,
        r.date AS receipt_date,
        r.total AS receipt_total,
        i.item_id,
        i.description,
        i.article_number,
        i.price,
        i.quantity,
        i.total AS item_total,
        i.discount
    FROM receipts r
    LEFT JOIN stores s ON r.store_id = s.store_id
    LEFT JOIN items i ON r.receipt_id = i.receipt_id
    ORDER BY r.date DESC, s.name, r.receipt_id, i.item_id
    """)

    rows = cursor.fetchall()
    connection.close()

    receipts_map: Dict[int, Dict] = {}
    for row in rows:
        (
            receipt_id,
            store_id,
            store_name,
            receipt_date,
            receipt_total,
            item_id,
            description,
            article_number,
            price,
            quantity,
            item_total,
            discount,
        ) = row

        if receipt_id not in receipts_map:
            receipts_map[receipt_id] = {
                "receipt_id": receipt_id,
                "store_id": store_id,
                "store_name": store_name,
                "date": receipt_date,
                "total": receipt_total,
                "items": []
            }

        # If there is an associated item row, append it
        if item_id is not None:
            receipts_map[receipt_id]["items"].append({
                "description": description,
                "article_number": article_number,
                "price": price,
                "quantity": quantity,
                "total": item_total,
                "discount": discount,
            })

    return list(receipts_map.values())

def get_number_of_receipts_in_db() -> int:
    """Get the total number of receipts in the database."""
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM receipts")
    count = cursor.fetchone()[0]
    
    connection.close()
    return count

def get_stats_from_db() -> Dict:
    """Get statistics about the database."""
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()
    
    # Get store count
    cursor.execute("SELECT COUNT(*) FROM stores")
    store_count = cursor.fetchone()[0]
    
    # Get receipt count
    cursor.execute("SELECT COUNT(*) FROM receipts")
    receipt_count = cursor.fetchone()[0]
    
    # Get item count
    cursor.execute("SELECT COUNT(*) FROM items")
    item_count = cursor.fetchone()[0]
    
    # Get date range
    cursor.execute("SELECT MIN(date), MAX(date) FROM items")
    date_range = cursor.fetchone()
    
    connection.close()
    
    return {
        "store_count": store_count,
        "receipt_count": receipt_count,
        "item_count": item_count,
        "earliest_date": date_range[0],
        "latest_date": date_range[1]
    }


def get_recent_items(limit: int = 50) -> List[Tuple]:
    """Get recent items from the database."""
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()
    
    cursor.execute("""
    SELECT s.name, i.date, i.description, i.price, i.quantity, i.total, i.discount
    FROM items i
    JOIN stores s ON i.store_id = s.store_id
    ORDER BY i.date DESC, i.item_id DESC
    LIMIT ?
    """, (limit,))
    
    items = cursor.fetchall()
    connection.close()
    
    return items


def clear_db():
    """Clear all data from database."""
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()
    cursor.execute("DELETE FROM items")
    cursor.execute("DELETE FROM stores")
    cursor.execute("DELETE FROM receipts")
    connection.commit()
    connection.close()


def get_stores() -> List[Tuple]:
    """Get all stores from the database."""
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()
    
    cursor.execute("SELECT store_id, name, post_address FROM stores ORDER BY name")
    stores = cursor.fetchall()
    connection.close()
    
    return stores


def get_items_by_store(store_id: int) -> List[Tuple]:
    """Get all items for a specific store."""
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()
    
    cursor.execute("""
    SELECT description, article_number, price, quantity, total, discount, date
    FROM items
    WHERE store_id = ?
    ORDER BY date DESC
    """, (store_id,))
    
    items = cursor.fetchall()
    connection.close()
    
    return items


def search_items(query: str) -> List[Tuple]:
    """Search for items by description."""
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()
    
    cursor.execute("""
    SELECT s.name, i.date, i.description, i.price, i.quantity, i.total, i.discount
    FROM items i
    JOIN stores s ON i.store_id = s.store_id
    WHERE i.description LIKE ?
    ORDER BY i.date DESC
    LIMIT 100
    """, (f"%{query}%",))
    
    items = cursor.fetchall()
    connection.close()
    
    return items
    

def view_db():
    """Return database contents as formatted text."""
    stats = get_stats_from_db()
    items = get_recent_items(50)

    output = "**Database Statistics**\n"
    
    if stats['earliest_date']:
        output += f"- Date Range: {stats['earliest_date']} to {stats['latest_date']}\n"
        
    output += f"- Total Stores: {stats['store_count']}\n"
    output += f"- Total Receipts: {stats['receipt_count']}\n"
    output += f"- Total Items: {stats['item_count']}\n"
    
    output += "\n**Recent Items (last 50):**\n\n"

    for item in items:
        store, date, desc, price, qty, total, discount = item
        discount_str = f" (discount: {discount} kr)" if discount else ""
        output += f"- **{store}** ({date}): {desc} - {qty}x @ {price} kr = {total} kr{discount_str}\n"

    return output    
    