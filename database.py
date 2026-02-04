"""
Database module for medical reports storage and retrieval.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

DATABASE_PATH = "reports.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory enabled."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Create reports table for storing medical transcription records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            procedure_type TEXT NOT NULL,
            specialty TEXT NOT NULL,
            report_name TEXT,
            report_text TEXT NOT NULL,
            keywords TEXT,
            source TEXT NOT NULL,
            is_deidentified BOOLEAN DEFAULT TRUE,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create generated_reports table for Track 2 (AI-generated reports)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS generated_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            procedure_type TEXT NOT NULL,
            surgeon_inputs TEXT NOT NULL,
            generated_report TEXT NOT NULL,
            user_rating INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def add_report(
    procedure_type: str,
    specialty: str,
    report_text: str,
    source: str,
    report_name: Optional[str] = None,
    keywords: Optional[str] = None,
    is_deidentified: bool = True
) -> int:
    """
    Add a new report to the database.

    Returns the ID of the inserted report.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO reports (procedure_type, specialty, report_name, report_text, keywords, source, is_deidentified)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (procedure_type, specialty, report_name, report_text, keywords, source, is_deidentified))

    report_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return report_id


def get_report(report_id: int) -> Optional[Dict[str, Any]]:
    """Get a report by its ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def search_reports(
    specialty: Optional[str] = None,
    procedure_type: Optional[str] = None,
    keyword: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Search reports with optional filters.

    Args:
        specialty: Filter by specialty (partial match)
        procedure_type: Filter by procedure type (partial match)
        keyword: Search in keywords field (partial match)
        source: Filter by source (exact match)
        limit: Maximum number of results to return

    Returns a list of matching reports.
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM reports WHERE 1=1"
    params = []

    if specialty:
        query += " AND specialty LIKE ?"
        params.append(f"%{specialty}%")

    if procedure_type:
        query += " AND procedure_type LIKE ?"
        params.append(f"%{procedure_type}%")

    if keyword:
        query += " AND keywords LIKE ?"
        params.append(f"%{keyword}%")

    if source:
        query += " AND source = ?"
        params.append(source)

    query += " LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def delete_report(report_id: int) -> bool:
    """
    Delete a report by its ID.

    Returns True if a report was deleted, False otherwise.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return deleted


def get_all_reports(limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
    """Get all reports with pagination."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM reports LIMIT ? OFFSET ?", (limit, offset))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_report_count_by_source() -> Dict[str, int]:
    """Get the count of reports grouped by source."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT source, COUNT(*) as count FROM reports GROUP BY source")
    rows = cursor.fetchall()
    conn.close()

    return {row['source']: row['count'] for row in rows}


# Generated reports functions for Track 2

def add_generated_report(
    procedure_type: str,
    surgeon_inputs: Dict[str, Any],
    generated_report: str,
    user_rating: Optional[int] = None
) -> int:
    """
    Add a new generated report to the database.

    Args:
        procedure_type: The type of procedure
        surgeon_inputs: Dictionary of surgeon inputs (will be stored as JSON)
        generated_report: The AI-generated report text
        user_rating: Optional user rating (1-5)

    Returns the ID of the inserted generated report.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO generated_reports (procedure_type, surgeon_inputs, generated_report, user_rating)
        VALUES (?, ?, ?, ?)
    """, (procedure_type, json.dumps(surgeon_inputs), generated_report, user_rating))

    report_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return report_id


def get_generated_report(report_id: int) -> Optional[Dict[str, Any]]:
    """Get a generated report by its ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM generated_reports WHERE id = ?", (report_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        result = dict(row)
        result['surgeon_inputs'] = json.loads(result['surgeon_inputs'])
        return result
    return None


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")
    print(f"Database file: {DATABASE_PATH}")
