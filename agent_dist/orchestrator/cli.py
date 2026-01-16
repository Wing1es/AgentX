import argparse
import sqlite3
import json
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "memory.db"

def list_traces(args):
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Check table existence
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='traces'")
            if not cursor.fetchone():
                print("'traces' table not found. Run the orchestrator to initialize it.")
                return

            cursor.execute("""
                SELECT id, session_id, timestamp 
                FROM traces 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (args.limit,))
            
            rows = cursor.fetchall()
            
            if not rows:
                print("No traces found.")
                return

            print(f"{'ID':<5} {'TIMESTAMP':<25} {'SESSION ID'}")
            print("-" * 60)
            for r in rows:
                ts = datetime.fromtimestamp(r[2]).strftime('%Y-%m-%d %H:%M:%S')
                print(f"{r[0]:<5} {ts:<25} {r[1]}")

    except Exception as e:
        print(f"Error reading database: {e}")

def show_trace(args):
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Try to query by ID (int) or Session UUID (str)
            if args.identifier.isdigit():
                query = "SELECT trace_data FROM traces WHERE id = ?"
            else:
                query = "SELECT trace_data FROM traces WHERE session_id = ?"

            cursor.execute(query, (args.identifier,))
            row = cursor.fetchone()
            
            if not row:
                print(f"Trace '{args.identifier}' not found.")
                return

            trace_json = row[0]
            # Parse and re-dump for pretty printing
            data = json.loads(trace_json)
            print(json.dumps(data, indent=2))

    except Exception as e:
        print(f"Error reading database: {e}")

def main():
    parser = argparse.ArgumentParser(description="Agentic Orchestrator Trace Inspector")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List recent traces")
    list_parser.add_argument("-n", "--limit", type=int, default=10, help="Number of traces to show")
    list_parser.set_defaults(func=list_traces)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show full trace details")
    show_parser.add_argument("identifier", help="Trace ID (integer) or Session ID (UUID)")
    show_parser.set_defaults(func=show_trace)

    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
