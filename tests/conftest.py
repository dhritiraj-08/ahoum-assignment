"""
Shared pytest configuration.

Makes sure `src` is importable regardless of where/how pytest is invoked
from (project root, tests/ directory, an IDE test runner, etc.) -- mirrors
the same sys.path pattern used in main.py, app.py, and the notebook.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
