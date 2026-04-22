"""
conftest.py — pytest configuration for the azure_upload_function test suite.

Adds the azure_upload_function directory to sys.path so that
`from services.xxx import yyy` works when pytest is run from either:
  - the repo root:          pytest azure_upload_function/tests/
  - the function directory: pytest tests/
"""
import sys
import os

# Ensure the azure_upload_function directory is on the path so service
# imports resolve correctly regardless of where pytest is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
