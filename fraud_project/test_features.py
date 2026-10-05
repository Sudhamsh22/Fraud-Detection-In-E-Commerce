"""test_features.py (root level discovery)
Delegates to tests/test_features.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tests.test_features import *
