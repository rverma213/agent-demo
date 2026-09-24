import sys
from pathlib import Path

# Make the project root importable (tools.py, agent.py) when running `pytest`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
