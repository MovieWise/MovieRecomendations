import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = PROJECT_ROOT / "service" / "backend"
SRC_ROOT = PROJECT_ROOT / "src"

for path in [BACKEND_ROOT, SRC_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

