import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)


@pytest.fixture(scope="session")
def artifacts():
    """Tests that inspect trained models or saved splits need the pipeline to have run."""
    needed = ["models/models.pkl", "data/train.csv", "data/validation.csv", "data/test.csv"]
    missing = [p for p in needed if not (ROOT / p).exists()]
    assert not missing, f"missing {missing} -- run ./run_all.sh before the tests"
    return ROOT
