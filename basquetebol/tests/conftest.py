import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["BASQ_DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app, follow_redirects=False) as c:
        yield c
