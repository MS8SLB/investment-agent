"""Configuração comum dos testes."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(autouse=True)
def fast_password_hashing(monkeypatch):
    """PBKDF2 com poucas iterações nos testes (a produção usa 600 000)."""
    from minibasket import auth
    monkeypatch.setattr(auth, "ITERATIONS", 1000)
