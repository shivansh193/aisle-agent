import pytest


@pytest.fixture(autouse=True)
def _layout_file(tmp_path, monkeypatch):
    """Keep the shop layout file out of the working directory."""
    monkeypatch.setenv("AISLE_LAYOUT", str(tmp_path / "layout.json"))
    monkeypatch.delenv("TOKEN_FACTORY_API_KEY", raising=False)     # no real Token Factory calls from tests
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setenv("TOKEN_FACTORY_NO_DOTENV", "1")


import os as _os
import sys as _sys

_VJEV = _os.path.join(_os.path.dirname(__file__), "..")                    # repo root: aisle_agent and vendored vjev
if _os.path.isdir(_VJEV) and _os.path.abspath(_VJEV) not in _sys.path:
    _sys.path.insert(0, _os.path.abspath(_VJEV))
