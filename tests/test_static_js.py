"""Every page script must parse: a syntax error in one file silently disables the whole UI piece."""
import os
import re
import shutil
import subprocess
import tempfile

import pytest

STATIC = os.path.join(os.path.dirname(__file__), "..", "aisle_agent", "static")
node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="node not installed")
def test_page_scripts_parse():
    scripts = [(f, open(os.path.join(STATIC, f), encoding="utf-8").read()) for f in os.listdir(STATIC) if f.endswith(".js")]
    for f in os.listdir(STATIC):
        if f.endswith(".html"):
            html = open(os.path.join(STATIC, f), encoding="utf-8").read()
            scripts += [(f"{f}#{k}", js) for k, js in enumerate(re.findall(r"<script>(.*?)</script>", html, re.S))]
    for name, js in scripts:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as t:
            t.write(js)
        r = subprocess.run([node, "--check", t.name], capture_output=True, text=True)
        os.unlink(t.name)
        assert r.returncode == 0, f"{name}: {r.stderr[-300:]}"
