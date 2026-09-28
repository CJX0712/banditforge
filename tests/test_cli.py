"""CLI smoke tests."""
import json

import pytest

from banditforge.cli import main


def test_cli_demo(capsys, tmp_path):
    out = tmp_path / "demo.json"
    rc = main(["demo", "--output", str(out)])
    assert rc in (0, 1)  # gates may fail on the tiny config; must not crash
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "gates" in data and "policy_benchmark" in data


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert "banditforge" in capsys.readouterr().out


def test_cli_benchmark_small(tmp_path, capsys):
    import os
    env = {"ENV_BF_SEED": "9", "ENV_BF_N_LOG": "300", "ENV_BF_N_MC": "20000",
           "ENV_BF_T_STEPS": "500", "ENV_BF_N_REPEAT": "4"}
    orig = dict(os.environ)
    os.environ.update(env)
    try:
        out = tmp_path / "bench.json"
        rc = main(["benchmark", "--output", str(out)])
    finally:
        os.environ.clear()
        os.environ.update(orig)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["seed"] == 9
    assert rc in (0, 1)
    assert "OPE benchmark" in capsys.readouterr().out
