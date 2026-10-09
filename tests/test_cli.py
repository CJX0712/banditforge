"""CLI smoke test."""

from cli import main


def test_cli_benchmark_smoke():
    data = main(["benchmark", "--seeds", "2", "--rounds", "200"])
    assert isinstance(data, dict)
    assert "methods" in data
    assert "BanditFuse" in data["methods"]
    assert data["determinism_ok"] is True


def test_cli_simulate_smoke():
    data = main(["simulate", "--seed", "1000", "--rounds", "200"])
    assert "cumulative_regret" in data
    assert data["cumulative_regret"] > 0
