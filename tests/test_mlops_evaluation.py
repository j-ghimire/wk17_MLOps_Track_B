import pandas as pd

from mlops.evaluate import run_configuration_comparison


def test_configuration_comparison_tracks_runs_and_artifacts(tmp_path):
    tracking_uri = (tmp_path / "mlruns").as_uri()
    result = run_configuration_comparison(
        tracking_uri=tracking_uri,
        reports_dir=tmp_path / "reports",
    )

    comparison = pd.read_csv(result["comparison_path"])
    assert set(comparison["configuration"]) == {"fast", "balanced", "thorough"}
    assert comparison["task_success_rate"].between(0, 1).all()
    assert (tmp_path / "reports/agent_query_evaluations.csv").exists()
    assert (tmp_path / "reports/evidently_agent_configuration_drift.html").exists()
