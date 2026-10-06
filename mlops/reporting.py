from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Create a data drift report comparing agent evaluation telemetry."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    from evidently import Report
    from evidently.presets import DataDriftPreset

    snapshot = Report(metrics=[DataDriftPreset()]).run(
        reference_data=reference_df,
        current_data=current_df,
    )
    snapshot.save_html(str(output))
    return str(output)
