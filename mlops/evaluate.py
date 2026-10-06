from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import pandas as pd

from app.assistant import AgentConfig, answer_locally
from mlops.reporting import generate_drift_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENT_NAME = "part-b-agent-configuration-comparison"

EVALUATION_CASES = [
    ("I forgot my password and did not get the reset email", "ACCOUNT"),
    ("My card payment was declined and I need to know what to check next", "CARD"),
    ("My card was stolen and I want to freeze it", "CARD"),
    ("I need help with withdrawals and ATM fees", "CASH_WITHDRAWAL"),
    ("My cash deposit balance has not updated and I kept the receipt", "CASH_DEPOSIT"),
    ("My transfer is pending and I need to check the recipient", "TRANSFER"),
    ("How can I check the status of a merchant refund?", "REFUND"),
    ("How do I track the delivery of my card?", "CARD_DELIVERY"),
    ("How do I top up my account?", "TOP_UP"),
    ("I need help with something else", "CLARIFY"),
    ("Tell me the special bank policy for my account", "CLARIFY"),
]

CONFIGURATIONS = {
    "fast": AgentConfig(retrieval_top_k=1, min_evidence_score=0.05, max_iterations=1),
    "balanced": AgentConfig(retrieval_top_k=3, min_evidence_score=0.18, max_iterations=2),
    "thorough": AgentConfig(retrieval_top_k=5, min_evidence_score=0.12, max_iterations=3),
}

_ALLOWED_ACTIONS = {
    "retrieve",
    "search_again",
    "verify",
    "answer_from_search",
    "ask_user",
    "guardrail",
    "llm_verify",
}


def _evaluate_configuration(name: str, config: AgentConfig) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for query_id, (prompt, expected) in enumerate(EVALUATION_CASES, start=1):
        result = answer_locally(prompt, config)
        asks_for_clarification = result.intent == "ask_for_clarification"
        if expected == "CLARIFY":
            completed = asks_for_clarification
        else:
            completed = result.category == expected and not asks_for_clarification

        actions = [step.action for step in result.steps]
        valid_actions = bool(actions) and all(action in _ALLOWED_ACTIONS for action in actions)
        estimated_tokens = sum(
            len((step.summary or "").split())
            + sum(len(evidence.split()) for evidence in step.evidence)
            for step in result.steps
        )
        rows.append(
            {
                "configuration": name,
                "query_id": query_id,
                "expected_category": expected,
                "predicted_category": result.category,
                "intent": result.intent,
                "completed": int(completed),
                "asked_for_clarification": int(asks_for_clarification),
                "valid_tool_actions": int(valid_actions),
                "trajectory_length": len(result.steps),
                "confidence": result.confidence,
                "evidence_score": max((score for _, score in result.sources), default=0.0),
                "estimated_context_words": estimated_tokens,
                "actions": actions,
                "prompt": prompt,
            }
        )

    detail_df = pd.DataFrame(rows)
    non_clarification_cases = detail_df[detail_df["expected_category"] != "CLARIFY"]
    summary: dict[str, object] = {
        "configuration": name,
        "query_count": len(detail_df),
        "task_success_rate": float(detail_df["completed"].mean()),
        "route_accuracy": float(
            (non_clarification_cases["predicted_category"] == non_clarification_cases["expected_category"]).mean()
        ),
        "safe_clarification_rate": float(
            detail_df.loc[detail_df["expected_category"] == "CLARIFY", "asked_for_clarification"].mean()
        ),
        "valid_tool_action_rate": float(detail_df["valid_tool_actions"].mean()),
        "mean_trajectory_length": float(detail_df["trajectory_length"].mean()),
        "mean_estimated_context_words": float(detail_df["estimated_context_words"].mean()),
        "retrieval_top_k": config.retrieval_top_k,
        "min_evidence_score": config.min_evidence_score,
        "max_iterations": config.max_iterations,
    }
    return summary, rows


def run_configuration_comparison(
    tracking_uri: str | None = None,
    reports_dir: str | Path | None = None,
) -> dict[str, object]:
    """Track agent configuration comparisons and generate an Evidently report."""
    output_dir = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if tracking_uri is None:
        tracking_uri = (PROJECT_ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    summaries: list[dict[str, object]] = []
    evaluations: list[dict[str, object]] = []
    for name, config in CONFIGURATIONS.items():
        summary, rows = _evaluate_configuration(name, config)
        with mlflow.start_run(run_name=f"agent-config-{name}"):
            mlflow.log_params(
                {
                    "configuration": name,
                    "retrieval_top_k": config.retrieval_top_k,
                    "min_evidence_score": config.min_evidence_score,
                    "max_iterations": config.max_iterations,
                    "evaluation_query_count": len(EVALUATION_CASES),
                }
            )
            for metric in (
                "task_success_rate",
                "route_accuracy",
                "safe_clarification_rate",
                "valid_tool_action_rate",
                "mean_trajectory_length",
                "mean_estimated_context_words",
            ):
                mlflow.log_metric(metric, float(summary[metric]))
            mlflow.log_dict({"evaluation_cases": rows}, "evaluation_results.json")
        summaries.append(summary)
        evaluations.extend(rows)

    comparison_df = pd.DataFrame(summaries)
    evaluation_df = pd.DataFrame(evaluations)
    comparison_path = output_dir / "agent_configuration_comparison.csv"
    evaluation_path = output_dir / "agent_query_evaluations.csv"
    comparison_df.to_csv(comparison_path, index=False)
    evaluation_df.to_csv(evaluation_path, index=False)

    selected = max(
        summaries,
        key=lambda row: (
            float(row["task_success_rate"]),
            float(row["route_accuracy"]),
            float(row["safe_clarification_rate"]),
            -float(row["mean_trajectory_length"]),
        ),
    )
    reference_df = evaluation_df[evaluation_df["configuration"] == "fast"].drop(
        columns=["configuration", "query_id", "intent", "actions", "prompt"]
    )
    current_df = evaluation_df[evaluation_df["configuration"] == selected["configuration"]].drop(
        columns=["configuration", "query_id", "intent", "actions", "prompt"]
    )
    report_path = generate_drift_report(
        reference_df,
        current_df,
        output_dir / "evidently_agent_configuration_drift.html",
    )

    return {
        "runs": summaries,
        "selected_configuration": selected["configuration"],
        "comparison_path": str(comparison_path),
        "evaluation_path": str(evaluation_path),
        "report_path": report_path,
        "tracking_uri": tracking_uri,
    }


if __name__ == "__main__":
    result = run_configuration_comparison()
    print(f"Selected configuration: {result['selected_configuration']}")
    print(f"Comparison table: {result['comparison_path']}")
    print(f"Evidently report: {result['report_path']}")
