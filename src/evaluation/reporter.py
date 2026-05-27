"""
Results reporting utilities for CarbonBench.

Generates formatted result tables for:
- LaTeX paper tables
- CSV export
- Console display
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ResultsReporter:
    """
    Collects and formats benchmark results across models, tasks, and domains.

    Usage:
        reporter = ResultsReporter()
        reporter.add_result("T1", "EPA", "RandomForest", {"mae": 1234, "rmse": 5678})
        reporter.add_result("T1", "NYC", "RandomForest", {"mae": 456, "rmse": 789})
        reporter.generate_csv_report("results/benchmark_results.csv")
        print(reporter.generate_latex_table("T1"))
    """

    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def add_result(
        self,
        task: str,
        domain: str,
        model: str,
        metrics: Dict[str, float],
        horizon: Optional[int] = None,
        split: str = "test",
    ) -> None:
        """
        Add a result entry.

        Args:
            task: Task name (e.g., 'T1', 'T2', 'T3').
            domain: Domain name (e.g., 'EPA', 'NYC', 'EU_ETS', 'Synthetic').
            model: Model name (e.g., 'RandomForest', 'LSTM').
            metrics: Dictionary of metric_name -> value.
            horizon: For T2, the prediction horizon (1, 2, or 3).
            split: Data split (e.g., 'test', 'val').
        """
        record = {
            "task": task,
            "domain": domain,
            "model": model,
            "split": split,
        }
        if horizon is not None:
            record["horizon"] = horizon
        record.update(metrics)
        self._records.append(record)

    def get_dataframe(self) -> pd.DataFrame:
        """Return all results as a DataFrame."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame(self._records)

    def get_summary(
        self,
        task: Optional[str] = None,
        domain: Optional[str] = None,
        split: str = "test",
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get a summary table filtered by task, domain, and split.

        Args:
            task: Filter to specific task (None = all tasks).
            domain: Filter to specific domain (None = all domains).
            split: Data split to show ('test' or 'val').
            metrics: List of metric names to include. If None, uses all.

        Returns:
            DataFrame with models as rows, metrics as columns.
        """
        df = self.get_dataframe()
        if df.empty:
            return df

        if task:
            df = df[df["task"] == task]
        if domain:
            df = df[df["domain"] == domain]
        if split:
            df = df[df["split"] == split]

        # Determine metric columns
        non_metric_cols = {"task", "domain", "model", "split", "horizon"}
        if metrics is None:
            metric_cols = [c for c in df.columns if c not in non_metric_cols]
        else:
            metric_cols = [c for c in metrics if c in df.columns]

        keep_cols = ["model"] + (["horizon"] if "horizon" in df.columns else []) + metric_cols
        keep_cols = [c for c in keep_cols if c in df.columns]

        return df[keep_cols].reset_index(drop=True)

    def generate_csv_report(self, output_path: Union[str, Path]) -> None:
        """
        Save all results to a CSV file.

        Args:
            output_path: Path for the output CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.get_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    def generate_latex_table(
        self,
        task: str,
        domain: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        caption: str = "",
        label: str = "",
        bold_best: bool = True,
        lower_is_better: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a LaTeX table for a given task.

        Args:
            task: Task to generate table for ('T1', 'T2', 'T3').
            domain: Optional domain filter.
            metrics: Metrics to include. If None, uses MAE, RMSE, MAPE, R², NRMSE.
            caption: LaTeX table caption.
            label: LaTeX table label for cross-referencing.
            bold_best: If True, bold the best value in each metric column.
            lower_is_better: Metrics where lower = better. If None, uses defaults.

        Returns:
            String containing valid LaTeX table code.
        """
        if metrics is None:
            metrics = ["mae", "rmse", "mape", "r2", "nrmse"]

        if lower_is_better is None:
            lower_is_better = ["mae", "rmse", "mape", "nrmse", "log_mae"]

        df = self.get_summary(task=task, domain=domain, metrics=metrics)

        if df.empty:
            return f"% No results found for task={task}, domain={domain}\n"

        # Format caption and label
        if not caption:
            caption = f"CarbonBench {task} Results"
            if domain:
                caption += f" on {domain} domain"
        if not label:
            label = f"tab:{task.lower()}_results"
            if domain:
                label += f"_{domain.lower()}"

        # Build header
        n_metrics = len([m for m in metrics if m in df.columns])
        col_spec = "l" + "r" * n_metrics
        metric_headers = " & ".join(
            m.upper().replace("_", "\\_") for m in metrics if m in df.columns
        )

        lines = [
            "\\begin{table}[htbp]",
            "  \\centering",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            f"  \\begin{{tabular}}{{{col_spec}}}",
            "    \\toprule",
            f"    Model & {metric_headers} \\\\",
            "    \\midrule",
        ]

        # Find best values for bolding
        best_vals = {}
        for metric in metrics:
            if metric not in df.columns:
                continue
            vals = df[metric].dropna()
            if vals.empty:
                continue
            if metric in lower_is_better:
                best_vals[metric] = vals.min()
            else:
                best_vals[metric] = vals.max()

        # Add data rows
        for _, row in df.iterrows():
            model_name = str(row["model"]).replace("_", "\\_")
            cells = [model_name]

            for metric in metrics:
                if metric not in df.columns:
                    continue
                val = row.get(metric, np.nan)
                if np.isnan(val):
                    cells.append("--")
                else:
                    # Format based on magnitude
                    formatted = _format_metric_value(metric, val)
                    if bold_best and metric in best_vals and abs(val - best_vals[metric]) < 1e-10:
                        formatted = f"\\textbf{{{formatted}}}"
                    cells.append(formatted)

            lines.append(f"    {' & '.join(cells)} \\\\")

        lines.extend([
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    def generate_multi_domain_latex_table(
        self,
        task: str,
        domains: List[str],
        metrics: Optional[List[str]] = None,
        caption: str = "",
        label: str = "",
    ) -> str:
        """
        Generate a LaTeX table showing results across multiple domains.

        Args:
            task: Task name.
            domains: List of domain names.
            metrics: Metrics to show per domain.
            caption: Table caption.
            label: Table label.

        Returns:
            LaTeX table string.
        """
        if metrics is None:
            metrics = ["mae", "rmse", "r2"]

        if not caption:
            caption = f"CarbonBench {task} Results Across Domains"
        if not label:
            label = f"tab:{task.lower()}_multi_domain"

        df_all = self.get_dataframe()
        if df_all.empty:
            return f"% No results found for task={task}\n"

        df_filtered = df_all[(df_all["task"] == task) & (df_all["split"] == "test")]

        # Pivot: rows = models, column groups = (domain, metric)
        models = sorted(df_filtered["model"].unique())

        # Build column spec
        n_cols = len(domains) * len(metrics)
        col_spec = "l" + "r" * n_cols
        domain_headers = " & ".join(
            f"\\multicolumn{{{len(metrics)}}}{{c}}{{{d}}}" for d in domains
        )
        metric_sub_headers = " & ".join(
            [m.upper() for m in metrics] * len(domains)
        )

        lines = [
            "\\begin{table*}[htbp]",
            "  \\centering",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            f"  \\begin{{tabular}}{{{col_spec}}}",
            "    \\toprule",
            f"    Model & {domain_headers} \\\\",
        ]

        # Cmidrule separators for domain columns
        cmidrules = []
        for i, _ in enumerate(domains):
            start = 2 + i * len(metrics)
            end = start + len(metrics) - 1
            cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        lines.append("    " + " ".join(cmidrules))
        lines.append(f"    & {metric_sub_headers} \\\\")
        lines.append("    \\midrule")

        for model in models:
            model_row = df_filtered[df_filtered["model"] == model]
            cells = [model.replace("_", "\\_")]

            for domain in domains:
                domain_row = model_row[model_row["domain"] == domain]
                for metric in metrics:
                    if domain_row.empty or metric not in domain_row.columns:
                        cells.append("--")
                    else:
                        val = domain_row[metric].iloc[0]
                        cells.append(_format_metric_value(metric, val))

            lines.append(f"    {' & '.join(cells)} \\\\")

        lines.extend([
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table*}",
        ])

        return "\n".join(lines)

    def print_summary(
        self,
        task: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> None:
        """Print a formatted summary to stdout."""
        df = self.get_summary(task=task, domain=domain)
        if df.empty:
            print("No results available.")
            return

        header = f"\n{'='*60}"
        if task:
            header += f"\nTask: {task}"
        if domain:
            header += f"  |  Domain: {domain}"
        print(header)
        print(df.to_string(index=False, float_format="%.2f"))
        print("=" * 60)


def _format_metric_value(metric: str, val: float) -> str:
    """Format a metric value for LaTeX display."""
    if np.isnan(val) or np.isinf(val):
        return "--"

    metric_lower = metric.lower()
    if metric_lower in ("r2",):
        return f"{val:.3f}"
    elif metric_lower in ("mape", "nrmse"):
        return f"{val:.2f}"
    elif metric_lower in ("mae", "rmse"):
        # Use scientific notation for very large/small values
        if abs(val) >= 1e6:
            return f"{val:.2e}"
        elif abs(val) >= 1000:
            return f"{val:,.0f}"
        else:
            return f"{val:.1f}"
    elif metric_lower == "log_mae":
        return f"{val:.3f}"
    else:
        return f"{val:.4f}"
