"""
Evaluation script for comparing multiple runs.
Fetches results from WandB and generates comparison visualizations.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def fetch_run_from_wandb(
    entity: str,
    project: str,
    run_id: str,
) -> Dict[str, Any]:
    """
    Fetch run data from WandB by display name.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name (run_id)
        
    Returns:
        Dictionary with run config, summary, and history
    """
    api = wandb.Api()
    
    # Fetch runs with matching display name
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": run_id},
        order="-created_at",
    )
    
    if len(runs) == 0:
        raise ValueError(f"No run found with display name: {run_id}")
    
    # Get most recent run
    run = runs[0]
    
    print(f"Fetched run: {run.name} (id: {run.id})")
    
    # Get config, summary, and history
    config = run.config
    summary = run.summary._json_dict
    history = run.history()
    
    return {
        "run_id": run_id,
        "wandb_id": run.id,
        "config": config,
        "summary": summary,
        "history": history,
    }


def export_per_run_metrics(
    run_data: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Export per-run metrics to JSON and create figures.
    
    Args:
        run_data: Run data dictionary
        output_dir: Output directory for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export metrics.json
    metrics = {
        "run_id": run_data["run_id"],
        "summary": run_data["summary"],
    }
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported metrics: {metrics_path}")
    
    # Create per-run figure (if history data available)
    history = run_data["history"]
    
    if len(history) > 0 and "accuracy" in history.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot accuracy over steps (if available)
        if "_step" in history.columns:
            ax.plot(history["_step"], history["accuracy"], marker="o")
            ax.set_xlabel("Step")
        else:
            ax.plot(history.index, history["accuracy"], marker="o")
            ax.set_xlabel("Index")
        
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy - {run_data['run_id']}")
        ax.grid(True, alpha=0.3)
        
        fig_path = output_dir / "accuracy.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Created figure: {fig_path}")


def export_comparison_metrics(
    all_run_data: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Export aggregated comparison metrics.
    
    Args:
        all_run_data: List of run data dictionaries
        output_dir: Comparison output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate metrics
    metrics_by_run = {}
    
    for run_data in all_run_data:
        run_id = run_data["run_id"]
        summary = run_data["summary"]
        
        metrics_by_run[run_id] = {
            "accuracy": summary.get("accuracy", 0.0),
            "correct": summary.get("correct", 0),
            "total": summary.get("total", 0),
        }
    
    # Identify proposed vs baseline
    proposed_runs = [r for r in all_run_data if "proposed" in r["run_id"]]
    baseline_runs = [r for r in all_run_data if "comparative" in r["run_id"]]
    
    best_proposed = None
    best_baseline = None
    
    if proposed_runs:
        best_proposed = max(
            proposed_runs,
            key=lambda r: r["summary"].get("accuracy", 0.0)
        )
    
    if baseline_runs:
        best_baseline = max(
            baseline_runs,
            key=lambda r: r["summary"].get("accuracy", 0.0)
        )
    
    # Calculate gap
    gap = None
    if best_proposed and best_baseline:
        gap = (
            best_proposed["summary"].get("accuracy", 0.0) -
            best_baseline["summary"].get("accuracy", 0.0)
        )
    
    # Create aggregated metrics
    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed["run_id"] if best_proposed else None,
        "best_baseline": best_baseline["run_id"] if best_baseline else None,
        "gap": gap,
    }
    
    agg_path = output_dir / "aggregated_metrics.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Exported aggregated metrics: {agg_path}")


def create_comparison_figures(
    all_run_data: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Create comparison figures overlaying all runs.
    
    Args:
        all_run_data: List of run data dictionaries
        output_dir: Comparison output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Figure 1: Bar chart of final accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    
    run_ids = [r["run_id"] for r in all_run_data]
    accuracies = [r["summary"].get("accuracy", 0.0) for r in all_run_data]
    
    colors = []
    for run_id in run_ids:
        if "proposed" in run_id:
            colors.append("steelblue")
        else:
            colors.append("coral")
    
    bars = ax.bar(range(len(run_ids)), accuracies, color=colors)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison")
    ax.set_ylim(0, 1.0)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Proposed"),
        Patch(facecolor="coral", label="Baseline"),
    ]
    ax.legend(handles=legend_elements)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    fig_path = output_dir / "comparison_accuracy.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Created comparison figure: {fig_path}")
    
    # Figure 2: If history data available, plot accuracy over time/steps
    has_history = any(
        len(r["history"]) > 0 and "accuracy" in r["history"].columns
        for r in all_run_data
    )
    
    if has_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for run_data in all_run_data:
            history = run_data["history"]
            if len(history) > 0 and "accuracy" in history.columns:
                run_id = run_data["run_id"]
                
                if "_step" in history.columns:
                    x = history["_step"]
                    xlabel = "Step"
                else:
                    x = history.index
                    xlabel = "Index"
                
                linestyle = "-" if "proposed" in run_id else "--"
                ax.plot(x, history["accuracy"], label=run_id, linestyle=linestyle)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig_path = output_dir / "comparison_accuracy_over_time.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Created comparison figure: {fig_path}")


def main():
    """Main evaluation entry point."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--run-ids", type=str, required=True, help="JSON list of run IDs")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)
    
    print(f"=== Evaluating runs: {run_ids} ===")
    print()
    
    # Get WandB config from environment or first run's local config
    entity = os.getenv("WANDB_ENTITY", "a")
    project = os.getenv("WANDB_PROJECT", "a")
    
    print(f"WandB entity: {entity}")
    print(f"WandB project: {project}")
    print()
    
    # Fetch all runs
    all_run_data = []
    
    for run_id in run_ids:
        print(f"Fetching run: {run_id}")
        try:
            run_data = fetch_run_from_wandb(entity, project, run_id)
            all_run_data.append(run_data)
        except Exception as e:
            print(f"Warning: Failed to fetch run {run_id}: {e}")
            continue
    
    print()
    print(f"Successfully fetched {len(all_run_data)} / {len(run_ids)} runs")
    print()
    
    # Export per-run metrics and figures
    for run_data in all_run_data:
        run_id = run_data["run_id"]
        run_dir = results_dir / run_id
        
        print(f"Processing run: {run_id}")
        export_per_run_metrics(run_data, run_dir)
        print()
    
    # Export comparison metrics and figures
    if len(all_run_data) > 1:
        comparison_dir = results_dir / "comparison"
        
        print("Creating comparison visualizations...")
        export_comparison_metrics(all_run_data, comparison_dir)
        create_comparison_figures(all_run_data, comparison_dir)
        print()
    
    print("=== Evaluation completed ===")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
