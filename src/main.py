"""
Main orchestrator for SVEC prompt tuning experiment.
Handles inference-only tasks with mode overrides (main, sanity_check, pilot).
"""

import os
import sys
import subprocess
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Orchestrate a single run_id for inference-only task.
    
    Mode overrides:
    - sanity_check: minimal samples, online wandb, separate namespace
    - main: full dataset, online wandb
    """
    
    # Print configuration
    print(f"=== Starting run: {cfg.run.run_id} ===")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method.name}")
    print(f"Model: {cfg.run.model.name}")
    print(f"Dataset: {cfg.run.dataset.name}")
    print()
    
    # Apply mode-specific overrides
    if cfg.mode == "sanity_check":
        # Override for lightweight sanity check
        OmegaConf.set_struct(cfg, False)  # Allow adding new keys
        cfg.run.inference.mode = "sanity"
        cfg.wandb.project = f"{cfg.wandb.project}-sanity"
        print(f"[SANITY_CHECK MODE] Using {cfg.run.inference.sanity_samples} samples")
        print(f"[SANITY_CHECK MODE] WandB project: {cfg.wandb.project}")
    elif cfg.mode == "main":
        OmegaConf.set_struct(cfg, False)
        cfg.run.inference.mode = "full"
        cfg.wandb.mode = "online"
    
    # Create results directory
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    print()
    
    # Save resolved config
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    print(f"Saved config to: {config_path}")
    
    # This is an inference-only task, so invoke inference.py
    print("=== Launching inference.py ===")
    
    # Build command
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.inference",
        f"--config-path={config_path}",
        f"--results-dir={results_dir}",
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run inference as subprocess
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path.cwd(),
            env=os.environ.copy()
        )
        print(f"\n=== Inference completed successfully (exit code: {result.returncode}) ===")
    except subprocess.CalledProcessError as e:
        print(f"\n=== Inference failed (exit code: {e.returncode}) ===", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
