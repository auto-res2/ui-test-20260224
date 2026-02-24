"""
Inference script for SVEC prompt tuning experiment.
Runs either SVEC (two-stage) or standard Self-Consistency method.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

from src.model import create_model
from src.preprocess import (
    load_gsm8k,
    extract_answer_from_response,
    extract_confidence_from_response,
    extract_verification_fields,
)


def run_svec_method(
    problems: List[Dict],
    model,
    cfg: Any,
) -> List[Dict[str, Any]]:
    """
    Run SVEC (Self-Verified Efficient Consistency) method.
    Two-stage: (1) generate candidates, (2) verify, (3) weighted aggregation.
    
    Args:
        problems: List of problem dictionaries
        model: LLM model instance
        cfg: Configuration object
        
    Returns:
        List of prediction dictionaries
    """
    method_cfg = cfg.run.method
    k_gen = method_cfg.candidate_generation.k_gen
    k_ver = method_cfg.self_verification.k_ver
    
    # Aggregation weights
    alpha = method_cfg.aggregation.alpha
    beta = method_cfg.aggregation.beta
    gamma = method_cfg.aggregation.gamma
    delta = method_cfg.aggregation.delta
    epsilon = method_cfg.aggregation.epsilon
    
    predictions = []
    
    for problem in tqdm(problems, desc="SVEC inference"):
        question = problem["question"]
        
        # Stage 1: Generate K candidates
        candidates = []
        gen_prompt_template = method_cfg.candidate_generation.prompt_template
        
        for _ in range(k_gen):
            prompt = gen_prompt_template.format(
                question=question,
                max_steps=method_cfg.candidate_generation.max_steps,
            )
            
            response = model.generate(
                prompt=prompt,
                temperature=method_cfg.candidate_generation.temperature,
                max_output_tokens=method_cfg.candidate_generation.max_output_tokens,
            )
            
            # Extract answer and confidence
            answer = extract_answer_from_response(response)
            confidence = extract_confidence_from_response(response)
            length = model.count_tokens(response)
            
            candidates.append({
                "response": response,
                "answer": answer,
                "confidence": confidence,
                "length": length,
            })
        
        # Stage 2: Self-verification
        ver_prompt_template = method_cfg.self_verification.prompt_template
        
        for candidate in candidates[:k_ver]:  # Verify k_ver candidates
            ver_prompt = ver_prompt_template.format(
                question=question,
                solution=candidate["response"],
            )
            
            ver_response = model.generate(
                prompt=ver_prompt,
                temperature=method_cfg.self_verification.temperature,
                max_output_tokens=method_cfg.self_verification.max_output_tokens,
            )
            
            # Extract verification fields
            ver_fields = extract_verification_fields(ver_response)
            candidate["pass_rate"] = ver_fields["pass_rate"]
            candidate["contradiction"] = ver_fields["contradiction"]
            candidate["ver_confidence"] = ver_fields["confidence"]
            candidate["ver_response"] = ver_response
        
        # For unverified candidates, use defaults
        for candidate in candidates[k_ver:]:
            candidate["pass_rate"] = 0.5
            candidate["contradiction"] = False
            candidate["ver_confidence"] = candidate["confidence"]
            candidate["ver_response"] = ""
        
        # Stage 3: Weighted aggregation
        # w_i = exp(-alpha * length + gamma * pass_rate + beta * log(confidence + eps) - delta * contradiction)
        answer_weights = {}
        
        for candidate in candidates:
            if math.isnan(candidate["answer"]):
                continue  # Skip invalid answers
            
            ans = candidate["answer"]
            length = candidate["length"]
            pass_rate = candidate["pass_rate"]
            confidence = max(candidate["confidence"], candidate["ver_confidence"])
            contradiction = 1 if candidate["contradiction"] else 0
            
            # Compute weight
            log_weight = (
                -alpha * length
                + gamma * pass_rate
                + beta * math.log(confidence + epsilon)
                - delta * contradiction
            )
            weight = math.exp(log_weight)
            
            # Accumulate weight for this answer
            if ans not in answer_weights:
                answer_weights[ans] = 0.0
            answer_weights[ans] += weight
        
        # Select answer with maximum total weight
        if answer_weights:
            pred_answer = max(answer_weights.items(), key=lambda x: x[1])[0]
        else:
            pred_answer = float("nan")
        
        predictions.append({
            "idx": problem["idx"],
            "question": question,
            "gold_answer": problem["gold_answer"],
            "pred_answer": pred_answer,
            "candidates": candidates,
            "answer_weights": answer_weights,
        })
    
    return predictions


def run_self_consistency_method(
    problems: List[Dict],
    model,
    cfg: Any,
) -> List[Dict[str, Any]]:
    """
    Run standard Self-Consistency method with majority voting.
    
    Args:
        problems: List of problem dictionaries
        model: LLM model instance
        cfg: Configuration object
        
    Returns:
        List of prediction dictionaries
    """
    method_cfg = cfg.run.method
    k_gen = method_cfg.candidate_generation.k_gen
    
    predictions = []
    
    for problem in tqdm(problems, desc="Self-Consistency inference"):
        question = problem["question"]
        
        # Generate K candidates
        candidates = []
        gen_prompt_template = method_cfg.candidate_generation.prompt_template
        
        for _ in range(k_gen):
            prompt = gen_prompt_template.format(
                question=question,
                max_steps=method_cfg.candidate_generation.max_steps,
            )
            
            response = model.generate(
                prompt=prompt,
                temperature=method_cfg.candidate_generation.temperature,
                max_output_tokens=method_cfg.candidate_generation.max_output_tokens,
            )
            
            # Extract answer
            answer = extract_answer_from_response(response)
            length = model.count_tokens(response)
            
            candidates.append({
                "response": response,
                "answer": answer,
                "length": length,
            })
        
        # Majority voting
        answer_counts = {}
        for candidate in candidates:
            if math.isnan(candidate["answer"]):
                continue
            ans = candidate["answer"]
            answer_counts[ans] = answer_counts.get(ans, 0) + 1
        
        # Select most common answer
        if answer_counts:
            pred_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        else:
            pred_answer = float("nan")
        
        predictions.append({
            "idx": problem["idx"],
            "question": question,
            "gold_answer": problem["gold_answer"],
            "pred_answer": pred_answer,
            "candidates": candidates,
            "answer_counts": answer_counts,
        })
    
    return predictions


def compute_metrics(predictions: List[Dict]) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Dictionary of metrics
    """
    correct = 0
    total = 0
    
    for pred in predictions:
        gold = pred["gold_answer"]
        pred_ans = pred["pred_answer"]
        
        if not math.isnan(gold) and not math.isnan(pred_ans):
            # Check if answers match (with small tolerance for floating point)
            if abs(gold - pred_ans) < 1e-3:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def sanity_validation(
    predictions: List[Dict],
    metrics: Dict[str, float],
    cfg: Any,
) -> None:
    """
    Perform sanity validation checks and print verdict.
    
    Args:
        predictions: List of predictions
        metrics: Computed metrics
        cfg: Configuration object
    """
    # Check minimum samples
    num_samples = len(predictions)
    min_samples = 5
    
    if num_samples < min_samples:
        print(f"SANITY_VALIDATION: FAIL reason=insufficient_samples ({num_samples}<{min_samples})")
        return
    
    # Check that outputs are valid (not all NaN)
    valid_preds = sum(1 for p in predictions if not math.isnan(p["pred_answer"]))
    
    if valid_preds == 0:
        print(f"SANITY_VALIDATION: FAIL reason=all_predictions_invalid")
        return
    
    # Check that not all predictions are identical
    unique_answers = len(set(
        p["pred_answer"] for p in predictions 
        if not math.isnan(p["pred_answer"])
    ))
    
    if unique_answers == 1 and num_samples > 1:
        print(f"SANITY_VALIDATION: FAIL reason=all_predictions_identical")
        return
    
    # Check that metrics are finite
    if not all(math.isfinite(v) for v in metrics.values() if isinstance(v, float)):
        print(f"SANITY_VALIDATION: FAIL reason=non_finite_metrics")
        return
    
    # All checks passed
    print(f"SANITY_VALIDATION: PASS")
    
    # Print summary
    summary = {
        "samples": num_samples,
        "valid_predictions": valid_preds,
        "unique_answers": unique_answers,
        "accuracy": metrics["accuracy"],
    }
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")


def main():
    """Main inference entry point."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.config_path)
    results_dir = Path(args.results_dir)
    
    print(f"=== Running inference for {cfg.run.run_id} ===")
    print(f"Method: {cfg.run.method.name}")
    print(f"Mode: {cfg.mode}")
    print()
    
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        print(f"WandB run: {wandb.run.url}")
    else:
        print("WandB disabled")
    
    # Load dataset
    if cfg.run.inference.mode == "sanity":
        # Sanity check: use only a few samples
        start_idx = cfg.run.dataset.eval_start
        end_idx = start_idx + cfg.run.inference.sanity_samples
    else:
        # Full evaluation
        start_idx = cfg.run.dataset.eval_start
        end_idx = cfg.run.dataset.eval_end
    
    problems = load_gsm8k(
        cache_dir=cfg.cache_dir,
        split=cfg.run.dataset.split,
        subset=cfg.run.dataset.subset,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    
    print(f"Loaded {len(problems)} problems")
    print()
    
    # Initialize model
    model = create_model(OmegaConf.to_container(cfg.run.model, resolve=True))
    print()
    
    # Run inference based on method
    method_name = cfg.run.method.name
    
    if method_name == "svec":
        predictions = run_svec_method(problems, model, cfg)
    elif method_name == "self_consistency":
        predictions = run_self_consistency_method(problems, model, cfg)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    print()
    print(f"Generated {len(predictions)} predictions")
    
    # Compute metrics
    metrics = compute_metrics(predictions)
    
    print()
    print("=== Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Log to WandB
    if cfg.wandb.mode != "disabled":
        wandb.log(metrics)
        wandb.summary.update(metrics)
    
    # Save predictions
    predictions_path = results_dir / "predictions.json"
    with open(predictions_path, "w") as f:
        # Convert to serializable format
        serializable_preds = []
        for pred in predictions:
            serializable_pred = {
                "idx": pred["idx"],
                "question": pred["question"],
                "gold_answer": float(pred["gold_answer"]) if not math.isnan(pred["gold_answer"]) else None,
                "pred_answer": float(pred["pred_answer"]) if not math.isnan(pred["pred_answer"]) else None,
            }
            serializable_preds.append(serializable_pred)
        
        json.dump(serializable_preds, f, indent=2)
    
    print(f"\nSaved predictions to: {predictions_path}")
    
    # Save metrics
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved metrics to: {metrics_path}")
    
    # Sanity validation for sanity_check mode
    if cfg.mode == "sanity_check":
        print()
        sanity_validation(predictions, metrics, cfg)
    
    # Finish WandB
    if cfg.wandb.mode != "disabled":
        wandb.finish()
    
    print("\n=== Inference completed ===")


if __name__ == "__main__":
    main()
