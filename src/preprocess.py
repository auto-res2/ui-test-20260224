"""
Dataset handling for GSM8K math word problems.
Loads, caches, and preprocesses the dataset.
"""

import re
from typing import Dict, List, Any
from pathlib import Path
from datasets import load_dataset


def load_gsm8k(
    cache_dir: str,
    split: str = "test",
    subset: str = "main",
    start_idx: int = 0,
    end_idx: int = 200,
) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset from HuggingFace.
    
    Args:
        cache_dir: Directory to cache the dataset
        split: Dataset split (train/test)
        subset: Dataset subset (main)
        start_idx: Start index for slicing
        end_idx: End index for slicing
        
    Returns:
        List of problem dictionaries with 'question' and 'answer' keys
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading GSM8K dataset (split={split}, subset={subset})...")
    
    # Load dataset
    dataset = load_dataset("gsm8k", subset, split=split, cache_dir=cache_dir)
    
    # Slice to requested range
    dataset = dataset.select(range(start_idx, min(end_idx, len(dataset))))
    
    print(f"Loaded {len(dataset)} problems (indices {start_idx} to {start_idx + len(dataset) - 1})")
    
    # Convert to list of dicts
    problems = []
    for idx, item in enumerate(dataset):
        question = item["question"]
        answer = item["answer"]
        
        # Extract the final numeric answer from the answer string
        # GSM8K answers end with "#### <number>"
        gold_answer = extract_answer(answer)
        
        problems.append({
            "idx": start_idx + idx,
            "question": question,
            "answer_text": answer,
            "gold_answer": gold_answer,
        })
    
    return problems


def extract_answer(answer_text: str) -> float:
    """
    Extract the final numeric answer from GSM8K answer text.
    GSM8K format: "<<computation>> ... #### <final_answer>"
    
    Args:
        answer_text: Full answer text from GSM8K
        
    Returns:
        Numeric answer as float
    """
    # GSM8K answers end with "#### <number>"
    match = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", answer_text)
    if match:
        # Remove commas and convert to float
        answer_str = match.group(1).replace(",", "")
        try:
            return float(answer_str)
        except ValueError:
            print(f"Warning: Could not convert answer to float: {answer_str}")
            return float("nan")
    else:
        print(f"Warning: Could not extract answer from: {answer_text}")
        return float("nan")


def extract_answer_from_response(response: str) -> float:
    """
    Extract numeric answer from model response.
    Expected format: "... Answer: <number> ..."
    
    Args:
        response: Model response text
        
    Returns:
        Predicted numeric answer as float
    """
    # Look for "Answer: <number>" pattern
    match = re.search(r"Answer:\s*([-+]?[\d,]+(?:\.\d+)?)", response, re.IGNORECASE)
    if match:
        answer_str = match.group(1).replace(",", "")
        try:
            return float(answer_str)
        except ValueError:
            return float("nan")
    
    # Fallback: try to find any number at the end
    numbers = re.findall(r"[-+]?[\d,]+(?:\.\d+)?", response)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            return float("nan")
    
    return float("nan")


def extract_confidence_from_response(response: str) -> float:
    """
    Extract confidence score from model response.
    Expected format: "... Confidence: <0.0-1.0> ..."
    
    Args:
        response: Model response text
        
    Returns:
        Confidence score as float (0.0-1.0), or 0.5 if not found
    """
    match = re.search(r"Confidence:\s*(0?\.\d+|1\.0|1)", response, re.IGNORECASE)
    if match:
        try:
            conf = float(match.group(1))
            return max(0.0, min(1.0, conf))  # Clamp to [0, 1]
        except ValueError:
            pass
    
    # Default confidence if not specified
    return 0.5


def extract_verification_fields(response: str) -> Dict[str, Any]:
    """
    Extract structured fields from self-verification response.
    Expected format:
        ChecksPassed: <m>/<M>
        Contradiction: <yes/no>
        Confidence: <0.0-1.0>
    
    Args:
        response: Verification prompt response
        
    Returns:
        Dictionary with pass_rate, contradiction, confidence
    """
    result = {
        "pass_rate": 0.5,  # Default
        "contradiction": False,
        "confidence": 0.5,
    }
    
    # Extract ChecksPassed: m/M
    checks_match = re.search(r"ChecksPassed:\s*(\d+)\s*/\s*(\d+)", response, re.IGNORECASE)
    if checks_match:
        m = int(checks_match.group(1))
        M = int(checks_match.group(2))
        result["pass_rate"] = m / M if M > 0 else 0.5
    
    # Extract Contradiction: yes/no
    contradiction_match = re.search(r"Contradiction:\s*(yes|no)", response, re.IGNORECASE)
    if contradiction_match:
        result["contradiction"] = contradiction_match.group(1).lower() == "yes"
    
    # Extract Confidence
    confidence_match = re.search(r"Confidence:\s*(0?\.\d+|1\.0|1)", response, re.IGNORECASE)
    if confidence_match:
        try:
            conf = float(confidence_match.group(1))
            result["confidence"] = max(0.0, min(1.0, conf))
        except ValueError:
            pass
    
    return result
