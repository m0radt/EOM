# embed_model.py
# compute a model embedding from validation responses.
import json
import os
from pathlib import Path
from typing import  List

import numpy as np
from sentence_transformers import SentenceTransformer

from evalplus.evaluate import evaluate

def load_llm_responses(results_path: Path) -> List[str]:
    """Load LLM responses from a JSONL results file."""
    responses = []
    with open(results_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            responses.append(entry['solution'])
    return responses

def compute_model_embedding(responses: List[str]) -> np.ndarray:
    """Compute a model embedding from a list of responses."""
    if not responses:
        raise ValueError("No responses provided for embedding computation.")
    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(responses, convert_to_numpy=True)

    # Compute fingerprint
    fingerprint = embeddings.mean(axis=0)
    return fingerprint

def save_embedding(fingerprint: np.ndarray, model_identifier: str, embedding_dir: str) -> None:
    """Save the model embedding to a .npy file."""
    model_dir = os.path.join(embedding_dir, f"{model_identifier}")
    os.makedirs(model_dir, exist_ok=True)
    fingerprint_file = os.path.join(model_dir, 'fingerprint.npy')
    np.save(fingerprint_file, fingerprint)
    print(f"Saved {model_identifier} embedding to {fingerprint_file} (shape {fingerprint.shape}).")
    return fingerprint_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model on perplexity.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model identifier")
    parser.add_argument("--dataset", "-d", type=str, default="humaneval", help="Dataset to evaluate on (default: humaneval)")
    parser.add_argument("--backend", "-b", type=str, default="hf", help="Backend to use (default: hf)")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="Temperature for sampling (default: 0.0)")
    parser.add_argument("--root", "-r", type=str, default="evalplus_results", help="Directory to save results")
    parser.add_argument("--validation_ids_path", type=str, required=True, help="Path to validation results JSON file")
    parser.add_argument("--embedding_dir", type=str, default="embeddings", help="Directory to save embeddings")
    
    args = parser.parse_args()


    os.makedirs(args.embedding_dir, exist_ok=True)

    evaluate(
        model=args.model,
        dataset=args.dataset,
        backend=args.backend,
        greedy=True,
        device_map="auto",
        trust_remote_code=True,
        subset_path=args.validation_ids_path,
        root=args.root,
        )
    
    identifier = args.model.strip("./").replace("/", "--") + f"_{args.backend}_temp_{args.temperature}"

    responses = load_llm_responses(os.path.join(args.root, args.dataset, identifier+".jsonl"))
    print(f"Loaded {len(responses)} responses from {args.validation_ids_path}")
    model_embedding = compute_model_embedding(responses)
    save_embedding(model_embedding, identifier, args.embedding_dir)
    