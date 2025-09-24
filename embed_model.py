# embed_model.py
# compute a model embedding from validation responses.
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from evalplus.evaluate import evaluate

# def compute_model_embedding(responses: List[str]) -> np.ndarray:
#     """Compute a model embedding from a list of responses."""
#     if not responses:
#         raise ValueError("No responses provided for embedding computation.")
#     # Load embedding model
#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings = embedding_model.encode(responses, convert_to_numpy=True)

#     # Compute fingerprint
#     fingerprint = embeddings.mean(axis=0)
#     return fingerprint

# def save_embedding(fingerprint: np.ndarray, model_name: str, embedding_dir: str) -> None:
#     """Save the model embedding to a .npy file."""
#     model_dir = os.path.join(embedding_dir, f"{model_name}")
#     os.makedirs(model_dir, exist_ok=True)
#     fingerprint_file = os.path.join(model_dir, 'fingerprint.npy')
#     np.save(fingerprint_file, fingerprint)
#     print(f"Saved {model_name} embedding to {fingerprint_file} (shape {fingerprint.shape}).")
#     return fingerprint_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model on perplexity.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model identifier")
    parser.add_argument("--param_num", "-p", type=float, required=True, help="Number of parameters in billions")
    parser.add_argument("--backend", "-b", type=str, default="hf", help="Backend to use (default: hf)")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="Temperature for sampling (default: 0.0)")
    parser.add_argument("--results_dir", "-r", type=str, default="results", help="Directory to save results")
    parser.add_argument("--validation_path", type=str, required=True, help="Path to validation results JSONL file")
    parser.add_argument("--embedding_dir", type=str, default="embeddings", help="Directory to save embeddings")
    
    args = parser.parse_args()


    os.makedirs(args.embedding_dir, exist_ok=True)
    evaluate(
        model=args.model,
        # dataset="humaneval",
        backend=args.backend,
        greedy=True,
        device_map="auto",
        trust_remote_code=True,
        )

    # embedding = compute_model_embedding(responses)
    # save_embedding(embedding, args.model.replace('/', '_'), args.embedding_dir)
    