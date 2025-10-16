import os

from pathlib import Path
import json
from typing import List, Dict, Any, Tuple


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from evalplus.evaluate import evaluate

import time

def reverse_identifier(identifier: str)-> tuple[str, str, float]:
    """
    Reverse:
      model.strip("./").replace("/", "--") + f"_{backend}_temp_{temperature}"
    Returns model, backend, temperature
    """
    # 1) split off temperature
    sep = "_temp_"
    pos_temp = identifier.rfind(sep)
    if pos_temp == -1:
        raise ValueError("Identifier missing '_temp_' segment")
    temp_str = identifier[pos_temp + len(sep):]

    left = identifier[:pos_temp]

    # 2) split off backend (last underscore before _temp_)
    pos_backend = left.rfind("_")
    if pos_backend == -1:
        raise ValueError("Identifier missing backend segment")
    backend = left[pos_backend + 1:]
    model_enc = left[:pos_backend]

    # 3) decode model (reverse the '/' → '--' mapping)
    model = model_enc.replace("--", "/")

    # 4) parse temperature
    temperature = float(temp_str)

    return model, backend, temperature

def load_embeddings(df: pd.DataFrame, embedding_dir)-> None:
    """
    Scan `embedding_dir` for */fingerprint.npy.
    Load them and populate `df` with columns [model_identifier, embedding]
    Args:
      df: DataFrame to populate (modified in place)
      embedding_dir: directory containing subdirs for each model, each with fingerprint.npy
    Returns: None
    """
    root = Path(embedding_dir)
    model_identifiers: List[str] = []
    embeddings: List[np.ndarray] = []

    if not root.exists():
        raise FileNotFoundError(f"Embedding dir not found: {embedding_dir}")

    for identifier in sorted(p for p in root.iterdir() if p.is_dir()):
        fp = identifier / "fingerprint.npy"
        if not fp.exists():
            print(f"Skip {fp}: not found")
            continue
        try:
            v = np.load(fp)
        except Exception as e:
            print(f"Skip {fp}: failed to load ({e})")
            continue

        model_identifiers.append(identifier.name)
        # ensure 1-D and dtype
        v = np.asarray(v).reshape(-1).astype(np.float32, copy=False)
        embeddings.append(v)

    if not embeddings:
        raise RuntimeError(f"No fingerprints found under {embedding_dir}")

    # sanity: all same dimensionality
    dims = {len(v) for v in embeddings}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent fingerprint dims found: {sorted(dims)}")
    # populate df
    df["model_identifier"] = model_identifiers
    df["embedding"] = embeddings

def cluster_models(df: pd.DataFrame, K=3, seed=42)-> None:
    """Cluster models based on their embeddings using KMeans.
    Args:
      df: DataFrame with columns [model_identifier, embedding]
      K: number of clusters
      seed: random seed for reproducibility
    Returns: None (modifies df in place, adding 'cluster' column)
    """

    X = np.stack(df['embedding'].to_numpy())
    kmeans = KMeans(n_clusters=K, random_state=seed)
    labels = kmeans.fit_predict(X)
    df['cluster'] = labels

def load_accuracies(results_root: Path, df: pd.DataFrame=None, plus: bool=True, k: int=1) -> dict[str, float]:
    """Load accuracies from a JSON file and optionally add to df in place.
    Args:
      results_root: directory containing *.eval_results.json files
      plus: whether to load 'plus' or 'base' accuracies
      k: which pass@k to load
      df: if provided, add 'accuracy' column in place
    Returns: None (modifies df in place) if df is not None else returns dict of accuracies
    """
    accuracies = {}
    for model_evaluation_path in results_root.glob("*.eval_results.json"):
        model_identifier = model_evaluation_path.name.replace(".eval_results.json","")
        with open(model_evaluation_path, 'r') as f:
            results = json.load(f)
            accuracy = results["pass_at_k"]["plus" if plus else "base"][f"pass@{k}"]
        accuracies[model_identifier] = accuracy

    if df is not None:
        df['accuracy'] = df['model_identifier'].map(accuracies)
    return accuracies
    

def select_representatives(df: pd.DataFrame) -> List[str]:
    """Select representative models from each cluster based on accuracy.
    Args:
      df: DataFrame with columns [model_identifier, embedding, cluster, accuracy]
    Returns: List of selected model_identifiers soreted by accuracy descending
    """
    selected_models = []
    for cluster_id, group in df.groupby('cluster'):
        # Select model with highest accuracy in the cluster
        best_model = group.loc[group['accuracy'].idxmax()]['model_identifier']
        selected_models.append(best_model)

    # Sort selected models by accuracy descending
    selected_models.sort(key=lambda m: df.loc[df['model_identifier'] == m, 'accuracy'].values[0], reverse=True)
    return selected_models

def save_selection(models: List[str], output_path: str) -> None:
    """Save selected model identifiers to a text file."""
    with open(output_path, 'w') as f:
        for model in models:
            f.write(f"{model}\n")
    print(f"Saved selected models to {output_path}")

def load_selection(input_path: str) -> List[str]:
    """Load selected model identifiers from a text file."""
    with open(input_path, 'r') as f:
        models = [line.strip() for line in f if line.strip()]
    return models
def run_evaluation(model_identifiers: List[str], dataset: str, root: str, test_ids_path: str) -> None:
    for model_identifier in model_identifiers:
        model, backend, temperature = reverse_identifier(model_identifier)
        trust_remote_code=False if "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" in model else True
        
        evaluate(
            model=model,
            dataset=dataset,
            backend=backend,
            greedy=True,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            subset_path=test_ids_path,
            root=os.path.join(root, "baseline"),
            # delete this line if not after
            # force_base_prompt=True,
            #-----------------------------
            )
        time.sleep(3)  # avoid overloading

def simualte_ensemble(model_identifiers: List[str], dataset, root, test_ids_path, k: int=1, ensemble_name: str="eom_ensemble") -> None:
    previous_model = None
    for i, model_identifier in enumerate(model_identifiers):
        if i >= k: break
        model, backend, temperature = reverse_identifier(model_identifier)
        trust_remote_code=False if "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" in model else True
        

        evaluate(
            model=model,
            dataset=dataset,
            backend=backend,
            greedy=True,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            subset_path=test_ids_path,
            root=os.path.join(root, ensemble_name),
            refinement_mode=True,
            attempt_num=i,
            previous_model = previous_model,
            )
        previous_model = str(model_identifier)
        time.sleep(3)  # avoid overloading



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model on perplexity.")
    parser.add_argument("--dataset", "-d", type=str, default="humaneval", help="Dataset to evaluate on (default: humaneval)")
    parser.add_argument("--root", "-r", type=str, default="evalplus_results", help="Directory to save results")
    parser.add_argument("--validation_ids_path", type=str, required=True, help="Path to validation results JSON file")
    parser.add_argument("--test_ids_path", type=str, required=True, help="Path to test results JSON file")
    parser.add_argument("--embedding_dir", type=str, default="embeddings", help="Directory to save embeddings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--models_num", type=int, default=3, help="Number of models to select for ensemble (default: 3)")
    
    args = parser.parse_args()

    os.makedirs(args.embedding_dir, exist_ok=True)
    validation_root = os.path.join(args.root, "validation")
    test_root = os.path.join(args.root, "test")


    df = pd.DataFrame() 
    load_embeddings(df, args.embedding_dir)
    print(f"Loaded {len(df)} model embeddings from {args.embedding_dir} (dim {len(df.iloc[0]['embedding'])})")
    
    cluster_models(df, K=args.models_num, seed=args.seed)
    print("Clustering done.")
    model2accuracy = load_accuracies(Path(os.path.join(validation_root, args.dataset)), df=df, plus=True, k=1)
    print(df)
    ensemble_models = select_representatives(df)
    print("Selected ensemble models:")
    for m in ensemble_models:
        acc = df.loc[df["model_identifier"] == m, "accuracy"].values[0]
        print(f"  {m} (accuracy: {acc})")

    # # --- Inspect results ---
    # # 1) which models ended up together?
    # cluster_members = (
    #     df.sort_values(["cluster","model_identifier"])
    #     .groupby("cluster")["model_identifier"].apply(list)
    # )
    # print(cluster_members.to_dict())

    # # 2) cluster sizes
    # print(df["cluster"].value_counts().sort_index())

    # Xp = PCA(n_components=2).fit_transform(df['embedding'].to_list())

    # plt.figure(figsize=(6,6))
    # plt.scatter(Xp[:,0], Xp[:,1], c=df["cluster"])
    # plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA Embedding Projection")
    # plt.tight_layout()
    # out_path = "embedding_pca.png"
    # Path(args.root).mkdir(parents=True, exist_ok=True)
    # plt.savefig(out_path, dpi=150)
    # plt.close()
    # print(f"Saved plot to {out_path}")


    print(f"Ensemble models: {ensemble_models}")
    models = sorted(model2accuracy, key=model2accuracy.get, reverse=True)
    top_models = models[:args.models_num]
    best_model_refinement = [ensemble_models[0]]* len(ensemble_models)
    least_model_ensemble = [ensemble_models[-1]]* len(ensemble_models)
    # simualte_ensemble(ensemble_models, args.dataset, test_root, args.test_ids_path, k=len(ensemble_models), ensemble_name="eom_ensemble")

    simualte_ensemble(top_models, args.dataset, test_root, args.test_ids_path, k=len(ensemble_models), ensemble_name="top_model_ensemble_new_prompt")
    # simualte_ensemble(best_model_refinement, args.dataset, test_root, args.test_ids_path, k=len(ensemble_models), ensemble_name="best_model_ensemble")
    # simualte_ensemble(least_model_ensemble, args.dataset, test_root, args.test_ids_path, k=len(ensemble_models), ensemble_name="least_model_ensemble")
    # run_evaluation(models, args.dataset, args.root, args.test_ids_path)


    



    
    