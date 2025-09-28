import os

from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

def load_embeddings(embedding_dir) -> pd.DataFrame:
    """
    Scan `embedding_dir` for */fingerprint.npy.
    Returns:
      X: (n_models, d) float32 embedding matrix
      info: DataFrame with columns [identifier, model, backend, temperature, path]
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

    df = pd.DataFrame({
        "model_identifier": model_identifiers,
        "embedding": embeddings,
    })
    return df
def cluster_models(df:pd.DataFrame, K=3, seed=42):# -> Dict[int, list]:
    """Cluster models based on their embeddings using KMeans.
    Args:
      df: DataFrame with columns [model_identifier, embedding]
      K: number of clusters
      seed: random seed for reproducibility
    Returns:
      clusters: dict mapping cluster_id to list of model_identifiers
    """

    X = np.stack(df['embedding'].to_numpy())
    kmeans = KMeans(n_clusters=K, random_state=seed)
    labels = kmeans.fit_predict(X)
    df['cluster'] = labels

    # clusters: Dict[int, List[str]] = {}
    # for label, model_id in zip(labels, df['model_identifier']):
    #     clusters.setdefault(label, []).append(model_id)

    # return clusters
# def select_representatives(clusters, embs, criterion='medoid') -> List[str]: ...
# def save_selection(models, output_path): ...




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model on perplexity.")
    parser.add_argument("--dataset", "-d", type=str, default="humaneval", help="Dataset to evaluate on (default: humaneval)")
    parser.add_argument("--root", "-r", type=str, default="evalplus_results", help="Directory to save results")
    parser.add_argument("--test_ids_path", type=str, required=True, help="Path to test results JSON file")
    parser.add_argument("--embedding_dir", type=str, default="embeddings", help="Directory to save embeddings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()

    os.makedirs(args.embedding_dir, exist_ok=True)

    df = load_embeddings(args.embedding_dir)
    print(f"Loaded {len(df)} model embeddings from {args.embedding_dir} (dim {len(df.iloc[0]['embedding'])})")
    
    cluster_models(df, K=2, seed=args.seed)
    print("Clustering done.")
    print(df.head())
    # --- Inspect results ---
    # 1) which models ended up together?
    cluster_members = (
        df.sort_values(["cluster","model_identifier"])
        .groupby("cluster")["model_identifier"].apply(list)
    )
    print(cluster_members.to_dict())

    # 2) cluster sizes
    print(df["cluster"].value_counts().sort_index())

    Xp = PCA(n_components=2).fit_transform(df['embedding'].to_list())

    plt.figure(figsize=(6,6))
    plt.scatter(Xp[:,0], Xp[:,1], c=df["cluster"])
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA Embedding Projection")
    plt.tight_layout()
    out_path = "embedding_pca.png"
    Path(args.root).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")

    


    # evaluate(
    #     model=args.model,
    #     dataset=args.dataset,
    #     backend=args.backend,
    #     greedy=True,
    #     device_map="auto",
    #     trust_remote_code=True,
    #     subset_path=args.test_ids_path,
    #     root=args.root,
    #     )
    
    