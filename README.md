# Ensemble of Minds: Combining Large Language Models for Enhanced Response Generation

This repo evaluates multiple **code LLMs** on HumanEval(+), computes **per-model embeddings**, and then **clusters & selects** an ensemble.

---

## ✅ Prerequisites

- **Python 3.11** (required)
- (HPC) **SLURM** + a GPU partition
- (Optional) **Hugging Face token** for gated models

---

## 1) Environment setup

### Conda (recommended)
```bash
module load anaconda            # if your cluster uses modules
conda create -n eom python=3.11 -y
conda activate eom
pip install -r requirements.txt
```

### venv
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> The SLURM script below assumes a conda env named **eom**. If you use a different env, change the `source activate` line.

---

## 2) Hugging Face auth (optional but recommended)

```bash
# Option A: export directly
export HUGGINGFACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX

# Option B: store once and let the job read it
echo hf_XXXXXXXXXXXXXXXXXXXXXXXX > ~/.hf_token
```

---

## 3) Create data splits

`split_data.py` writes **JSON** files (arrays of task IDs like `"HumanEval/0"`), used as subsets for validation/test.

```bash
python split_data.py \
  --dataset humaneval \
  --out_prefix splits/humaneval_split \
  --val_size 100 \
  --seed 42
```

This produces:
```
splits/humaneval_split_val_ids.json
splits/humaneval_split_test_ids.json
```

**Format example (`*_ids.json`)**
```json
[
  "HumanEval/0",
  "HumanEval/3",
  "HumanEval/17"
]
```

> ⚠️ Must be **JSON arrays**, not JSONL. Each entry should be a full task ID (e.g., `"HumanEval/0"`).

---

## 4) Submit jobs for each model (SLURM array)

Use `my_job_array.sh` to (a) generate & evaluate on the validation subset and (b) compute the model embedding.

### `my_job_array.sh`
```bash
#!/bin/bash
#SBATCH --partition main
#SBATCH --time 6-02:00:00
#SBATCH --job-name my_job
#SBATCH --output job-%J.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --mail-user=you@example.com
#SBATCH --mail-type=ALL
#SBATCH --array=0-0          # <-- set to 0-(N-1) for N models
##SBATCH --mem=48G
##SBATCH --cpus-per-task=9

echo "$(date)"
echo -e "\nSLURM_JOBID:\t\t$SLURM_JOBID"
echo -e "SLURM_JOB_NODELIST:\t$SLURM_JOB_NODELIST\n"

# Model_Name                                Backend  Param_B  Optional_Flags
models_and_settings=(
  deepseek-ai/deepseek-coder-6.7b-instruct  hf       6.7      ""
  # Add more lines in groups of four:
  # Salesforce/codegen-6B-multi             hf       6.0      ""
  # meta-llama/CodeLlama-7b-Instruct-hf     hf       6.74     ""
)

i=$((SLURM_ARRAY_TASK_ID * 4))
model=${models_and_settings[$i]}
backend=${models_and_settings[$((i + 1))]}
param_num=${models_and_settings[$((i + 2))]}
optional_flags=${models_and_settings[$((i + 3))]}

temperature=0.0
dataset=humaneval
val_size=100
random_seed=42

validation_ids_path="splits/${dataset}_split_val_ids.json"
test_ids_path="splits/${dataset}_split_test_ids.json"

embedding_dir="embeddings"
validation_root="evalplus_validation_results"
test_root="evalplus_test_results"

echo "Running evaluation for model: $model (backend=$backend, params=${param_num}B, dataset=$dataset, T=$temperature) flags: $optional_flags"

module load anaconda
source activate eom

# HF token (optional)
export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-$(cat ~/.hf_token 2>/dev/null)}
python - <<'PY'
import os
try:
    from huggingface_hub import login
    tok = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if tok: login(token=tok)
    else: print("Warning: Hugging Face token not found. Gated models may not be accessible.")
except Exception as e:
    print("HF login skipped or failed:", e)
PY


# 1) Create/overwrite splits (id files are JSON arrays)
python split_data.py \
  --dataset $dataset \
  --out_prefix splits/${dataset}_split \
  --val_size $val_size \
  --seed $random_seed

# 2) Generate, evaluate, and compute the model embedding (validation subset)
python embed_model.py \
  --model $model \
  --dataset $dataset \
  --backend $backend \
  --temperature $temperature \
  --embedding_dir $embedding_dir \
  --validation_ids_path $validation_ids_path \
  --root $validation_root \
  $optional_flags


### Submit the array

For **N** models, set `--array=0-(N-1)`:

```bash
# Example: 3 models (indices 0,1,2)
sbatch --array=0-2 my_job_array.sh
```

Check status & logs:
```bash
squeue -u $USER
tail -f job-<JOBID>.out
```

---

## 5) Files & directories

- **Generations / evaluation (validation)**  
  `evalplus_validation_results/humaneval/`  
  ```
  <model>_<backend>_temp_<T>.jsonl          # sanitized generations
  <model>_<backend>_temp_<T>.raw.jsonl      # raw generations
  <model>_<backend>_temp_<T>.eval_results.json
  ```

- **Embeddings**  
  `embeddings/<model>_<backend>_temp_<T>/fingerprint.npy`

- **Ensemble selection output**  
  `ensembles/selected.json`

---

## 6) Ensemble selection (manual run)


```bash
python ensemble_select.py \
  --embeddings_dir embeddings \
  --strategy cluster_representatives \
  --distance cosine \
  --k 5 \
  --output ensembles/selected.json
```

### Dependency method (run after the whole array finishes)

```bash
JID=$(sbatch --array=0-2 my_job_array.sh | awk '{print $4}')
cat > run_ensemble.sh <<'BASH'
#!/bin/bash
#SBATCH --partition main
#SBATCH --time 01:00:00
#SBATCH --job-name ensemble
#SBATCH --output ensemble-%J.out
module load anaconda
source activate eom
python ensemble_select.py --embeddings_dir embeddings --strategy cluster_representatives --distance cosine --k 5 --output ensembles/selected.json
BASH
sbatch --dependency=afterok:$JID run_ensemble.sh
```

---

## 7) Troubleshooting

**“All samples are already cached. Skipping codegen.”**  
You already have `...temp_<T>.jsonl` under `--root`. To force new generations, either:
- delete the old JSONL, or
- change `--root`, or
- change `--temperature` (affects the filename/identifier).


**Splits not applied**  
Ensure your `*_ids.json` is a **JSON array** of **full** task IDs like `"HumanEval/37"`. Numeric IDs (`[0,1,2]`) won’t match unless you normalize them to full keys.

**HF gated models fail to load**  
Export `HUGGINGFACE_HUB_TOKEN` or create `~/.hf_token`.

---

## 8) Quick start

```bash
# 1) env
conda create -n eom python=3.11 -y
conda activate eom
pip install -r requirements.txt

# 2) splits
python split_data.py --dataset humaneval --out_prefix splits/humaneval_split --val_size 10 --seed 42

# 3) run models (edit script first)
sbatch --array=0-0 my_job_array.sh

# 4) (if not run inside the array)
python ensemble_select.py --embeddings_dir embeddings --strategy cluster_representatives --distance cosine --k 5 --output ensembles/selected.json
```

---

## 9) Repo structure (typical)

```
.
├── README.md
├── requirements.txt
└── evalplus/
├── split_data.py
├── embed_model.py
├── ensemble_select.py
├── my_job_array.sh
├── embeddings/
├── ensembles/
└── evalplus_validation_results/
```