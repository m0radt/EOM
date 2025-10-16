#!/bin/bash

#SBATCH --partition main                        ### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 6-02:00:00                        ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name my_job                       ### name of the job
#SBATCH --output job-%J.out                     ### output log for running job - %J for job number
#SBATCH --gpus=rtx_6000:1                               ### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1
#SBATCH --mail-user=tayamor@post.bgu.ac.il      ### user's email for sending job status messages
#SBATCH --mail-type=ALL                 ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --array=0-0                   # Array indices (e.g., 9 different models)
##SBATCH --mem=48G                              ### ammount of RAM memory
##SBATCH --cpus-per-task=9                   ### number of CPU cores
################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
### Start your code below ####


models_and_settings=(
    #      Model_Name                         Backend    Parameter_Count Optional_Flags
    # deepseek-ai/deepseek-coder-6.7b-instruct  hf         6.7              ""
    # TheBloke/Phind-CodeLlama-34B-v1-GPTQ      vllm       34.0             ""
    # microsoft/phi-4                         hf         14.7              ""
    # microsoft/NextCoder-14B              hf         14.0             ""
    # Qwen/Qwen2.5-Coder-14B-Instruct     hf         14.0             ""
    # Qwen/Qwen3-8B                  hf         8.0              ""
    # Qwen/Qwen3-14B                 hf         14.0             ""
    # google/gemma-2-9b-it           hf       9.0              ""
    deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct hf         15.7              ""
    # nvidia/OpenCodeReasoning-Nemotron-14B       hf         14.0             ""
    # nvidia/NVIDIA-Nemotron-Nano-12B-v2        vllm         12.0              ""


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
root="evalplus_results"


echo "Running evaluation for model: $model, backend: $backend, parameters: $param_num B, dataset: $dataset , temperature: $temperature and flags: $optional_flags"

module load anaconda                            ### load anaconda module (must be present when working with conda environments)
source activate eom                 ### activate a conda environment.

export HUGGINGFACE_HUB_TOKEN=$(cat /home/tayamor/.hf_token)
python -c "
import os
from huggingface_hub import login

hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
if hf_token:
    login(token=hf_token)
else:
    print('Warning: Hugging Face token not found. Gated models may not be accessible.')
"

nvidia-smi 
# python split_data.py --dataset $dataset --out_prefix splits/${dataset}_split --val_size $val_size --seed $random_seed
python embed_model.py --model $model --dataset $dataset --backend $backend --embedding_dir $embedding_dir --validation_ids_path $validation_ids_path --root $root $optional_flags
# evalplus.evaluate --model $model --dataset $dataset --backend $backend  --temperature $temperature --device_map auto --trust_remote_code True $optional_flags
nvidia-smi 





# Example commands to run outside the SLURM script:

# python split_data.py --dataset humaneval --out_prefix splits/humaneval_split --val_size 10 --seed 42
# python embed_model.py --model deepseek-ai/deepseek-coder-6.7b-instruct --dataset humaneval --backend hf  --validation_ids_path splits/humaneval_split_val_ids.json --root evalplus_results
#python ensemble_select.py --test_ids_path splits/humaneval_split_test_ids.json
# python ensemble_select.py --dataset humaneval --models_num 5 --test_ids_path splits/humaneval_split_test_ids.json --validation_ids_path splits/humaneval_split_val_ids.json  --root evalplus_results 
