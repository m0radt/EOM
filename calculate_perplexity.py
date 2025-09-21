import os
import math
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.data import read_problems
from human_eval.execution import check_correctness
from huggingface_hub import login
from pygments.lexers import PythonLexer
from pygments.token import Token
from pygments import lex


# Ensure that the TOKENIZERS_PARALLELISM environment variable is set to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_model(model_name, device):
    """
    Loads the tokenizer and causal language model from HuggingFace.

    Args:
        model_name (str): The identifier for the pre-trained model on HuggingFace Hub.
        device (str): The device to load the model onto (e.g., "cuda" or "cpu").

    Returns:
        tuple: A tuple containing the loaded tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    # model.to(device)
    model.eval()
    return tokenizer, model


# def compute_log_likelihood_and_pygments_tokens(model, tokenizer, prompt: str, completion: str):
#     """
#     Returns (sum_nll, num_pygments_tokens) for `completion` given `prompt`,
#     or (None, None) if something went wrong.
#     """
#     if not completion:
#         return None, None

#     # 1) Tokenize the prompt once (with specials) and get the special mask.
#     enc = tokenizer(
#         prompt,
#         add_special_tokens=True,
#         return_special_tokens_mask=True
#     )
#     prompt_ids_with_specials = enc["input_ids"]
#     special_mask            = enc["special_tokens_mask"]

#     # 2) Identify prefix & suffix specials
#     #    (for single-sequence causal LM, they only appear at the edges)
#     # special tokens in the middle does not cause any threats
#     # count leading specials
#     prefix_specials = next(
#         (i for i, bit in enumerate(special_mask) if bit == 0),
#         len(special_mask)
#     )
#     # count trailing specials
#     suffix_specials = next(
#         (i for i, bit in enumerate(reversed(special_mask)) if bit == 0),
#         len(special_mask)
#     )

#     # 3) Recover raw prompt IDs (no specials) by slicing
#     start = prefix_specials
#     end   = len(prompt_ids_with_specials) - suffix_specials
#     raw_prompt_ids = prompt_ids_with_specials[start:end]

#     # 4) Compute prompt length in the full token stream
#     #    = all specials (prefix+suffix) that come before the completion
#     prompt_len = prefix_specials + len(raw_prompt_ids)

#     # 5) Tokenize the concatenated prompt+completion (adds specials once)
#     full_text = prompt + completion
#     inputs = tokenizer(
#         full_text,
#         return_tensors="pt",
#         # truncation=True
#         ).to(model.device)

#     # 6) Build labels: mask out prompt + specials
#     labels = inputs.input_ids.clone()
#     labels[:, :prompt_len] = -100

#     # 7) Forward pass to get average NLL, then scale to sum NLL
#     with torch.no_grad():
#         outputs = model(
#             input_ids      = inputs.input_ids,
#             attention_mask = inputs.attention_mask if "attention_mask" in inputs else None,
#             labels         = labels
#         )
#     loss = outputs.loss
#     if loss is None or math.isnan(loss.item()):
#         return None, None

#     valid_token_count = (labels != -100).sum().item()
#     sum_nll = loss.item() * valid_token_count

#     # 8) Pygments token count for normalization
#     norm = count_pygments_tokens(completion)
#     if norm == 0:
#         return None, None

#     return sum_nll, norm

def compute_log_likelihood_and_pygments_tokens(model, tokenizer, prompt: str, completion: str):
    """
    Returns (sum_nll, num_pygments_tokens) for `completion` given `prompt`,
    or (None, None) if something went wrong.
    """
    if not completion:
        return None, None

    # 1) Encode the prompt to get its full token length (including special tokens)
    prompt_inputs = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt"
    ).to(model.device)
    prompt_len = prompt_inputs.input_ids.size(1)

    # 2) Tokenize the concatenated prompt+completion without truncation
    full_text = prompt + completion
    inputs = tokenizer(
        full_text,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=False
    ).to(model.device)

    # 3) Build labels: mask out the prompt tokens (including specials)
    labels = inputs.input_ids.clone()
    labels[:, :prompt_len] = -100  # -100 is the ignore index in PyTorch CrossEntropyLoss

    # 4) Forward pass to compute sum of negative log-likelihood over the completion
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask if hasattr(inputs, 'attention_mask') else None,
            labels=labels
        )
    loss = outputs.loss
    if loss is None or math.isnan(loss.item()):
        return None, None

    # Number of tokens over which loss was computed
    valid_token_count = (labels != -100).sum().item()
    sum_nll = loss.item() * valid_token_count

    # 5) Pygments token count for normalization
    norm = count_pygments_tokens(completion)
    if norm == 0:
        return None, None

    return sum_nll, norm

def save_results(filename, results):
        """Save results to a file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)


def get_function_body(s: str) -> str:
    """Extracts the function body from a string containing a Python function definition."""
    def_idx = s.find("def")
    if def_idx == -1:
        return s
    s = s[def_idx:]  # Start from the function definition
    if '"""' not in s:
        return s
    parts = s.split('"""', 2)
    return parts[2][1:] if parts[2].startswith('\n') else parts[2]

def count_pygments_tokens(code: str) -> int:
    pygments_tokens = list(lex(code, PythonLexer()))
    # Count only meaningful tokens (excluding whitespace, comments, etc., based on Pygments' `Token.Text`).
    return len([1 for token_type, token in pygments_tokens if token_type != Token.Text and token.strip()])


def evaluate_model_on_perplexity(model_name, param_num, backend, temperature):    
    dir = os.path.join( "evalplus_results", "humaneval")
    total_sum_nll = 0.0 # Accumulator for sum of negative log-likelihoods
    total_pygments_tokens = 0 # Accumulator for total Pygments tokens
    num_processed_snippets = 0 # Counter for successfully processed snippets

    metrics = {}
    filename_raw = f'{model_name.replace(os.sep, "--")}_{backend}_temp_{temperature}.jsonl'
    if os.path.exists(os.path.join(dir, filename_raw)):
        tokenizer, model = load_model(model_name, "cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(dir, filename_raw), "r") as f:
            for line in tqdm(f, desc=f"Processing {model_name}"):
                obj = json.loads(line.strip())
                if "task_id" in obj and "solution" in obj:
                    task_id = obj["task_id"]
                    raw_solutions = obj["solution"]
                    function_body = get_function_body(raw_solutions)
                    prompt = raw_solutions[:-len(function_body)]
                    # Compute NLL and Pygments tokens for the current snippet
                    sum_nll, num_pyg_tokens = compute_log_likelihood_and_pygments_tokens(model, tokenizer, prompt, function_body)
                    # Accumulate results if the calculation was successful and Pygments tokens are positive
                    if sum_nll is not None and num_pyg_tokens is not None:
                        total_sum_nll += sum_nll
                        total_pygments_tokens += num_pyg_tokens
                        num_processed_snippets += 1
        # Calculate the average negative log-likelihood per Pygments token.
        # This is the value that will be exponentiated to get perplexity.
        avg_nll_per_pyg_token = total_sum_nll / total_pygments_tokens if total_pygments_tokens > 0 else float('inf')
        
        # Calculate perplexity: exp(average negative log-likelihood).
        # Handle infinite case for when no valid tokens were processed.
        avg_normalized_perplexity = round(math.exp(avg_nll_per_pyg_token), 4) if avg_nll_per_pyg_token != float('inf') else float('inf')
        filename_result = f'{model_name.replace(os.sep, "--")}_{backend}_temp_{temperature}.eval_results.json'
        with open(os.path.join(dir, filename_result), "r") as f:
            obj = json.loads(f.read())
            pass_at_k = obj.get("pass_at_k")
        metrics[model_name] = {
            "model_name": model_name,
            "param_num": param_num,
            "backend": backend,
            "temperature": temperature,
            "avg_normalized_perplexity": avg_normalized_perplexity,
            "pass_at_k": pass_at_k
        }
        print(f"metrics: {metrics[model_name]}")
        print(f"num_processed_snippets: {num_processed_snippets} out of 164")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model on perplexity.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model identifier")
    parser.add_argument("--param_num", "-p", type=float, required=True, help="Number of parameters in billions")
    parser.add_argument("--backend", "-b", type=str, default="hf", help="Backend to use (default: hf)")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="Temperature for sampling (default: 0.0)")
    parser.add_argument("--results_dir", "-r", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()

    metrics = evaluate_model_on_perplexity(args.model, args.param_num, args.backend, args.temperature)
    save_results(os.path.join(args.results_dir, f"{args.model.replace(os.sep, '--')}.json"), metrics)
    