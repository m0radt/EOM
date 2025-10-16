import gc
import json
import os
from typing import Dict, List, Optional

from evalplus.data import get_evalperf_data, get_human_eval_plus, get_mbpp_plus
from evalplus.provider import DecoderBase, make_model
from evalplus.sanitize import sanitize
from evalplus.utils import progress


def codegen(
    target_path: str,
    model: DecoderBase,
    dataset: Dict,
    greedy=False,
    n_samples=1,
    id_range=None,
    resume=True,
    num_ctx=None,
    attempt_num: int = 0,
):
    print(f"\nUsing model {model.name}")
    print(f"Force_base_prompt {model.force_base_prompt}")
    print(f"Chat template is None: {model.tokenizer.chat_template is None}\n")

    metadata = {"model": str(model),
                "Force_base_prompt": model.force_base_prompt,
                "Chat template is None": model.tokenizer.chat_template is None,
                "is_direct_completion": model.is_direct_completion(),
                "refinement_mode is": model.refinement_mode,
                "attempt_num": attempt_num,
                "samples_metadata": {},
                }
    task2nexist = {}
    if resume and target_path.endswith(".jsonl") and os.path.isfile(target_path):
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                task_id = json.loads(line)["task_id"]
                task2nexist[task_id] = task2nexist.get(task_id, 0) + 1

    if target_path.endswith(".jsonl"):
        raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
    else:
        raw_target_path = target_path + ".raw"
        os.makedirs(target_path, exist_ok=True)

    print(f"Sanitized code outputs will be saved to {target_path}")
    print(f"Raw outputs will be saved to {raw_target_path}")

    backend_type: str = type(model).__name__
    with progress(backend_type) as p:
        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            if not target_path.endswith(".jsonl"):
                p_name = task_id.replace("/", "_")
                os.makedirs(os.path.join(target_path, p_name), exist_ok=True)
                task2nexist[task_id] = len(
                    [
                        f
                        for f in os.listdir(os.path.join(target_path, p_name))
                        if f.endswith(".py")
                    ]
                )

            n_more_samples = n_samples
            log = f"Codegen: {task_id} @ {model}"
            if resume and task2nexist.get(task_id, 0) > 0:
                log += f" (resuming from {task2nexist[task_id]})"
                n_more_samples -= task2nexist[task_id]

            p.console.print(log)

            sidx = n_samples - n_more_samples
            while sidx < n_samples:
                prompt = task["prompt"].strip() + "\n"
                if model.refinement_mode and attempt_num > 0:
                    outputs = model.codegen(
                        task["prompt_for_refinement"].strip() + "\n" if model.is_direct_completion() else prompt,
                        task["previous_solution"].strip(),
                        do_sample=not greedy,
                        num_samples=n_samples - sidx,
                    )
                else:
                    outputs = model.codegen(
                        prompt,
                        do_sample=not greedy,
                        num_samples=n_samples - sidx,
                    )
                assert outputs, "No outputs from model!"
                for impl in outputs:
                    metadata["samples_metadata"][task_id] = {"task_id": task_id,
                                                      "prompt": prompt,
                                                      "refinement_prompt": task["prompt_for_refinement"].strip() + "\n" if model.refinement_mode and attempt_num > 0 and model.is_direct_completion() else None,
                                                      "impl": impl,
                                                      }
                    solution = prompt + impl if model.is_direct_completion() else impl
                    sanitized_solution = sanitize(
                        solution, entrypoint=task["entry_point"]
                    )
                    if target_path.endswith(".jsonl"):
                        # Writing the sanitized version
                        with open(target_path, "a") as f:
                            f.write(
                                json.dumps(
                                    {"task_id": task_id, "solution": sanitized_solution}
                                )
                                + "\n"
                            )

                        # Writing the raw version
                        with open(raw_target_path, "a") as f:
                            f.write(
                                json.dumps({"task_id": task_id, "solution": solution})
                                + "\n"
                            )
                    else:
                        # Writing the sanitized version
                        with open(
                            os.path.join(target_path, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(sanitized_solution)

                        # Writing the raw version
                        with open(
                            os.path.join(raw_target_path, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(solution)
                    sidx += 1
    # Write to a JSON file
    if model.refinement_mode:
        with open(os.path.join(os.path.dirname(target_path), f"sample_metadata{attempt_num}.json"), "w") as f:
            json.dump(metadata, f, indent=4)

def run_codegen(
    model: str,
    dataset: str,
    root: str = "evalplus_results",
    bs: Optional[int] = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    num_ctx: Optional[int] = None,
    resume: bool = True,
    greedy: bool = False,
    id_range: List = None,
    version: str = "default",
    backend: str = "vllm",
    force_base_prompt: bool = False,
    base_url: str = None,
    verify_certificate: bool = True,
    tp: int = 1,
    evalperf_type: str = None,  # For EvalPerf
    jsonl_fmt: bool = True,
    attn_implementation: str = "eager",
    device_map: Optional[str] = None,
    trust_remote_code: bool = False,
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = False,
    dtype: str = "bfloat16",
    gptqmodel_backend: str = "auto",  # For GPTQModel
    gguf_file: Optional[str] = None,
    subset_path: Optional[str] = None,
    refinement_mode: bool = False,
    attempt_num: int = 0,
    previous_model: Optional[str] = None,
):
    # if refinement_mode: force_base_prompt = True
    assert dataset in ["humaneval", "mbpp", "evalperf"], f"Invalid dataset {dataset}"
    assert evalperf_type is None or evalperf_type in [
        "instruct",
        "perf-instruct",
        "perf-CoT",
    ]

    # Make dir for codes generated by each model
    identifier = model.strip("./").replace("/", "--") + f"_{backend}_temp_{temperature}"
    if evalperf_type:
        identifier += f"-{evalperf_type}"
    if refinement_mode:
        target_path = os.path.join(root, dataset, f"refinement_attempt_{attempt_num}", identifier)
    else:
        target_path = os.path.join(root, dataset, identifier)
    if jsonl_fmt:
        target_path += ".jsonl"
    else:
        os.makedirs(target_path, exist_ok=True)

    if dataset == "humaneval":
        dataset_dict = get_human_eval_plus(version=version)
    elif dataset == "mbpp":
        dataset_dict = get_mbpp_plus(version=version)
    elif dataset == "evalperf":
        original_dataset = {**get_human_eval_plus(), **get_mbpp_plus()}
        dataset_dict = {k: original_dataset[k] for k in get_evalperf_data()}
        assert id_range is None, "id_range not supported for evalperf"
    else:
        raise ValueError(f"Invalid dataset {dataset}")
    
    if subset_path:
        with open(subset_path) as f:
            ids = json.load(f)
            if not isinstance(ids, list):
                ids = [ids]
        filtered = {k: v for k, v in dataset_dict.items() if k in ids}
        dataset_dict = filtered
    
    if refinement_mode and attempt_num > 0:
        # get the previous attempt results
        previous_target_path = os.path.join(root, dataset, f"refinement_attempt_{attempt_num-1}", previous_model+".jsonl") if jsonl_fmt else os.path.join(root, dataset, f"refinement_attempt_{attempt_num-1}", previous_model)
        with open(previous_target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_id = data["task_id"]
                previous_solution = data["solution"]
                # dataset_dict[task_id]["prompt_for_refinement"] = (
                #                         "Please provide an improved and self-contained Python script that refines the following problem solution. "
                #                         "The existing attempt may contain bugs, inefficiencies, or stylistic issues. "
                #                         "Ensure the final code is concise, correct, and readable, while preserving the function signature and docstring.\n\n"
                #                         f"Problem description:\n{dataset_dict[task_id]['prompt'].strip()}\n\n"
                #                         f"Existing solution:\n{previous_solution.strip()}\n\n"
                #                         "Enhance the code by fixing any logical errors, optimizing performance, and improving style where appropriate. "
                #                         "Output only the complete, refined Python solution within a markdown code block.\n\n"
                #                         "Refined solution:\n"
                #                         f"{dataset_dict[task_id]['prompt'].strip()}\n"
                #                     )
                # dataset_dict[task_id]["prompt_for_refinement"] = (
                #                         "Please provide an improved and self-contained Python script that refines the following problem solution. "
                #                         "The existing attempt may contain bugs, inefficiencies, or stylistic issues. "
                #                         "Ensure the final code is concise, correct, and readable, while preserving the function signature and docstring. "
                #                         "If the existing solution is already correct, efficient, and readable, return it EXACTLY as given with no changes. "
                #                         "Only modify code when a clear improvement is needed (e.g., fixing logical errors, improving asymptotic or memory performance, "
                #                         "or clarifying the implementation without changing behavior). Avoid unnecessary refactors, renaming, or stylistic churn.\n\n"
                #                         f"Problem description:\n{dataset_dict[task_id]['prompt'].strip()}\n\n"
                #                         f"Existing solution:\n{previous_solution.strip()}\n\n"
                #                         "Output only the complete, refined Python solution (no explanations, only code).\n\n"
                #                         "Refined solution:\n"
                #                         f"{dataset_dict[task_id]['prompt'].strip()}\n"
                #                         "    # Complete Python solution below\n"
                #                     )
                dataset_dict[task_id]["prompt_for_refinement"] = (
                                        "You are given a coding problem and a previous model’s attempt at solving it. "
                                        "Your task is to refine and improve the previous solution to produce a final, high-quality code implementation. "
                                        "Carefully analyze the problem description and the previous answer. "
                                        "Identify any errors, inefficiencies, poor practices, or incomplete logic in the previous code, and correct them. "
                                        "If the code is mostly correct, improve it by enhancing clarity, structure, performance, and adherence to best practices. "
                                        "The final code should be correct, efficient, clean, and comprehensive, including any necessary components such as imports or helper functions. "
                                        "of precision, clarity, and reliability.\n\n"
                                        f"Problem Description:\n{dataset_dict[task_id]['prompt'].strip()}\n\n"
                                        f"Previous Solution:\n{previous_solution.strip()}\n\n"
                                        # "Output only the complete, refined Python solution (no explanations, only code).\n\n"
                                        "Produce the refined and improved final solution in code.\n\n"
                                        "Refined solution:\n"
                                        f"{dataset_dict[task_id]['prompt'].strip()}\n"
                                        )
                dataset_dict[task_id]["previous_solution"] = previous_solution
                # dataset_dict[task_id]["prompt_for_refinement"] = (
                #                         f"You are given the following programming problem.\n\n"
                #                         f"{dataset_dict[task_id]["prompt"].strip()}\n\n"
                #                         "Here is an existing attempt at solving the problem:\n\n"
                #                         f"{previous_solution.strip()}\n\n"
                #                         "The existing attempt may contain bugs, inefficiencies, or stylistic "
                #                         "issues.  Improve upon this solution to produce a correct, concise, and "
                #                         "readable implementation.  Preserve the function signature and docstring. "
                #                         "If possible, fix any logical errors and optimize the code.  Output only "
                #                         "the full, refined Python solution."
                #                     )
            

    all_tasks_complete = False
    if jsonl_fmt and os.path.isfile(target_path):
        task_counts = {}
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_id = data["task_id"]
                task_counts[task_id] = task_counts.get(task_id, 0) + 1

            all_tasks_complete = all(
                task_counts.get(task_id, 0) >= n_samples
                for task_id in dataset_dict.keys()
            )

    if all_tasks_complete:
        print("All samples are already cached. Skipping codegen.")
        return target_path

    if greedy and (temperature != 0 or bs != 1 or n_samples != 1):
        temperature = 0.0
        bs = 1
        n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if id_range is not None:
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    if bs is None:
        bs = min(n_samples, 32)
        print(f"Setting batch size to {bs}")

    if backend != "ollama" and num_ctx is not None:
        print(
            "Warning --num_ctx can be set on ollama backend only. num_ctx will be ignored."
        )

    # Make project dir
    os.makedirs(root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(root, dataset), exist_ok=True)
    # Make refinement attempt dir
    if refinement_mode:
        os.makedirs(os.path.join(root, dataset, f"refinement_attempt_{attempt_num}"), exist_ok=True)

    # Model instructions
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

    if refinement_mode and attempt_num > 0:
        instruction_prefix = """You are given a coding problem and a previous model’s solution to that problem. Produce a single, high-quality final solution by critically evaluating and refining the previous answer. Read the problem carefully, verify correctness against the full specification and edge cases, and fix any bugs, logical gaps, inefficiencies, or omissions. If the approach is sound, keep it while improving clarity, structure, and performance; if it is flawed, replace it with a better implementation. Preserve the required input/output contract and public API of the problem, add any necessary imports, include type hints and a concise docstring when helpful, avoid unnecessary I/O and side effects unless explicitly required, ensure deterministic behavior, and rely only on the Python standard library unless instructed otherwise. The result must be self-contained, executable, and faithful to the specification. Return only the final code; do not include analysis or prose outside the code block."""
        response_prefix = "Below is a refined and improved final Python script with a self-contained function that solves the problem and passes corresponding tests:"
    elif evalperf_type == "perf-instruct":
        instruction_prefix = "Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type == "perf-CoT":
        instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type is not None and evalperf_type != "instruct":
        raise ValueError(f"Invalid evalperf_type: {evalperf_type}")

    # Model creation
    model_runner = make_model(
        model=model,
        backend=backend,
        batch_size=bs,
        temperature=temperature,
        num_ctx=num_ctx,
        force_base_prompt=force_base_prompt,
        dataset=dataset,
        base_url=base_url,
        verify_certificate=verify_certificate,
        tp=tp,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        dtype=dtype,
        gptqmodel_backend=gptqmodel_backend,
        gguf_file=gguf_file,
        refinement_mode=refinement_mode,
    )

    codegen(
        target_path=target_path,
        dataset=dataset_dict,
        greedy=greedy,
        model=model_runner,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
        attempt_num=attempt_num,
    )

    # force shutdown the model runner
    del model_runner

    gc.collect()

    return target_path


def main():
    from fire import Fire

    Fire(run_codegen)


if __name__ == "__main__":
    main()
