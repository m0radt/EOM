from concurrent.futures import ThreadPoolExecutor
from typing import List

EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]


def extra_eos_for_direct_completion(dataset) -> List[str]:
    if dataset.lower() == "humaneval":
        return ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    elif dataset.lower() == "mbpp":
        return ['\n"""', "\nassert"]
    raise ValueError(f"Unknown dataset: {dataset}")


# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def make_raw_chat_prompt(
    task_prompt: str,
    instruction_prefix: str,
    response_prefix: str,
    tokenizer,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer.chat_template is None:
        return task_prompt

    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"

    task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""
    task_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]
    return task_prompt

def make_raw_chat_refinement_prompt(
    problem_description: str,
    previous_solution: str,
    instruction_prefix: str,
    response_prefix: str,
    tokenizer,
) -> str:
    assert instruction_prefix is not None, "Instruction prefix is required!"
    assert response_prefix is not None, "Response prefix is required!"

    user_msg = f"""\
{instruction_prefix}
Problem Description:
```
{problem_description.strip()}
```

Previous Solution:
```python
{previous_solution.strip()}
```
"""

    assistant_seed = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""

    # If no chat template, fall back to a plain prompt that starts the code block
    if getattr(tokenizer, "chat_template", None) is None:
        # Start the assistant code block so generation continues with code immediately.
        print("Warning: tokenizer has no chat_template; using plain prompt without refinement context.")
        return f"{user_msg}\n{response_prefix}\n```python\n"

    # With a chat template: apply, then cut everything after the splitter
    prompt_with_roles = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_seed},
        ],
        tokenize=False,
    )

    return prompt_with_roles.split(_MAGIC_SPLITTER_)[0]

def concurrent_call(n, callback, /, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(callback, *args, **kwargs) for _ in range(n)]
        return [future.result() for future in futures]
