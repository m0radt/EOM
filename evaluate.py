
import sys
from evalplus.evaluate import main as evalplus_main

if __name__ == "__main__":
    # Example args simulating CLI:
    # You can replace these values with your own paths and options
    args = [
        "evalplus.evaluate",
        "--model", "codellama/CodeLlama-7b-Instruct-hf",  # or your model name
        "--samples", "samples.jsonl",
        "--dataset", "humaneval",
        "--backend", "hf",
         "--greedy",  # optionally add more flags here
    ]

    # Replace sys.argv with the args list so EvalPlus CLI can parse them
    sys.argv = args

    # Run EvalPlus main entrypoint
    evalplus_main()
