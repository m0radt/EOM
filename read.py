import json

# file_path_raw = "evalplus_results/humaneval/codellama--CodeLlama-7b-Instruct-hf_hf_temp_0.0.raw.jsonl"
# file_path = "evalplus_results/humaneval/codellama--CodeLlama-7b-Instruct-hf_hf_temp_0.0.jsonl"
# data_raw = {}
# data = {}

# with open(file_path, "r") as f:
#     for line in f:
#         obj = json.loads(line)
#         task_id = obj["task_id"]
#         data[task_id] = obj
# with open(file_path_raw, "r") as f_raw:
#     for line in f_raw:
#         obj = json.loads(line)
#         task_id = obj["task_id"]
#         data_raw[task_id] = obj
# id = "HumanEval/0"

# # Now `data` is a list of dictionaries
# print("Processed data:")
# print(data[id]["solution"])
# print("Raw data:")
# print(data_raw[id]["solution"])
from pygments.lexers import PythonLexer
from pygments.token import Token
from pygments import lex

def count_pygments_tokens(code: str) -> int:
    """
    Counts the number of meaningful Pygments tokens in a Python code string.
    """
    tokens = lex(code, PythonLexer())
    count = 0
    for ttype, value in tokens:
        if ttype in Token.Name or ttype in Token.Keyword or ttype in Token.Literal or ttype in Token.Operator:
            count += 1
    return count
print(count_pygments_tokens("x = 5"))
