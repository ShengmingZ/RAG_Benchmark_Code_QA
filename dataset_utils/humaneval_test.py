import json
import os
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
import sys, platform
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_configs import HumanEvalLoader
from dataset_utils.match_oracle_docs import extract_func_name
from generator.run_model import chatgpt


if __name__ == '__main__':
    humaneval_loader = HumanEvalLoader()
    oracle_list = humaneval_loader.load_oracle_list()
    qs_list = humaneval_loader.load_qs_list()
    gene_results = []
    for (qs, oracle) in zip(qs_list, oracle_list):
        gold_output = oracle['output']
        func_list = extract_func_name(gold_output)
        # output = chatgpt(prompt=qs['nl'])
        output = oracle['output']
        gene_results.append(dict(task_id=qs['qs_id'], completion=output))

    save_file = os.path.join(root_path, 'data/HumanEval/oracle')
    write_jsonl(save_file, gene_results)

    results = evaluate_functional_correctness(save_file, [1])
    print(results)