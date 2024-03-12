import json
import os
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm
import sys, platform
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_configs import HumanEvalLoader, DS1000Loader
from dataset_utils.match_oracle_docs import extract_func_name
from generator.run_model import chatgpt


if __name__ == '__main__':
    # gene results
    humaneval_loader = HumanEvalLoader()
    oracle_list = humaneval_loader.load_oracle_list()
    qs_list = humaneval_loader.load_qs_list()
    ds1000_loader = DS1000Loader()
    ds1000_oracle_list = ds1000_loader.load_oracle_list()
    ds1000_qs_list = ds1000_loader.load_qs_list()
    # gene_results = []
    # n = 1
    # for (qs, oracle) in tqdm(zip(qs_list, oracle_list)):
    #     gold_output = oracle['output']
    #     func_list = extract_func_name(gold_output)
    #     outputs = chatgpt(prompt=qs['nl'], n=n)
    #     # output = oracle['output']
    #     for idx in range(n):
    #         gene_results.append(dict(task_id=qs['qs_id'], completion=outputs[idx]))
    # save_file = os.path.join(root_path, 'data/HumanEval/test')
    # write_jsonl(save_file, gene_results)
    #
    # results = evaluate_functional_correctness(save_file, [1])
    # print(results)

    # result_file = '../data/HumanEval/test_results.jsonl'
    # test_results = []
    # with open(result_file, "r") as fp:
    #     for line in fp:
    #         if any(not x.isspace() for x in line):
    #             test_results.append(json.loads(line))
    # print(test_results[0])
    # for (qs, oracle, result) in zip(qs_list, oracle_list, test_results):
    #     assert qs['qs_id'] == result['task_id']
    #     if result['passed'] is False:
    #         print(result['task_id'])
    #         print([qs['nl']])
    #         print(extract_func_name(oracle['output']))

    for qs in ds1000_qs_list[:2]:
        print(qs['nl'])
