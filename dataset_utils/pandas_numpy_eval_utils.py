import json
import gzip
import random
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
from multiprocessing import Pool
import platform
from tqdm import tqdm
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from data.pandas_numpy_eval.pandas_numpy_eval.execution import check_correctness

random.seed(0)


class PandasNumpyEvalLoader:
    def __init__(self):
        self.root = root_path
        self.pandas_eval_file = os.path.join(self.root, 'data/pandas_numpy_eval/data/PandasEval.jsonl.gz')
        self.numpy_eval_file = self.pandas_eval_file.replace('PandasEval', 'NumpyEval')
        self.oracle_docs_matched_file = os.path.join(self.root, 'data/pandas_numpy_eval/data/oracle_docs_matched_processed.json')

    def load_qs_list(self):
        """
        all elements in qs list should be in the format of {'nl': nl, 'qs_id': qs_id}
        """
        qs_list = list()
        with gzip.open(self.pandas_eval_file, 'rt') as f:
            for line in f:
                data = json.loads(line)
                qs_list.append(dict(qs_id=data['qs_id'], question=data['prompt']))
        with gzip.open(self.numpy_eval_file, 'rt') as f:
            for line in f:
                data = json.loads(line)
                qs_list.append(dict(qs_id=data['qs_id'], question=data['prompt']))
        oracle_list = json.load(open(self.oracle_docs_matched_file, 'r'))
        _qs_list = list()
        for oracle in oracle_list:
            for qs in qs_list:
                if qs['qs_id'] == oracle['qs_id']:
                    _qs_list.append(qs)
                    break
        return _qs_list

    def load_oracle_list(self):
        """
        {'qs_id': str, 'oracle_docs': a list of libs}
        """
        oracle_list = json.load(open(self.oracle_docs_matched_file, 'r'))
        oracle_list = [dict(qs_id=oracle['qs_id'], oracle_docs=oracle['oracle_docs'], output=oracle['output']) for oracle in oracle_list]
        return oracle_list

    # def test_helper(self, problem_code_pair):
    #     data, outputs = problem_code_pair
    #     results = []
    #     for idx, output in enumerate(outputs):
    #         result = check_correctness(problem=data, completion=output, timeout=60, completion_id=idx)
    #         results.append(result['passed'])
    #     return results

    def eval_passk(self, predictions, k_list, num_procs=16):
        data_list = list()
        with gzip.open(self.pandas_eval_file, 'rt') as f:
            for line in f:
                data_list.append(json.loads(line))
        with gzip.open(self.numpy_eval_file, 'rt') as f:
            for line in f:
                data_list.append(json.loads(line))

        with ThreadPoolExecutor(max_workers=num_procs) as executor:
            results_list = defaultdict(list)
            futures = list()
            for pred in predictions:
                for data in data_list:
                    if data['task_id'] == pred['qs_id']:
                        assert type(pred['outputs']) is list and len(pred['outputs']) >= max(k_list)
                        for idx, output in enumerate(pred['outputs']):
                            args = (data, output, 3, idx)
                            future = executor.submit(check_correctness, *args)
                            futures.append(future)
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results_list[result['task_id']].append(result['passed'])
            results_list = list(results_list.values())

        return self.pass_rate(results_list, k_list)

    @staticmethod
    def pass_rate(results_list, k_list):
        def pass_at_k(n, c, k):
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        pass_scores = dict()
        for k in k_list:
            pass_scores[f'pass@{k}'] = 0
            for results in results_list:
                passed_num = results.count(True)
                total_num = len(results)
                pass_scores[f'pass@{k}'] += pass_at_k(n=total_num, c=passed_num, k=k)
        for key in pass_scores.keys():
            pass_scores[key] = pass_scores[key] / len(results_list)
        return pass_scores


if __name__ == '__main__':
    # sanity check pass@k
    pandas_numpy_eval_utils = PandasNumpyEvalLoader()
    oracle_list = pandas_numpy_eval_utils.load_oracle_list()
    preds = [dict(qs_id=oracle['qs_id'], outputs=[oracle['output']]) for oracle in oracle_list]
    pass_scores = pandas_numpy_eval_utils.eval_passk(preds, [1])
    print(pass_scores)

    # data_list = list()
    # with gzip.open(pandas_numpy_eval_utils.pandas_eval_file, 'rt') as f:
    #     for line in f:
    #         data_list.append(json.loads(line))
    # with gzip.open(pandas_numpy_eval_utils.numpy_eval_file, 'rt') as f:
    #     for line in f:
    #         data_list.append(json.loads(line))
    # count = 0
    # for pred in preds:
    #     for data in data_list:
    #         if data['task_id'] == pred['qs_id']:
    #             for idx, output in enumerate(pred['outputs']):
    #                 count += 1
    # print(len(preds))
    # print(count)