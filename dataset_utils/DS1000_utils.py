import json
from multiprocessing import Pool
import os
import bz2
import random
import platform
import sys
from tqdm import tqdm
from typing import List, Tuple
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from data.DS1000.ds1000 import DS1000Dataset, DS1000Problem

random.seed(0)


class DS1000Loader:
    def __init__(self):
        self.root = root_path
        # self.ds1000 = DS1000Dataset(source_dir=ow.root, 'data/DS1000/sampled_idx.json')
        self.sampled_data_file = os.path.join(self.root, 'data/DS1000/sampled_data.json')
        self.oracle_doc_file = os.path.join(self.root, 'data/DS1000/oracle_docs_matched_processed.json')

    def load_qs_list(self):
        """
        all elements in qs list should be in the format of {'nl': nl, 'qs_id': qs_id}
        """
        sampled_data = json.load(open(self.sampled_data_file, 'r'))
        qs_list = []
        for data in sampled_data:
            qs_list.append(dict(question=data['prompt'], qs_id=data['qs_id']))
        return qs_list

    def load_oracle_list(self):
        """
        {'qs_id': str, 'doc_keys': a list of libs, 'output': output}
        """
        oracle_list = json.load(open(self.oracle_doc_file, 'r'))
        _oracle_list = list()
        for oracle in oracle_list:
            _oracle_list.append(dict(qs_id=oracle['qs_id'], oracle_docs=oracle['oracle_docs'], output=oracle['output']))
        return json.load(open(self.oracle_doc_file))

    # sample 20%
    def sample_dataset(self):
        import data.DS1000.ds1000
        import random

        random.seed(0)

        ds1000 = DS1000Dataset(source_dir='../data/DS1000/ds1000_data', libs='all', mode='Completion')

        sampled_idx_dict = dict()
        sampled_data_list = list()
        for lib in ds1000.libs:
            num_sampled = int(len(ds1000[lib]) * 0.2)
            problem_id_list = list(range(0, len(ds1000[lib])))
            sampled_idx = random.sample(problem_id_list, num_sampled)
            sampled_idx_dict[lib] = sampled_idx
            print(len(sampled_idx))
            for idx in sampled_idx:
                sampled_data = ds1000[lib][idx]
                sampled_data_list.append(dict(qs_id=f'{lib}_{idx}',
                                              reference_code=sampled_data['reference_code'],
                                              code_context=sampled_data['code_context'],
                                              prompt=sampled_data['prompt']))

        with open('../data/DS1000/sampled_data.json', 'w+') as f:
            json.dump(sampled_data_list, f, indent=2)

    def test_helper(self, problem_code_pair: Tuple[DS1000Problem, List[str]]):
        problem, code_list = problem_code_pair
        test_results = []
        for code in code_list:
            test_results.append(problem.test(code))
        return test_results

    def eval_passk(self, result_list, k_list, num_procs=16):
        ds1000 = DS1000Dataset(source_dir=root_path + '/data/DS1000/ds1000_data', mode='Completion', libs='all')
        # process generate code
        processed_gene_codes = dict()
        for result in result_list:
            gene_codes = result['outputs']
            [lib, problem_id] = result['qs_id'].split('_')
            problem_id = int(problem_id)
            if lib not in processed_gene_codes: processed_gene_codes[lib] = []
            processed_gene_codes[lib].append((ds1000[lib][problem_id], gene_codes))

        total_pass_score = dict()
        for lib in processed_gene_codes.keys():
            lib_results = list()
            if num_procs > 1 and lib != "Sklearn":
                with Pool(processes=num_procs) as pool:
                    for problem_id, test_results in tqdm(
                            enumerate(pool.imap(self.test_helper, processed_gene_codes[lib])),
                            total=len(processed_gene_codes[lib]),
                            desc=f"Executing test for {lib} questions",
                    ):
                        lib_results.append(test_results)
                        for result in result_list:
                            [result_lib, result_problem_id] = result['nl']['qs_id'].split('_')
                            if result_lib == lib and int(result_problem_id) == problem_id:
                                result['test_results'] = test_results
            else:
                for problem_id, problem_code_pair in tqdm(enumerate(processed_gene_codes[lib])):
                    test_results = self.test_helper(problem_code_pair)
                    lib_results.append(test_results)
                    for result in result_list:
                        [result_lib, result_problem_id] = result['nl']['qs_id'].split('_')
                        if result_lib == lib and int(result_problem_id) == problem_id:
                            result['test_results'] = test_results

            pass_scores = self.pass_rate(lib_results, k_list)
            total_pass_score[lib] = pass_scores
        avg_pass_score = {key: 0 for key in pass_scores}
        for lib in total_pass_score.keys():
            print(f'{lib} pass score: {total_pass_score[lib]}')
            for key in avg_pass_score.keys():
                avg_pass_score[key] += total_pass_score[lib][key]
        for key in avg_pass_score.keys():
            avg_pass_score[key] = avg_pass_score[key] / len(total_pass_score.keys())
        print(f'avg pass score: {avg_pass_score}')

    @staticmethod
    def pass_rate(results_list, k_list):
        def pass_at_k(n, c, k):
            import numpy as np
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
    ds1000_loader = DS1000Loader()
    oracle_list = ds1000_loader.load_oracle_list()
    preds = [dict(qs_id=oracle['qs_id'], outputs=[oracle['output']]) for oracle in oracle_list]
    ds1000_loader.eval_passk(result_list=preds, k_list=[1])
