import json
from multiprocessing import Pool
from tqdm import tqdm
from typing import Tuple, List
import numpy as np
# import local packages
import sys, platform
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from data.DS1000.ds1000 import DS1000Dataset, DS1000Problem, ScoreRecord, check_version, check_cpu_count
from generator.generate_utils import generate_config
from dataset_utils.dataset_configs import DS1000Loader


def pass_rate(results, k_list):
    def pass_at_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if k_list == [1]:
        pass_scores = dict(pass_1=0)
    elif k_list == [1, 5, 10, 50, 100]:
        pass_scores = dict(pass_1=0, pass_5=0, pass_10=0, pass_50=0, pass_100=0)
    for result in results:
        passed_num = result.count(True)
        total_num = len(result)
        if k_list == [1]:
            pass_scores['pass_1'] += pass_at_k(n=total_num, c=passed_num, k=1)
        elif k_list == [1, 5, 10, 50, 100]:
            pass_scores['pass_1'] += pass_at_k(n=total_num, c=passed_num, k=1)
            pass_scores['pass_5'] += pass_at_k(n=total_num, c=passed_num, k=5)
            pass_scores['pass_10'] += pass_at_k(n=total_num, c=passed_num, k=10)
            pass_scores['pass_50'] += pass_at_k(n=total_num, c=passed_num, k=50)
            pass_scores['pass_100'] += pass_at_k(n=total_num, c=passed_num, k=100)
    for key in pass_scores.keys():
        if len(results) == 0: pass_scores[key] = 0
        pass_scores[key] = pass_scores[key]/len(results)

    return pass_scores


def test_helper(problem_code_pair: Tuple[DS1000Problem, List[str]]):
    problem, code_list = problem_code_pair
    test_results = []
    for code in code_list:
        test_results.append(problem.test(code))
    return test_results

def ds1000_passk(result_file, mode='Completion', num_procs=16):
    check_version()
    result_list = json.load(open(result_file, 'r'))
    ds1000 = DS1000Dataset(source_dir=root_path+'/data/DS1000/ds1000_data', mode=mode, libs='all')
    # process generate code
    processed_gene_codes = dict()
    for result in result_list:
        gene_codes = result['outputs']
        [lib, problem_id] = result['nl']['qs_id'].split('_')
        problem_id = int(problem_id)
        if lib not in processed_gene_codes: processed_gene_codes[lib] = []
        processed_gene_codes[lib].append((ds1000[lib][problem_id], gene_codes))

    total_pass_score = dict()
    for lib in processed_gene_codes.keys():
        lib_results = list()
        if num_procs > 1 and lib != "Sklearn":
            with Pool(processes=num_procs) as pool:
                for problem_id, test_results in tqdm(
                        enumerate(pool.imap(test_helper, processed_gene_codes[lib])),
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
                lib_results.append(test_helper(problem_code_pair))
                for result in result_list:
                    [result_lib, result_problem_id] = result['nl']['qs_id'].split('_')
                    if result_lib == lib and int(result_problem_id) == problem_id:
                        result['test_results'] = test_results

        if len(processed_gene_codes[lib][0][1]) == 1:
            pass_scores = pass_rate(lib_results, [1])
        elif len(processed_gene_codes[lib][0][1]) == 100:
            pass_scores = pass_rate(lib_results, [1, 5, 10, 50, 100])
        else:
            raise Exception('unexpected n')
        total_pass_score[lib] = pass_scores
    avg_pass_score = {key: 0 for key in pass_scores}
    for lib in total_pass_score.keys():
        print(f'{lib} pass score: {total_pass_score[lib]}')
        for key in avg_pass_score.keys():
            avg_pass_score[key] += total_pass_score[lib][key]
    for key in avg_pass_score.keys():
        avg_pass_score[key] = avg_pass_score[key] / len(total_pass_score.keys())
    print(f'avg pass score: {avg_pass_score}')

    with open(result_file.replace('.json', '_test_results.json'), 'w+') as f:
        json.dump(result_list, f, indent=2)



if __name__ == '__main__':
    in_program_call = '--dataset ds1000 --top_k 1 --retriever bm25 --ret_doc_type oracle --prompt_type original --n 1'
    args = generate_config(in_program_call)
    ds1000_passk(result_file=args.save_file)

    # ds1000 = DS1000Dataset(source_dir=root_path + '/data/DS1000/ds1000_data', mode='Completion', libs='all')
    # test_data = ds1000['Numpy'][20]
    # outputs = [
    #     "import numpy as np\na = np.array([[10,50,30],[60,20,40]])\nresult = np.unravel_index(np.argmax(a), a.shape, order='F')\n"
    # ]
    # outputs = [
    #     "result = np.unravel_index(np.argmax(a), a.shape, order='F')\n"
    # ]
    # outputs = ["result = np.unravel_index(a.argmax(), a.shape, order = 'F')\n"]
    # outputs = [
    #   "result = np.unravel_index(np.argmax(a, axis=None), a.shape, order='F')\n"
    # ]
    # test_result = test_helper((test_data, outputs))
    # print(test_result)

    # oracle_save_file = args.save_file.replace('.json', '_test_results.json')
    # none_save_file = oracle_save_file.replace('oracle', 'none')
    # oracle_results = json.load(open(oracle_save_file, 'r'))
    # none_results = json.load(open(none_save_file, 'r'))
    # _none_results = list()
    # for result in none_results:
    #     if result['nl']['qs_id'].split('_')[0].lower() == 'scipy': continue
    #     _none_results.append(result)
    # oracle_win_count = 0
    # none_win_count = 0
    # for (oracle_result, none_result) in zip(oracle_results, _none_results):
    #     assert oracle_result['nl']['qs_id'] == none_result['nl']['qs_id'], print(oracle_result['nl']['qs_id'], none_result['nl']['qs_id'])
    #     if oracle_result['test_results'][0] is True and none_result['test_results'][0] is False:
    #         oracle_win_count += 1
    #     elif oracle_result['test_results'][0] is False and none_result['test_results'][0] is True:
    #         none_win_count += 1
    #         print(oracle_result['nl']['qs_id'])
    #         print(oracle_result['oracle_libs'])
    #         print(oracle_result['nl']['nl'], end='\n\n')
    #         print('oracle output:', oracle_result['outputs'][0], end='\n\n')
    #         print('none output:', none_result['outputs'][0], end='\n\n\n\n')
    #
    # print('oracle win count:', oracle_win_count)
    # print('none win count:', none_win_count)

