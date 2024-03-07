import json
from multiprocessing import Pool
from tqdm import tqdm
from typing import Tuple, List
# import local packages
import sys, platform
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
sys.path.insert(0, root_path + '/DS-1000')
from ds1000 import DS1000Dataset, DS1000Problem, ScoreRecord, check_version, check_cpu_count
from generator.generate import generate_config
from dataset_utils.dataset_configs import DS1000Loader
from generator.run_model import chatgpt


def test_helper(problem_code_pair: Tuple[DS1000Problem, List[str]]):
    problem, code_list = problem_code_pair
    test_results = []
    for code in code_list:
        test_results.append(problem.test(code))
    return test_results

def ds1000_passk(save_file, mode='completion', num_procs=16):
    result_list = json.load(open(save_file, 'r'))
    ds1000 = DS1000Dataset(source_dir='../DS-1000/ds1000_data', mode=mode, libs='all')
    # process generate code
    processed_gene_codes = dict()
    for result in result_list:
        gene_codes = result['outputs']
        [lib, problem_id] = result['qs_id'].split('_')
        problem_id = int(problem_id)
        if lib not in processed_gene_codes: processed_gene_codes[lib] = []
        processed_gene_codes[lib].append((ds1000[lib][problem_id], gene_codes))

    for lib in processed_gene_codes.keys():
        lib_results = []
        if num_procs > 1 and lib != "Sklearn":
            with Pool(processes=num_procs) as pool:
                for test_results in tqdm(
                        pool.imap(test_helper, processed_gene_codes[lib]),
                        total=len(ds1000[lib]),
                        desc=f"Executing test for {lib} questions",
                ):
                    lib_results.append(test_results)
        else:
            for problem_code_pair in tqdm(processed_gene_codes[lib]):
                lib_results.append(test_helper(problem_code_pair))




if __name__ == '__main__':
    # in_program_call = '--dataset ds1000 --top_k 1 --k_line 10 --retriever bm25 --ret_doc_type none'
    # args = generate_config(in_program_call)
    #
    # # ds1000_passk(args.save_file)
    ds1000 = DS1000Dataset(source_dir=root_path+'/DS-1000/ds1000_data', mode='completion', libs='all')
    #
    # ds1000_loader = DS1000Loader()
    # qs_list = ds1000_loader.load_qs_list()
    # oracle_list = ds1000_loader.load_oracle_list()
    # qs, oracle = qs_list[0], oracle_list[0]
    # [lib, problem_id] = qs['qs_id'].split('_')
    # problem_id = int(problem_id)
    # prompt = qs['nl']
    # output = chatgpt(prompt=prompt, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens, stop=["</code>", "# SOLUTION END"], n=10)
    # print(len(output))
    # print(output)
    lib = 'Pandas'
    problem_id = 0
    print(ds1000[lib][problem_id]['reference_code'])

    result = test_helper((ds1000[lib][problem_id], [ds1000[lib][problem_id]['reference_code']]))
    print(result)