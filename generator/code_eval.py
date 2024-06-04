import platform
import sys
import json
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_utils.conala_utils import ConalaLoader
from generator.generate_utils import generate_config


def conala_eval(args):
    gene_results = json.load(open(args.save_file, 'r'))
    _gene_results = list()
    for result in gene_results:
        outputs = [output.replace('<code>', '') for output in result['outputs']]
        # outputs = [result['oracle_output']]
        _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
    passk = ConalaLoader().eval_passk(_gene_results, top_k=[1])
    return passk


if __name__ == '__main__':
    # in_program_call = '--model llama3-8b --dataset conala --retriever BM25 --analysis_type retrieval_recall --ret_acc 1'
    in_program_call = None
    args = generate_config(in_program_call)
    passk = conala_eval(args)
    save_file = args.save_file.replace('.json', 'eval.json')