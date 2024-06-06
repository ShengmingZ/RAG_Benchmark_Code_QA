import platform
import sys
import json
import re
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_utils.conala_utils import ConalaLoader
from generator.generate_utils import generate_config


def process_gene_results(args, outputs):
    if args.dataset == 'conala' and args.model == 'codellama-13b-instruct':
        preds = []
        for output in outputs:
            if output.startswith(' '): output = output[1:]
            pred = output.replace('<code>', '')
            if '`' in pred:
                lines = pred.split('\n')
                try:
                    _pred = lines[1]
                    pred = _pred
                except: ...
            # if len(lines) > 1: pred = lines[1].split('\n```')[0]
            # lines = pred.split('`')
            # if len(lines) > 1: pred = lines[1].split('`')[0]
            preds.append(pred)

    return preds


def conala_eval(args):
    gene_results = json.load(open(args.save_file, 'r'))
    _gene_results = list()
    for result in gene_results:
        outputs = process_gene_results(args, result['outputs'])
        # outputs = [result['oracle_output']]
        _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
    passk = ConalaLoader().eval_passk(_gene_results, top_k=[1])
    return passk


if __name__ == '__main__':
    in_program_call = '--model codellama-13b-instruct --dataset conala --retriever openai-embedding --analysis_type retrieval_recall --ret_acc 0.8'
    # in_program_call = None
    args = generate_config(in_program_call)

    passk = conala_eval(args)

    # gene_results = json.load(open(args.save_file, 'r'))
    # for result in gene_results:
    #     # print(result['outputs'])
    #     outputs = process_gene_results(args, result['outputs'])
    #     print(outputs)
