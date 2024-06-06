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
from dataset_utils.DS1000_utils import DS1000Loader
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader


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


def code_eval(args):
    gene_results = json.load(open(args.save_file, 'r'))

    if args.dataset == 'conala':
        _gene_results = list()
        for result in gene_results:
            outputs = process_gene_results(args, result['outputs'])
            # outputs = [result['oracle_output']]
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
        passk = ConalaLoader().eval_passk(_gene_results, top_k=[1])

    elif args.dataset == 'DS1000':
        gene_results = json.load(open(DS1000Loader().oracle_doc_file, 'r'))
        for i in range(len(gene_results)):
            gene_results[i]['outputs'] = [gene_results[i]['output']]
        passk = DS1000Loader().eval_passk(gene_results, k_list=[1])

    elif args.dataset == 'pandas_numpy_eval':
        _gene_results = []
        oracle_list = PandasNumpyEvalLoader().load_oracle_list()
        for oracle in oracle_list:
            _gene_results.append(dict(qs_id=oracle['qs_id'], outputs=[oracle['output']]))
        passk = PandasNumpyEvalLoader().eval_passk(_gene_results, k_list=[1])


    return passk


if __name__ == '__main__':
    in_program_call = '--model codellama-13b-instruct --dataset pandas_numpy_eval --retriever openai-embedding --analysis_type retrieval_recall --ret_acc 1'
    # in_program_call = None
    args = generate_config(in_program_call)

    passk = code_eval(args)

    # gene_results = json.load(open(args.save_file, 'r'))
    # for result in gene_results:
    #     # print(result['outputs'])
    #     outputs = process_gene_results(args, result['outputs'])
    #     print(outputs)
