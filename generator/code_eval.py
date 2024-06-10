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
from data.DS1000.ds1000 import DS1000Dataset
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader


def process_gene_results(args, outputs, code_prompt=None):
    preds = []
    if args.dataset == 'conala' and args.model == 'codellama-13b-instruct':
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
    elif args.dataset == 'DS1000' and args.model == 'codellama-13b-instruct':
        loader = DS1000Loader()
        qs_list = loader.load_qs_list()
        for output in outputs:
            # first extract code
            if output.startswith(' '): output = output[1:]
            pred = output
            pred = pred.replace('```python', '```').replace('BEGIN SOLUTION', '').replace(' BEGIN SOLUTION', '')
            try:
                pred = pred.split('<code>')[1]
            except: ...
            try:
                pred = pred.split('```')[1].split('```')[0]
            except: ...
            # then remove dup
            prompt_lines = code_prompt.split('\n')
            pred_lines = pred.split('\n')
            print(prompt_lines)
            print(pred_lines)
            _pred_lines = []
            for pred_line in pred_lines:
                if pred_line not in prompt_lines:
                    _pred_lines.append(pred_line)
            pred = '\n'.join(_pred_lines)
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
        # gene_results = json.load(open(DS1000Loader().oracle_doc_file, 'r'))
        qs_list = DS1000Loader().load_qs_list()
        _gene_results = list()
        for idx, result in enumerate(gene_results):
            assert qs_list[idx]['qs_id'] == result['qs_id']
            outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1])
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
        passk = DS1000Loader().eval_passk(_gene_results, k_list=[1])

    elif args.dataset == 'pandas_numpy_eval':
        _gene_results = []
        oracle_list = PandasNumpyEvalLoader().load_oracle_list()
        for oracle in oracle_list:
            _gene_results.append(dict(qs_id=oracle['qs_id'], outputs=[oracle['output']]))
        passk = PandasNumpyEvalLoader().eval_passk(_gene_results, k_list=[1])


    return passk


if __name__ == '__main__':
    in_program_call = None
    # in_program_call = '--model codellama-13b-instruct --dataset DS1000 --retriever openai-embedding --analysis_type retrieval_recall --ret_acc 1'
    args = generate_config(in_program_call)

    passk = code_eval(args)

    """
    test process outputs for DS1000
    """
    # ds1000 = DS1000Dataset(source_dir='../data/DS1000/ds1000_data', libs='all', mode='Insertion')
    # gene_results = json.load(open(args.save_file, 'r'))
    # loader = DS1000Loader()
    # oracle_list = loader.load_oracle_list()
    # qs_list = loader.load_qs_list()
    # for idx, result in enumerate(gene_results):
    #     qs_id = result['qs_id']
    #     [lib, problema_id] = qs_id.split('_')
    #     data = ds1000[lib][int(problema_id)]
    #     print(f'<processed code {idx}>]\n')
    #     # print([result['outputs'][0]])
    #     outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1])
    #     print([outputs[0]])
    #     # print([qs_list[idx]['question'].split('\nA:')[1]])
    #     # print(oracle_list[idx+1]['output'])
    #     # print(data.keys())
    #     # print(data['test_code'])
    #     # print(data['code_context'])
