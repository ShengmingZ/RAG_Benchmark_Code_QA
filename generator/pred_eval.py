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
from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils


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
            _pred_lines = []
            for pred_line in pred_lines:
                if pred_line not in prompt_lines:
                    _pred_lines.append(pred_line)
            pred = '\n'.join(_pred_lines)
            preds.append(pred)

    elif args.dataset == 'pandas_numpy_eval' and args.model == 'codellama-13b-instruct':
        for output in outputs:
            # first extract code
            if output.startswith(' '): output = output[1:]
            pred = output
            pred = pred.replace('</s>', '').replace('```python', '```')
            try:
                pred = pred.split('<code>')[1]
            except: ...
            try:
                pred = pred.split('```')[1].split('```')[0]
            except: ...
            try:
                pred = pred.split('# Example usage')[0]
            except: ...
            try:
                pred = pred.split('[out]')[0]
            except: ...
            pred = pred.replace('`', '')
            # remove dup
            prompt_lines = code_prompt.split('\n')
            prompt_lines = [line for line in prompt_lines if line]
            prompt_lines = [line for line in prompt_lines if not line.startswith('#') and not line.startswith('    #')]
            code_prompt = '\n'.join(prompt_lines)
            output_lines = pred.split('\n')
            output_lines = [line for line in output_lines if line]
            output_lines = [line for line in output_lines if not line.startswith('#') and not line.startswith('    #')]
            output = '\n'.join(output_lines)
            if output.startswith(code_prompt):
                pred = output.replace(code_prompt, '')
            elif prompt_lines[-1].startswith('def'):
                if output_lines[0].startswith('def'):   # duplicate def, remove
                    output_lines = output_lines[1:]
                elif not output_lines[0].startswith('    '):    # add indent
                    output_lines = ['    ' + line for line in output_lines]
                # if 'return' not in output:
                #     output_lines[-1] = output_lines[-1].replace('    ', '    return ')
                pred = '\n'.join(output_lines)
            elif output_lines[0].startswith(prompt_lines[-1]):  # only last line dup
                output_lines[0] = output_lines[0].replace(prompt_lines[-1], '')
                pred = '\n'.join(output_lines)
            elif prompt_lines[-1] != '    ' and output_lines[0].startswith('return'):
                output_lines[0] = '    ' + output_lines[0]
                pred = '\n'.join(output_lines)
            preds.append(pred)

    elif args.dataset == 'NQ' or args.dataset == 'TriviaQA':
        for output in outputs:
            try:
                pred = output.split('<answer>')[1]
            except: pred = output
            preds.append(pred)

    else:
        raise Exception('Not Implemented')

    return preds


def code_eval(args):
    gene_results = json.load(open(args.save_file, 'r'))
    if args.n == 10:
        k_list = [1,3,5,10]
    elif args.n == 1:
        k_list = [1]
    else:
        raise ValueError('args.n must be 1 or 10')

    if args.dataset == 'conala':
        _gene_results = list()
        for result in gene_results:
            outputs = process_gene_results(args, result['outputs'])
            # outputs = [result['oracle_output']]
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
        scores = ConalaLoader().eval_passk(_gene_results, top_k=k_list)

    elif args.dataset == 'DS1000':
        # gene_results = json.load(open(DS1000Loader().oracle_doc_file, 'r'))
        qs_list = DS1000Loader().load_qs_list()
        _gene_results = list()
        for idx, result in enumerate(gene_results):
            assert qs_list[idx]['qs_id'] == result['qs_id']
            outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1])
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
        scores = DS1000Loader().eval_passk(_gene_results, k_list=k_list)

    elif args.dataset == 'pandas_numpy_eval':
        qs_list = PandasNumpyEvalLoader().load_qs_list()
        _gene_results = []
        for idx, result in enumerate(gene_results):
            assert qs_list[idx]['qs_id'] == result['qs_id']
            outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'])
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
        scores = PandasNumpyEvalLoader().eval_passk(_gene_results, k_list=k_list)

    elif args.dataset == 'NQ' or args.dataset == 'TriviaQA':
        loader = NQTriviaQAUtils(args.dataset)
        oracle_list = loader.load_oracle_list()
        preds, answers_list = [], []
        for result, oracle in zip(gene_results, oracle_list):
            assert result['qs_id'] == oracle['qs_id']
            preds.append(process_gene_results(args, result['outputs'])[0])  # Todo: now only 1 inference
            answers_list.append(oracle['answers'])
        scores = loader.pred_eval(preds=preds, answers_list=answers_list)

    else:
        raise ValueError('Not supported dataset {}'.format(args.dataset))

    return scores


if __name__ == '__main__':
    in_program_call = None
    # in_program_call = '--model codellama-13b-instruct --dataset pandas_numpy_eval --retriever openai-embedding --analysis_type retrieval_recall --ret_acc 1'
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

    """
    test process outputs for pandas_numpy_eval
    """
    # dataset = json.load(open('../data/pandas_numpy_eval/data/pandas_numpy_eval.json', 'r'))
    # gene_results = json.load(open(args.save_file, 'r'))
    # for idx, result in enumerate(gene_results):
    #     print(f'<processed code {idx}>]\n')
    #     print([result['outputs'][0]])
    #     for data in dataset:
    #         if data['task_id'] == result['qs_id']:
    #             code_prompt = data['prompt']
    #     outputs = process_gene_results(args, result['outputs'], code_prompt)
    #     print([outputs[0]])
    #     # print([code_prompt])
    #     # prompt_lines = code_prompt.split('\n')
    #     # prompt_lines = [line for line in prompt_lines if line]
    #     # prompt_lines = [line for line in prompt_lines if not line.startswith('#') and not line.startswith('    #')]
    #     # code_prompt = '\n'.join(prompt_lines)
    #     # output_lines = outputs[0].split('\n')
    #     # output_lines = [line for line in output_lines if line]
    #     # output_lines = [line for line in output_lines if not line.startswith('#') and not line.startswith('    #')]
    #     # output = '\n'.join(output_lines)
    #     # if not output.startswith(code_prompt):
    #     #     print(f'\n{result["qs_id"]}')
    #     #     print(code_prompt)
    #     #     print('<output>')
    #     #     print(output)
    #     #     print(result['outputs'])
    #     # if prompt_lines[-1].startswith('#'):
    #     #     continue
    #     # elif prompt_lines[-1].startswith('    #'):
    #     #     continue
    #     # elif prompt_lines[-1].startswith('    '):
    #     #     continue
    #     # elif prompt_lines[-1].endswith('return ') or prompt_lines[-1].endswith('= ') or prompt_lines[-1].endswith('='):
    #     #     continue
    #     # else:
    #     #     print([code_prompt])
