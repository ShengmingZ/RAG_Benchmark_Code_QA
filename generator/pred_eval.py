import os.path
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
from dataset_utils.hotpotQA_utils import HotpotQAUtils
from retriever.retriever_utils import ret_eval_by_doc_keys



def process_gene_results(args, outputs, code_prompt=None):
    preds = []
    if args.dataset == 'conala':
        if 'llama' in args.model:
            for output in outputs:
                if output.startswith(' '): output = output[1:]
                pred = output.replace('<code>', '')
                try:
                    pred = pred.split('</code>')[0]
                except: ...
                if '`' in pred:
                    lines = pred.split('\n')
                    try:
                        _pred = lines[1]
                        pred = _pred
                    except: ...
                preds.append(pred)
        elif 'gpt' in args.model:
            for output in outputs:
                pred = output.replace('<code>', '').replace('</code>', '').replace('\n', '')
                preds.append(pred)
        else:
            raise Exception('Unknown model')

    elif args.dataset == 'DS1000':
        if 'llama' in args.model:
            for output in outputs:
                # first extract code
                if output.startswith(' '): output = output[1:]
                pred = output
                pred = pred.replace('```python', '```').replace('BEGIN SOLUTION', '').replace(' BEGIN SOLUTION', '')
                try:
                    pred = pred.split('<code>')[1]
                except: ...
                try:
                    pred = pred.split('</code>')[0]
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
        elif 'gpt' in args.model:
            for output in outputs:
                pred = output
                try:
                    pred = pred.split('BEGIN SOLUTION')[1]
                except: ...
                try:
                    pred = pred.split('END SOLUTION')[0]
                except: ...
                pred = pred.replace('<code>', '').replace('</code>', '')
                preds.append(pred)
        else:
            raise NotImplementedError('Unknown model')


    elif args.dataset == 'pandas_numpy_eval':
        if 'llama' in args.model:
            for output in outputs:
                # first extract code
                if output.startswith(' '): output = output[1:]
                pred = output
                pred = pred.replace('</s>', '').replace('```python', '```')
                try:
                    pred = pred.split('<code>')[1]
                except: ...
                try:
                    pred = pred.split('</code>')[0]
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
        elif 'gpt' in args.model:   # gpt
            for output in outputs:
                pred = output.replace('<code>\n', '').replace('<code>', '').replace('</code>', '')
                # remove "df =" dup
                prompt_lines = code_prompt.split('\n')
                prompt_lines = [line for line in prompt_lines if line]
                prompt_lines = [line for line in prompt_lines if not line.startswith('#') and not line.startswith('    #')]
                output_lines = pred.split('\n')
                output_lines = [line for line in output_lines if line]
                output_lines = [line for line in output_lines if not line.startswith('#') and not line.startswith('    #')]
                if output_lines[0].startswith(prompt_lines[-1]):
                    output_lines[0] = output_lines[0].replace(prompt_lines[-1], '')
                    pred = '\n'.join(output_lines)
                preds.append(pred)
        else:
            raise ValueError('Unrecognized model: {}'.format(args.model))

    elif args.dataset == 'NQ' or args.dataset == 'TriviaQA' or args.dataset == 'hotpotQA':
        for output in outputs:
            try:
                pred = output.split('<answer>')[1].split('</answer>')[0]
            except: pred = output
            preds.append(pred)

    else:
        raise Exception('Not Implemented')

    return preds


def pred_eval(args):
    eval_save_file = args.result_save_file.replace('.json', '_eval.json')
    if os.path.exists(eval_save_file):
        print('eval file exists already, {}'.format(eval_save_file))
        eval_results = json.load(open(eval_save_file, 'r'))
        print(eval_results['scores'])
        return

    gene_results = json.load(open(args.result_save_file, 'r'))
    if args.n == 10:
        k_list = [1,3,5,10]
    elif args.n == 1:
        k_list = [1]
    else:
        raise ValueError('args.n must be 1 or 10')

    if args.dataset == 'conala':
        loader = ConalaLoader()
        _gene_results = list()
        for result in gene_results:
            outputs = process_gene_results(args, result['outputs'])
            # outputs = [result['oracle_output']]
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
        scores, eval_records = loader.eval_passk(_gene_results, top_k=k_list)

    elif args.dataset == 'DS1000':
        # gene_results = json.load(open(DS1000Loader().oracle_doc_file, 'r'))
        loader = DS1000Loader()
        qs_list = loader.load_qs_list()
        _gene_results = list()
        for idx, result in enumerate(gene_results):
            assert qs_list[idx]['qs_id'] == result['qs_id']
            outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1])
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
        scores, eval_records = loader.eval_passk(_gene_results, k_list=k_list)

    elif args.dataset == 'pandas_numpy_eval':
        loader = PandasNumpyEvalLoader()
        qs_list = loader.load_qs_list()
        _gene_results = []
        for idx, result in enumerate(gene_results):
            assert qs_list[idx]['qs_id'] == result['qs_id']
            outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'])
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
        scores, eval_records = loader.eval_passk(_gene_results, k_list=k_list)

    elif args.dataset == 'NQ' or args.dataset == 'TriviaQA':
        loader = NQTriviaQAUtils(args.dataset)
        oracle_list = loader.load_oracle_list()
        preds, answers_list = [], []
        for result, oracle in zip(gene_results, oracle_list):
            assert str(result['qs_id']) == str(oracle['qs_id'])
            preds.append(process_gene_results(args, result['outputs'])[0])  # Todo: now only 1 inference
            answers_list.append(oracle['answers'])
        scores, _eval_records = loader.pred_eval(preds=preds, answers_list=answers_list)
        eval_records = dict()
        for idx, oracle in enumerate(oracle_list):
            eval_records[oracle['qs_id']] = _eval_records[idx]

    elif args.dataset == 'hotpotQA':
        loader = HotpotQAUtils()
        oracle_list = loader.load_oracle_list()
        pred_list = []
        for result in gene_results:
            output = process_gene_results(args, result['outputs'])[0]
            pred_list.append(dict(qs_id=result['qs_id'], output=output))
        scores, eval_records = loader.eval_pred(pred_list=pred_list, oracle_list=oracle_list)

    else:
        raise ValueError('Not supported dataset {}'.format(args.dataset))

    # prompt length, only calc when prompt save file exists
    # if os.path.exists(args.prompt_save_file):
    #     total_prompt_length = 0
    #     with open(args.prompt_save_file, 'r') as f:
    #         for line in f:
    #             total_prompt_length += json.loads(line)['prompt_length']
    #     avg_prompt_length = total_prompt_length / len(gene_results)
    #     assert isinstance(scores, dict)
    #     scores['avg_prompt_length'] = avg_prompt_length
    ret_doc_keys_list, prompts, pl_list = [], [], []
    with open(args.prompt_save_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if len(data['ret_doc_keys']) != 0: ret_doc_keys_list.append(data['ret_doc_keys'])
            prompts.append(data['prompt'])
            pl_list.append(data['prompt_length'])
    if len(ret_doc_keys_list) != 0:
        oracle_list = loader.load_oracle_list()
        ret_doc_key_flags_list, avg_ret_recall, avg_oracle_percent, avg_oracle_rank = ret_eval_by_doc_keys(dataset=args.dataset, oracle_list=oracle_list, ret_doc_keys_list=ret_doc_keys_list)
        # print('ret recall: ', avg_ret_recall)
        # print('avg oracle doc percentage: ', avg_oracle_percent)
        # print('avg oracle doc rank: ', avg_oracle_rank + 1)  # rank start from 1
        # print('avg prompt length: ', sum(pl_list) / len(pl_list))
        scores['ret_recall'] = avg_ret_recall
        scores['oracle_percent'] = avg_oracle_percent
        scores['oracle_rank'] = avg_oracle_rank
    scores['prompt_length'] = sum(pl_list) / len(pl_list)
    scores = {key: round(value, 3) for key, value in scores.items() if value is not None}
    print(scores)
    with open(eval_save_file, 'w') as f:
        json.dump(dict(scores=scores, eval_records=eval_records), f, indent=2)

    return scores


if __name__ == '__main__':
    in_program_call = None
    # in_program_call = '--model llama2-13b-chat --dataset hotpotQA --retriever openai-embedding --analysis_type retrieval_recall --ret_acc 1'
    # in_program_call = '--model llama2-13b-chat --dataset NQ --retriever openai-embedding --analysis_type retrieval_recall --ret_acc 1'
    args = generate_config(in_program_call)

    scores = pred_eval(args)

    """
    test process outputs for DS1000
    """
    # ds1000 = DS1000Dataset(source_dir='../data/DS1000/ds1000_data', libs='all', mode='Insertion')
    # gene_results = json.load(open(args.result_save_file, 'r'))
    # loader = DS1000Loader()
    # oracle_list = loader.load_oracle_list()
    # qs_list = loader.load_qs_list()
    # for idx, result in enumerate(gene_results):
    #     qs_id = result['qs_id']
    #     [lib, problema_id] = qs_id.split('_')
    #     data = ds1000[lib][int(problema_id)]
    #     print(f'<processed code {idx}>]\n')
    #     print([result['outputs'][0]])
    #     outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1])
    #     print([outputs[0]])

    """
    test process outputs for pandas_numpy_eval
    """
    # dataset = json.load(open('../data/pandas_numpy_eval/data/pandas_numpy_eval.json', 'r'))
    # gene_results = json.load(open(args.result_save_file, 'r'))
    # for idx, result in enumerate(gene_results):
    #     print(f'<processed code {idx}>]\n')
    #     print([result['outputs'][0]])
    #     for data in dataset:
    #         if data['task_id'] == result['qs_id']:
    #             code_prompt = data['prompt']
    #     print(code_prompt)
    #     outputs = process_gene_results(args, result['outputs'], code_prompt)
    #     print([outputs[0]])

    """
    test for conala
    """
    # gene_results = json.load(open(args.result_save_file, 'r'))
    # for idx, result in enumerate(gene_results):
    #     print(f'<processed code {idx}>]\n')
    #     print([result['outputs'][0]])
    #     outputs = process_gene_results(args, result['outputs'])
    #     print([outputs[0]])

    """
    test for NQ, TriviaQA
    """
    # gene_results = json.load(open(args.result_save_file, 'r'))
    # for idx, result in enumerate(gene_results):
    #     print(f'<processed code {idx}>]\n')
    #     print([result['outputs'][0]])
    #     outputs = process_gene_results(args, result['outputs'])
    #     print([outputs[0]])
