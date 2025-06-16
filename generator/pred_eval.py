import os.path
import platform
import sys
import json
import re
import numpy as np
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_utils.conala_utils import ConalaLoader
from generator.generate_utils import generate_config, get_docs_tokens
from dataset_utils.DS1000_utils import DS1000Loader
from data.DS1000.ds1000 import DS1000Dataset
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
from dataset_utils.hotpotQA_utils import HotpotQAUtils
from retriever.retriever_utils import ret_eval_by_doc_keys
from data_processing.analyze_result import analyze_results_for_code


def conala_result_process(prompt_type, output, output_before=None):
    pred = output
    if prompt_type == 'self-refine' and '```' not in output and '<code>' not in output: pred = output_before   # if no refine in self-refine, just use output before
    pred = pred.replace('</s>', '').replace('```python', '```')
    try: pred = pred.split('Potential documents')[0]
    except: ...
    if prompt_type in ['least_to_most']:
        try: pred = pred.rsplit('```', 1)[0].rsplit('```', 1)[1]
        except: ...
    try: pred = pred.split('<code>')[1].split('</code>')[0]
    except: ...
    try: pred = pred.split('```')[1].split('```')[0]
    except: ...
    # only keep one code line
    pred_lines = [line for line in pred.split('\n') if line != '' and not line.startswith('#') and not line.startswith('    #')]
    try:
        if pred_lines[-1].startswith('print'): pred = pred_lines[-2]
        else: pred = pred_lines[-1]
    except: ...
    return pred


def DS1000_result_process(prompt_type, output, code_prompt, output_before=None):
    pred = output
    if prompt_type == 'self-refine' and not '```' in output and not '<code>' in output: pred = output_before
    pred = pred.replace('</s>', '').replace('```python', '```')
    try: pred = pred.split('Potential documents')[0]
    except: ...
    if prompt_type in ['least_to_most']:
        try: pred = pred.rsplit('```', 1)[0].rsplit('```', 1)[1]
        except: ...
    try: pred = pred.split('BEGIN SOLUTION')[1]
    except: ...
    try: pred = pred.split('END SOLUTION')[0]
    except: ...
    try: pred = pred.split('```', 1)[1].split('```', 1)[0]
    except: ...
    try: pred = pred.split('<code>')[1].split('</code>')[0]
    except: ...

    prompt_lines = code_prompt.split('\n')
    prompt_lines = [line for line in prompt_lines if line != '' and not line.startswith('#') and not line.startswith('    #')]
    # print(prompt_lines)
    pred_lines = pred.split('\n')
    pred_lines = [line for line in pred_lines if line != '' and not line.startswith('#') and not line.startswith('    #')]
    # print(pred_lines)

    preload_variables = []  # get pre-defined variables in code prompt
    for prompt_line in prompt_lines:
        if ' = ' in prompt_line: preload_variables.extend([var.replace(' ', '') for var in prompt_line.split('=')[0].split(',')])
        if 'BEGIN SOLUTION' in prompt_line: break
    # if model output full code snippet, need to remove duplicated ones
    # and sometimes LLM would change the definition of preload variables e.g.: softmax_output = load_data() -> softmax_output = torch.tensor([[0.2, 0.1, 0.7], ...
    _pred_lines = []
    for pred_line in pred_lines:
        is_same = False
        # remove dup
        for prompt_line in prompt_lines:
            if prompt_line.replace(' ', '') == pred_line.replace(' ', ''):
                for var in preload_variables:
                    if ' = ' in prompt_line and var in prompt_line.split('=')[0]: preload_variables.remove(var)  # if not change the defi...
                is_same = True
        if not is_same: _pred_lines.append(pred_line)
    pred = '\n'.join(_pred_lines)

    # if model have output full code snippet and change the definition of preload variables
    if len(pred_lines) - len(_pred_lines) >= 2 and len(preload_variables) > 0:
        _pred_lines = []
        pred_lines = pred.split('\n')
        for pred_line in pred_lines:
            is_same = False
            for var in preload_variables:
                if pred_line.startswith(f'{var} ='):
                    preload_variables.remove(var)
                    is_same = True
            if not is_same: _pred_lines.append(pred_line)
        pred = '\n'.join(_pred_lines)
    return pred


def pandas_numpy_eval_result_process(prompt_type, output, code_prompt, output_before=None):
    # first extract code
    pred = output
    if prompt_type == 'self-refine' and not '```' in output and not '<code>' in output: pred = output_before
    if pred.startswith(' '): pred = pred[1:]
    pred = pred.replace('</s>', '').replace('```python', '```')
    try: pred = pred.split('Potential documents')[0]
    except: ...
    if prompt_type in ['least_to_most']:
        try: pred = pred.rsplit('```', 1)[0].rsplit('```', 1)[1]
        except: ...
    try: pred = pred.split('<code>')[1]
    except: ...
    try: pred = pred.split('</code>')[0]
    except: ...
    try: pred = pred.split('```', 1)[1].split('```', 1)[0]
    except: ...
    try: pred = pred.split('# Example usage')[0]
    except: ...
    try: pred = pred.split('[out]')[0]
    except: ...
    pred = pred.replace('`', '')
    # clean code
    prompt_lines = code_prompt.split('\n')
    prompt_lines = [line for line in prompt_lines if line != '' and not line.startswith('#') and not line.startswith('    #')]
    # print(prompt_lines)
    pred_lines = pred.split('\n')
    pred_lines = [line for line in pred_lines if line != '' and not line.startswith('#') and not line.startswith('    #')]
    # print(pred_lines)
    # remove dup lines
    _pred_lines = []
    for pred_line in pred_lines:
        is_same = False
        for prompt_line in prompt_lines:
            if prompt_line.replace(' ', '') == pred_line.replace(' ', ''): is_same = True
        if not is_same: _pred_lines.append(pred_line)
    pred_lines = _pred_lines
    # remove partial dup in the last line
    try:
        if pred_lines[0].startswith(prompt_lines[-1]): pred_lines[0] = pred_lines[0].replace(prompt_lines[-1], '')
    except: ...
    # add indent for return
    try:
        if prompt_lines[-1] != '    ' and pred_lines[0].startswith('return'): pred_lines[0] = '    ' + pred_lines[0]
    except: ...
    # add intent for all pred lines
    try:
        if 'def' in code_prompt and prompt_lines[-1] != '    ' and not pred_lines[0].startswith('    '): pred_lines = ['    ' + line for line in pred_lines]
    except: ...
    # add return for function
    try:
        if 'def' in code_prompt and 'return' not in pred:
            var = pred_lines[-1].split(' = ')[0].replace(' ', '')
            pred_lines.append(f'    return {var}')
    except: ...
    pred = '\n'.join(pred_lines)
    return pred


def process_gene_results(dataset, outputs, prompt_type=None, code_prompt=None, outputs_before=None):
    preds = []
    if dataset == 'conala':
        for idx, output in enumerate(outputs):
            if prompt_type == 'self-refine':
                pred = conala_result_process(prompt_type, output, outputs_before[idx])
            else:
                pred = conala_result_process(prompt_type, output)
            preds.append(pred)

    elif dataset == 'DS1000':
        for idx, output in enumerate(outputs):
            if prompt_type == 'self-refine':
                pred = DS1000_result_process(prompt_type, output, code_prompt, outputs_before[idx])
            else:
                pred = DS1000_result_process(prompt_type, output, code_prompt)
            preds.append(pred)

    elif dataset == 'pandas_numpy_eval':
        for idx, output in enumerate(outputs):
            if prompt_type == 'self-refine':
                pred = pandas_numpy_eval_result_process(prompt_type, output, code_prompt, outputs_before[idx])
            else:
                pred = pandas_numpy_eval_result_process(prompt_type, output, code_prompt)
            preds.append(pred)

    elif dataset == 'NQ' or dataset == 'TriviaQA' or dataset == 'hotpotQA':
        for idx, output in enumerate(outputs):
            pred = output
            if prompt_type == 'RaR':
                try: pred = pred.split('Answer:\n')[1]
                except: ...
                try: pred = pred.split('the answer')[1]
                except: ...
            if prompt_type == 'self-refine':
                if not '<answer>' in output and not '```' in output:
                    pred = outputs_before[idx]
            try: pred = pred.split('Potential documents')[0]
            except: ...
            try: pred = pred.split('<answer>')[1].split('</answer>')[0]
            except: ...
            try: pred = pred.split('```')[1].split('```')[0]
            except: ...
            preds.append(pred)

    else:
        raise Exception('Not Implemented')

    return preds


def process_outputs_for_self_consistency(outputs):
    outputs_dict = dict()
    for output in outputs:
        if output in outputs_dict.keys():
            outputs_dict[output] += 1
        else:
            outputs_dict[output] = 1
    most_output = sorted(outputs_dict.items(), key=lambda item: item[1], reverse=True)[0][0]
    return most_output




def pred_eval_new(dataset, result_path):
    eval_save_file = result_path.replace('.json', '_eval.json')
    results = json.load(open(result_path, 'r'))
    if dataset == 'conala':
        loader = ConalaLoader()
        _gene_results = list()
        for idx, result in enumerate(results):
            # outputs = process_gene_results(dataset, [result['response']])   # only one response is in the key: response
            outputs = [result['response'].replace('<code>', '').replace('</code>', '')]
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
        scores, eval_records = loader.eval_passk(_gene_results, top_k=[1])
        syntax_error_count = 0
        for qid in eval_records:
            if eval_records[qid]['syntax_error']:
                syntax_error_count += 1
        print('number of syntax errors: {}'.format(syntax_error_count))
    elif dataset == 'DS1000':
        loader = DS1000Loader()
        qs_list = loader.load_qs_list()
        _gene_results = list()
        for idx, result in enumerate(results):
            assert qs_list[idx]['qs_id'] == result['qs_id']
            outputs = process_gene_results(dataset, [result['response']], code_prompt=qs_list[idx]['question'].split('\nA:')[1])
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
        scores, eval_records = loader.eval_passk(_gene_results, k_list=[1])
    elif dataset == 'pandas_numpy_eval':
        loader = PandasNumpyEvalLoader()
        qs_list = loader.load_qs_list()
        _gene_results = []
        for idx, result in enumerate(results):
            assert qs_list[idx]['qs_id'] == result['qs_id']
            outputs = process_gene_results(dataset, [result['response']], code_prompt=qs_list[idx]['question'])
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
        scores, eval_records = loader.eval_passk(_gene_results, k_list=[1])
    elif dataset == 'NQ' or dataset == 'TriviaQA':
        loader = NQTriviaQAUtils(dataset)
        oracle_list = loader.load_oracle_list()
        preds, answers_list = [], []
        for idx, (result, oracle) in enumerate(zip(results, oracle_list)):
            assert str(result['qs_id']) == str(oracle['qs_id'])
            pred = process_gene_results(dataset, [result['response']])[0]   # no k trial for QA datasets
            preds.append(pred)
            answers_list.append(oracle['answers'])
        scores, _eval_records = loader.pred_eval(preds=preds, answers_list=answers_list)
        eval_records = dict()
        for idx, oracle in enumerate(oracle_list):
            eval_records[oracle['qs_id']] = _eval_records[idx]
    elif dataset == 'hotpotQA':
        loader = HotpotQAUtils()
        oracle_list = loader.load_oracle_list()
        pred_list = []
        for idx, result in enumerate(results):
            output = process_gene_results(dataset, [result['response']])[0]   # Todo: now only 1 inference
            pred_list.append(dict(qs_id=result['qs_id'], output=output))    # format for eval_pred()
        scores, eval_records = loader.eval_pred(pred_list=pred_list, oracle_list=oracle_list)
    else:
        raise ValueError('Not supported dataset {}'.format(dataset))
    scores = {key: round(value, 3) for key, value in scores.items() if value is not None}
    print(scores)
    with open(eval_save_file, 'w') as f:
        json.dump(dict(scores=scores, eval_records=eval_records), f, indent=2)




def pred_eval(args, if_eval_retrieval=False, if_calc_perplexity=True, if_code_analysis=True, if_save=True):
    eval_save_file = args.result_save_file.replace('.json', '_eval.json')
    # if os.path.exists(eval_save_file):
    #     print('eval file exists already, {}'.format(eval_save_file))
    #     eval_results = json.load(open(eval_save_file, 'r'))
    #     print(eval_results['scores'])
    #     return

    gene_results = json.load(open(args.result_save_file, 'r'))
    if args.prompt_type == 'self-refine': gene_results_before = json.load(open(args.result_save_file.replace('self-refine', '3shot'), 'r'))
    # if args.n == 10:
    #     k_list = [1,3,5,10]
    # elif args.n == 1:
    #     k_list = [1]
    # else:
    #     raise ValueError('args.n must be 1 or 10')
    k_list = [1]

    output_records = dict()
    retrieval_records = dict()



    if args.dataset == 'conala':
        loader = ConalaLoader()
        _gene_results = list()
        for idx, result in enumerate(gene_results):
            if args.prompt_type == 'self-refine': outputs = process_gene_results(args, result['outputs'], outputs_before=gene_results_before[idx]['outputs'])
            elif args.prompt_type == 'self-consistency': outputs = [process_outputs_for_self_consistency(process_gene_results(args, result['outputs']))]
            else: outputs = process_gene_results(args, result['outputs'])
            # outputs = [result['oracle_output']]
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
            output_records[result['qs_id']] = outputs
            retrieval_records[result['qs_id']] = result['ret_docs']
        scores, eval_records = loader.eval_passk(_gene_results, top_k=k_list)

    elif args.dataset == 'DS1000':
        # gene_results = json.load(open(DS1000Loader().oracle_doc_file, 'r'))
        loader = DS1000Loader()
        qs_list = loader.load_qs_list()
        _gene_results = list()
        for idx, result in enumerate(gene_results):
            assert qs_list[idx]['qs_id'] == result['qs_id']
            if args.prompt_type == 'self-refine': outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1], outputs_before=gene_results_before[idx]['outputs'])
            elif args.prompt_type == 'self-consistency': outputs = [process_outputs_for_self_consistency(process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1]))]
            else: outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1])
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
            output_records[result['qs_id']] = outputs
            retrieval_records[result['qs_id']] = result['ret_docs']
        scores, eval_records = loader.eval_passk(_gene_results, k_list=k_list)
        # scores, eval_records = dict(), dict()

    elif args.dataset == 'pandas_numpy_eval':
        loader = PandasNumpyEvalLoader()
        qs_list = loader.load_qs_list()
        _gene_results = []
        for idx, result in enumerate(gene_results):
            assert qs_list[idx]['qs_id'] == result['qs_id']
            if args.prompt_type == 'self-refine': outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'], outputs_before=gene_results_before[idx]['outputs'])
            elif args.prompt_type == 'self-consistency': outputs = [process_outputs_for_self_consistency(process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question']))]
            else: outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'])
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
            output_records[result['qs_id']] = outputs
            retrieval_records[result['qs_id']] = result['ret_docs']
        scores, eval_records = loader.eval_passk(_gene_results, k_list=k_list)
        # scores, eval_records = dict(), dict()

    elif args.dataset == 'NQ' or args.dataset == 'TriviaQA':
        loader = NQTriviaQAUtils(args.dataset)
        oracle_list = loader.load_oracle_list()
        preds, answers_list = [], []
        for idx, (result, oracle) in enumerate(zip(gene_results, oracle_list)):
            assert str(result['qs_id']) == str(oracle['qs_id'])
            if args.prompt_type == 'self-refine': pred = process_gene_results(args, result['outputs'], outputs_before=gene_results_before[idx]['outputs'])[0]
            elif args.prompt_type == 'self-consistency': pred = process_outputs_for_self_consistency(process_gene_results(args, result['outputs']))
            else: pred = process_gene_results(args, result['outputs'])[0]  # Todo: now only 1 inference
            preds.append(pred)
            answers_list.append(oracle['answers'])
            output_records[result['qs_id']] = [pred]
            retrieval_records[result['qs_id']] = result['ret_docs']
        scores, _eval_records = loader.pred_eval(preds=preds, answers_list=answers_list)
        eval_records = dict()
        for idx, oracle in enumerate(oracle_list):
            eval_records[oracle['qs_id']] = _eval_records[idx]

    elif args.dataset == 'hotpotQA':
        loader = HotpotQAUtils()
        oracle_list = loader.load_oracle_list()
        pred_list = []
        for idx, result in enumerate(gene_results):
            if args.prompt_type == 'self-refine': output = process_gene_results(args, result['outputs'], outputs_before=gene_results_before[idx]['outputs'])[0]
            elif args.prompt_type == 'self-consistency': output = process_outputs_for_self_consistency(process_gene_results(args, result['outputs']))
            else: output = process_gene_results(args, result['outputs'])[0]   # Todo: now only 1 inference
            pred_list.append(dict(qs_id=result['qs_id'], output=output))
            output_records[result['qs_id']] = [output]
            retrieval_records[result['qs_id']] = result['ret_docs']
        scores, eval_records = loader.eval_pred(pred_list=pred_list, oracle_list=oracle_list)

    else:
        raise ValueError('Not supported dataset {}'.format(args.dataset))

    ret_doc_keys_list, prompts, pl_list = [], [], []
    with open(args.prompt_save_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if len(data['ret_doc_keys']) != 0: ret_doc_keys_list.append(data['ret_doc_keys'])
            prompts.append(data['prompt'])
            pl_list.append(data['prompt_length'])

    ret_doc_key_flags_list = None
    if len(ret_doc_keys_list) != 0 and if_eval_retrieval is True:
        oracle_list = loader.load_oracle_list()
        ret_doc_key_flags_list, avg_ret_recall, avg_oracle_percent, avg_oracle_rank = ret_eval_by_doc_keys(dataset=args.dataset, oracle_list=oracle_list, ret_doc_keys_list=ret_doc_keys_list)
        # print('ret recall: ', avg_ret_recall)
        # print('avg oracle doc percentage: ', avg_oracle_percent)
        # print('avg oracle doc rank: ', avg_oracle_rank + 1)  # rank start from 1
        # print('avg prompt length: ', sum(pl_list) / len(pl_list))
        scores['ret_recall'] = avg_ret_recall
        scores['oracle_percent'] = avg_oracle_percent
        scores['oracle_rank'] = avg_oracle_rank
        print('ishjdsfeuhisjocjseghijso')

    # avg prompt length
    scores['prompt_length'] = sum(pl_list) / len(pl_list)


    # calc perplexity
    if if_calc_perplexity is True:
        perplexity = 0
        perplexity_list = []
        batch_idx = 0; valid_outputs = []; logprobs_list = []
        for result_idx, result in enumerate(gene_results):
            logprobs = result['logprobs'][0]  # todo: only for n=1
            # llama would output extra content, remove them when calculating perplexity
            if 'llama' in args.model and args.analysis_type == 'prompt_method':
                batch_idx += 1
                valid_outputs.append(result['outputs'][0].split('Potential documents')[0].replace('\n\n\n', ''))
                logprobs_list.append(logprobs[0])   # for llama
                if batch_idx == 100 or result_idx == len(gene_results) - 1:
                    valid_outputs_length = get_docs_tokens(valid_outputs, args.model)
                    logprobs_list = [logprobs[:length] for logprobs, length in zip(logprobs_list, valid_outputs_length)]
                    for logprobs in logprobs_list:
                        perplexity += np.exp(-sum(logprobs) / len(logprobs))
                        perplexity_list.append(np.exp(-sum(logprobs) / len(logprobs)))
                    batch_idx = 0; valid_outputs = []; logprobs_list = []
            else:
                if 'llama' in args.model:
                    logprobs = logprobs[0]  # for llama
                try:
                    perplexity += np.exp(-sum(logprobs) / len(logprobs))
                    perplexity_list.append(np.exp(-sum(logprobs) / len(logprobs)))
                except: print(logprobs)
        # scores['perplexity'] = perplexity / len(gene_results)
        # oracle_list = loader.load_oracle_list()
        # for idx, oracle in enumerate(oracle_list):
        #     eval_records[oracle['qs_id']]['perplexity'] = perplexity_list[idx]
        print(perplexity_list)


    # extra analyze for code
    if if_code_analysis is True and args.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
        eval_datas = dict(eval_records=eval_records, output_records=output_records, retrieval_records=retrieval_records, ret_eval_records=ret_doc_key_flags_list if len(ret_doc_keys_list) != 0 else [])
        retrieval_consistency, syntax_error, semantic_error = analyze_results_for_code(args.dataset, eval_datas)
        scores['retrieval_consistency'] = retrieval_consistency
        scores['syntax_error_percent'] = syntax_error
        scores['semantic_error_percent'] = semantic_error

    scores = {key: round(value, 3) for key, value in scores.items() if value is not None}
    print(scores)
    if if_save:
        with open(eval_save_file, 'w') as f:
            json.dump(dict(scores=scores, eval_records=eval_records, output_records=output_records, retrieval_records=retrieval_records, ret_eval_records=ret_doc_key_flags_list), f, indent=2)

    return scores



if __name__ == '__main__':
    in_program_call = None
    ret_accs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for ret_acc in ret_accs:
        in_program_call = f'--model codellama-13b-instruct --dataset DS1000 --retriever openai-embedding --analysis_type prompt_method --prompt_type 3shot --n 1'
        # in_program_call = '--model codellama-13b-instruct --dataset conala --retriever openai-embedding --n 1 --analysis_type retrieval_doc_selection --doc_selection_type top_5'
        # args = generate_config(in_program_call)

        # scores = pred_eval(args, if_eval_retrieval=False, if_code_analysis=True, if_calc_perplexity=False, if_save=True)

    # if args.dataset == 'DS1000':
    #     """
    #     test process outputs for DS1000
    #     """
    #     cannot_answer_count = 0
    #     ds1000 = DS1000Dataset(source_dir='../data/DS1000/ds1000_data', libs='all', mode='Insertion')
    #     gene_results = json.load(open(args.result_save_file, 'r'))
    #     if args.prompt_type == 'self-refine': gene_results_before = json.load(open(args.result_save_file.replace('self-refine', '3shot'), 'r'))
    #     loader = DS1000Loader()
    #     oracle_list = loader.load_oracle_list()
    #     qs_list = loader.load_qs_list()
    #     for idx, result in enumerate(gene_results):
    #         qs_id = result['qs_id']
    #         [lib, problema_id] = qs_id.split('_')
    #         data = ds1000[lib][int(problema_id)]
    #         print(f'\n<processed code {idx}>]')
    #         print([result['outputs'][0]])
    #         if '\n\n\n\n\n\n\n\n' in result['outputs'][0]: cannot_answer_count += 1
    #         # print([result['outputs'][0]])
    #         # print(qs_list[idx]['question'].split('\nA:')[1])
    #         if args.prompt_type == 'self-refine':
    #             outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1], outputs_before=gene_results_before[idx]['outputs'])
    #         else:
    #             outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1])
    #         if args.prompt_type == 'self-consistency':
    #             print(outputs)
    #             most_output = process_outputs_for_self_consistency(outputs)
    #             print(most_output)
    #         else:
    #             print([outputs[0]])
    #     print(cannot_answer_count)
    #
    # elif args.dataset == 'pandas_numpy_eval':
    #     """
    #     test process outputs for pandas_numpy_eval
    #     """
    #     dataset = json.load(open('../data/pandas_numpy_eval/data/pandas_numpy_eval.json', 'r'))
    #     gene_results = json.load(open(args.result_save_file, 'r'))
    #     if args.prompt_type == 'self-refine': gene_results_before = json.load(open(args.result_save_file.replace('self-refine', '3shot'), 'r'))
    #     for idx, result in enumerate(gene_results):
    #         print(f'\n<processed code {idx}>]')
    #         print([result['outputs'][0]])
    #         # print([result['outputs'][0]])
    #         for data in dataset:
    #             if data['task_id'] == result['qs_id']:
    #                 code_prompt = data['prompt']
    #         if args.prompt_type == 'self-refine':
    #             outputs = process_gene_results(args, result['outputs'], code_prompt, gene_results_before[idx]['outputs'])
    #         else:
    #             outputs = process_gene_results(args, result['outputs'], code_prompt)
    #         if args.prompt_type == 'self-consistency':
    #             print(outputs)
    #             most_output = process_outputs_for_self_consistency(outputs)
    #             print(most_output)
    #         else:
    #             print([outputs[0]])
    #
    # elif args.dataset == 'conala':
    #     """
    #     test for conala
    #     """
    #     gene_results = json.load(open(args.result_save_file, 'r'))
    #     if args.prompt_type == 'self-refine': gene_results_before = json.load(open(args.result_save_file.replace('self-refine', '3shot'), 'r'))
    #     for idx, result in enumerate(gene_results):
    #         print(f'\n<processed code {idx}>]')
    #         print([result['outputs'][0]])
    #         if args.prompt_type == 'self-refine':
    #             outputs = process_gene_results(args, result['outputs'], outputs_before=gene_results_before[idx]['outputs'])
    #         else:
    #             outputs = process_gene_results(args, result['outputs'])
    #         if args.prompt_type == 'self-consistency':
    #             print(outputs)
    #             most_output = process_outputs_for_self_consistency(outputs)
    #             print(most_output)
    #         else:
    #             print([outputs[0]])
    #
    # else:
    #     """
    #     test for NQ, TriviaQA, hotpotQA
    #     """
    #     cannot_answer_count = 0
    #     gene_results = json.load(open(args.result_save_file, 'r'))
    #     if args.prompt_type == 'self-refine': gene_results_before = json.load(open(args.result_save_file.replace('self-refine', '3shot'), 'r'))
    #     for idx, result in enumerate(gene_results):
    #         print(f'\n<processed answer {idx}>]')
    #         print([result['outputs']])
    #         if "I'm sorry" in result['outputs'][0]: cannot_answer_count += 1
    #         if args.prompt_type == 'self-refine':
    #             outputs = process_gene_results(args, result['outputs'], outputs_before=gene_results_before[idx]['outputs'])
    #         else:
    #             outputs = process_gene_results(args, result['outputs'])
    #         if args.prompt_type == 'self-consistency':
    #             print(outputs)
    #             most_output = process_outputs_for_self_consistency(outputs)
    #             print(most_output)
    #         else:
    #             print([outputs[0]])
    #     print(cannot_answer_count)
    #
    # # todo: for self-consistency
