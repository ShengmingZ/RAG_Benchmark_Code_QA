import argparse
import subprocess


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['llama3-8b', 'llama2-13b-chat', 'codellama-13b-instruct', 'gpt-3.5-turbo-0125', 'gpt-4o'])
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--dataset', type=str, choices=['conala', 'DS1000', 'pandas_numpy_eval', 'NQ', 'TriviaQA', 'hotpotQA'])
    parser.add_argument('--retriever', type=str, choices=['best', 'BM25', 'contriever', 'miniLM', 'openai-embedding'])
    parser.add_argument('--analysis_type', type=str, choices=['retrieval_recall', 'retrieval_doc_type', 'retrieval_doc_selection_topk',
                                                              'retrieval_doc_selection_pl', 'prompt_length'])
    parser.add_argument('--action', type=str, choices=['gene_prompts', 'gene_responses', 'eval_pred'])
    parser.add_argument('--n', type=int)
    parser.add_argument('--batch', action='store_true')
    args = parser.parse_args()
    return args


args = config()

# ret acc
if args.analysis_type == "retrieval_recall":
    ret_acc_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    cmds = []
    if args.action == 'gene_prompts' or (args.action == 'gene_responses' and args.batch is True and 'gpt' in args.model):  # run in the same time
        for ret_acc in ret_acc_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} --batch '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --ret_acc {ret_acc}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' & '
        subprocess.check_output(batch_cmd, shell=True)
    else:
        for ret_acc in ret_acc_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --ret_acc {ret_acc}')
            subprocess.check_output(cmd, shell=True)
            print(f'done ret_acc {ret_acc}')

elif args.analysis_type == "retrieval_doc_type":
    ret_doc_type_list = ['oracle', 'retrieved', 'distracting', 'random', 'irrelevant_dummy', 'irrelevant_diff', 'none']
    cmds = []
    if args.action == 'gene_prompts' or (args.action == 'gene_responses' and args.batch is True and 'gpt' in args.model):  # run in the same time
        for ret_doc_type in ret_doc_type_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} --batch '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --ret_doc_type {ret_doc_type}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' & '
        subprocess.check_output(batch_cmd, shell=True)
    else:       # run in a sequence
        for ret_doc_type in ret_doc_type_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --ret_doc_type {ret_doc_type}')
            subprocess.check_output(cmd, shell=True)
            print(f'done ret_doc_type {ret_doc_type}')

elif args.analysis_type.startswith("retrieval_doc_selection"):
    if args.analysis_type.rsplit('_',1)[-1] == 'topk':
        if args.dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
            if 'gpt' in args.model:
                doc_selection_type_list = ['top_1', 'top_20', 'top_40', 'top_60', 'top_80']
            else:
                doc_selection_type_list = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']
        else:
            doc_selection_type_list = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']
    elif args.analysis_type.rsplit('_',1)[-1] == 'pl':
        if args.dataset in ['NQ', 'TriviaQA', 'hotpotQA'] and 'llama' in args.model:
            doc_selection_type_list = ['pl_500', 'pl_1000', 'pl_1500', 'pl_2000', 'pl_2500']
        else:
            doc_selection_type_list = ['pl_500', 'pl_2000', 'pl_4000', 'pl_6000', 'pl_8000']
    else:
        raise ValueError('invalid analysis type {}'.format(args.analysis_type))
    cmds = []
    args.analysis_type = args.analysis_type.rsplit('_', 1)[0]
    if args.action == 'gene_prompts' or (args.action == 'gene_responses' and args.batch is True and 'gpt' in args.model):
        for doc_selection_type in doc_selection_type_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} --batch '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --doc_selection_type {doc_selection_type}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' & '
        subprocess.check_output(batch_cmd, shell=True)
    else:
        for doc_selection_type in doc_selection_type_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --doc_selection_type {doc_selection_type}')
            subprocess.check_output(cmd, shell=True)
            print(f'done doc_selection_type {doc_selection_type}')

elif args.analysis_type == 'prompt_length':
    pl_analysis_list = ['oracle_top10', 'distracting_top10', 'random_top10', 'irrelevant_diff_top10', 'irrelevant_dummy_top10']
    cmds = []
    if args.action == 'gene_prompts' or (args.action == 'gene_responses' and args.batch is True and 'gpt' in args.model):  # use batch, run simo
        for pl_analysis in pl_analysis_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} --batch '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --pl_analysis {pl_analysis}')
            cmds.append(cmd)
        batch_cmd = ''
        for cmd in cmds:
            batch_cmd = batch_cmd + cmd + ' & '
        subprocess.check_output(batch_cmd, shell=True)
    else:
        for pl_analysis in pl_analysis_list:
            cmd = (f'python generator/generate.py --action {args.action} --model {args.model} --temperature {args.temperature} '
                   f'--dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --pl_analysis {pl_analysis}')
            subprocess.check_output(cmd, shell=True)
            print(f'done prompt length analysis {pl_analysis}')
