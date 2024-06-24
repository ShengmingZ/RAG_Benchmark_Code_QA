import argparse
import subprocess


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106', choices=['llama3-8b', 'llama2-13b-chat', 'codellama-13b-instruct', 'gpt-3.5-turbo-0125', 'gpt-4o'])
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--dataset', type=str, choices=['conala', 'DS1000', 'pandas_numpy_eval', 'hotpotQA', 'NQ', 'TriviaQA'])
    parser.add_argument('--retriever', type=str, default='best', choices=['best', 'BM25', 'contriever', 'miniLM', 'openai-embedding'])
    parser.add_argument('--analysis_type', type=str, choices=['retrieval_recall', 'retrieval_doc_type'])
    parser.add_argument('--n', type=int, default=1)
    args = parser.parse_args()
    return args


args = config()

# ret acc
if args.analysis_type == "retrieval_recall":
    ret_acc_list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    # ret_acc_list = [1]
    result_list = list()
    for ret_acc in ret_acc_list:
        cmd = f'python generator/pred_eval.py --model {args.model} --temperature {args.temperature} --dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --ret_acc {ret_acc}'
        proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, error) = proc.communicate()
        # if args.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
        passk = output.decode().split('\n')[-2]
        result_list.append(passk)
    for ret_acc, result in zip(ret_acc_list, result_list):
        print(f"{ret_acc}: {result},")

elif args.analysis_type == "retrieval_doc_type":
    ret_doc_type_list = ['oracle', 'retrieved', 'distracting', 'random', 'irrelevant_dummy', 'irrelevant_diff', 'none']
    result_list = list()
    for ret_doc_type in ret_doc_type_list:
        cmd = f'python generator/pred_eval.py --model {args.model} --temperature {args.temperature} --dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --ret_doc_type {ret_doc_type}'
        proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, error) = proc.communicate()
        passk = output.decode().split('\n')[-2]
        result_list.append(passk)
    for ret_doc_type, result in zip(ret_doc_type_list, result_list):
        print(f'"{ret_doc_type}": {result},')