import argparse
import subprocess


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106', choices=['llama3-8b', 'llama2-13b-chat', 'codellama-13b-instruct', 'gpt-3.5-turbo-1106', 'gpt-4o'])
    parser.add_argument('--dataset', type=str, choices=['conala', 'DS1000', 'pandas-numpy-eval', 'hotpotQA'])
    parser.add_argument('--retriever', type=str, default='best', choices=['best', 'BM25', 'contriever', 'miniLM', 'openai-embedding'])
    parser.add_argument('--analysis_type', type=str, choices=['retrieval_recall', 'retrieval_doc_type'])
    args = parser.parse_args()
    return args


args = config()

# ret acc
if args.analysis_type == "retrieval_recall":
    ret_acc_list = [1, 0.8, 0.6, 0.4, 0.2, 0]
    # ret_acc_list = [1]
    result_list = list()
    for ret_acc in ret_acc_list:
        cmd = f'python generator/code_eval.py --model {args.model} --dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --ret_acc {ret_acc}'
        proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, error) = proc.communicate()
        passk = output.decode().split('\n')[-2]
        result_list.append(passk)
    for ret_acc, result in zip(ret_acc_list, result_list):
        print('ret_acc: ', ret_acc, result)