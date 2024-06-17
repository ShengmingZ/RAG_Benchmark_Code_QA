import argparse
import subprocess


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', choices=['llama3-8b', 'llama2-13b-chat', 'codellama-13b-instruct', 'gpt-3.5-turbo-0125', 'gpt-4o'])
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--dataset', type=str, choices=['conala', 'DS1000', 'pandas_numpy_eval', 'NQ', 'TriviaQA'])
    parser.add_argument('--retriever', type=str, default='best', choices=['best', 'BM25', 'contriever', 'miniLM', 'openai-embedding'])
    parser.add_argument('--analysis_type', type=str, choices=['retrieval_recall', 'retrieval_doc_type', 'retrieval_doc_num'])
    parser.add_argument('--n', type=int, default=1)
    args = parser.parse_args()
    return args


args = config()

# ret acc
if args.analysis_type == "retrieval_recall":
    ret_acc_list = [1, 0.8, 0.6, 0.4, 0.2, 0]
    # ret_acc_list = [1]
    for ret_acc in ret_acc_list:
        cmd = f'python generator/generate.py --model {args.model} --temperature {args.temperature} --dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --ret_acc {ret_acc}'
        subprocess.check_output(cmd, shell=True)
        print(f'done ret_acc {ret_acc}')

elif args.analysis_type == "retrieval_doc_type":
    ret_doc_type_list = ['oracle', 'retrieved', 'distracting', 'random', 'irrelevant_dummy', 'irrelevant_diff', 'none']
    for ret_doc_type in ret_doc_type_list:
        cmd = f'python generator/generate.py --model {args.model} --temperature {args.temperature} --dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --ret_doc_type {ret_doc_type}'
        subprocess.check_output(cmd, shell=True)
        print(f'done ret_doc_type {ret_doc_type}')

elif args.analysis_type == "rertieval_doc_num":
    top_k_list = [1, 3, 5, 10, 20]    # top_k vs truncate
    for top_k in top_k_list:
        cmd = f'python generator/generate.py --model {args.model} --temperature {args.temperature} --dataset {args.dataset} --retriever {args.retriever} --analysis_type {args.analysis_type} --n {args.n} --top_k {top_k}'
        subprocess.check_output(cmd, shell=True)
        print(f'done top_k {top_k}')