import json
import shlex
import argparse
import platform
import sys, os
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from prompt import conala_prompt, tldr_prompt
from retriever.sparse_retriever import sparse_retriever_config
from retriever.dense_retriever import dense_retriever_config


def save_results_to_files(save_file, gene_results):
    if os.path.exists(save_file):
        user_input = input(f'The file {save_file} already exists. Overwrite? (y/n): ').lower()
        if user_input == 'y':
            with open(save_file, 'w+') as f:
                json.dump(gene_results, f, indent=2)
            print('overwrite file done')
        else:
            print('save file not overwrite')
    else:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, 'w+') as f:
            json.dump(gene_results, f, indent=2)


def approximate_token(prompts, model='gpt-3.5-turbo'):
    import tiktoken
    max_tokens, avg_tokens = 0, 0
    encoding = tiktoken.encoding_for_model(model)
    for prompt in prompts:
        tokens = len(encoding.encode(prompt))
        avg_tokens += tokens
        if tokens > max_tokens: max_tokens = tokens
    avg_tokens = avg_tokens / len(prompts)
    print(f"Average tokens: {avg_tokens:.3f}, Max tokens: {max_tokens}")


def truncate_doc(doc, model='gpt-3.5-turbo', max_length=1000):
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    encoded_doc = encoding.encode(doc)
    if len(encoded_doc) > max_length:
        encoded_doc = encoded_doc[:max_length]
        doc = encoding.decode(encoded_doc)
    return doc


# todo: could be docs from another corpus like wiki or just random words
# control prompt length
def get_dummy_text(model='gpt-3.5-turbo', prompt_length=1000, dataset='tldr'):
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    if dataset == 'tldr':
        encoded_doc = encoding.encode(tldr_prompt.tldr_original_3shots_prompt)
    elif dataset == 'conala':
        encoded_doc = encoding.encode(conala_prompt.conala_original_3shots_prompt)
    doc_length = int((prompt_length - len(encoded_doc))/10)
    dummy_docs = tldr_prompt.dummy_docs.split('\n')[:10]
    docs = list()
    for dummy_doc in dummy_docs:
        encoded_doc = encoding.encode(dummy_doc)[:doc_length]
        doc = encoding.decode(encoded_doc)
        docs.append(doc)
    return docs


def generate_config(in_program_call=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['tldr', 'conala', 'ds1000', 'pandas_numpy_eval', 'hotpotqa'])
    parser.add_argument('--sampled', action='store_true')
    parser.add_argument('--save_file', type=str, default=None)
    # model parameters
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--max_tokens', type=int, default=1000)
    # analysis type
    parser.add_argument('--analysis_type', type=str, choice=['retrieval_quality', 'prompt_generation'])
    # retrieval quality analysis, default: retriever with best performance
    parser.add_argument('--retriever', type=str, default='best', choices=['best', 'bm25', 'codeT5-FT'])
    parser.add_argument('--retrieval_acc', type=float, default=1)
    parser.add_argument('--ret_info_type', type=str, default='retrieved', choices=['oracle', 'retrieved', 'related', 'random', 'unrelated', 'none'])
    # info processing analysis, default: top_k 5, truncate 4000
    parser.add_argument('--top_k', type=int, default=5)     # k docs
    parser.add_argument('--doc_max_length', type=int, default=4000)
    # prompt method analysis, default: original prompt 3-shots
    parser.add_argument('--prompt_type', type=str, default='original', choices=['original', '0shot', 'instruct', 'CoT'])

    args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))

    # construct save file
    if args.save_file is None:
        args.save_file = (f'data/{args.dataset}/results/'
                          f'sampled_{args.sampled}_'
                          f'model_{args.model}_temperature_{args.temperature}_n_{args.n}_'
                          f'analysis_type_{args.analysis_type}_')
        if args.analysis_type == 'retrieval_quality':
            args.save_file += (f'retriever_{args.retriever}_'
                               f'retrieval_acc_{args.retrieval_acc}_'
                               f'ret_info_type_{args.ret_info_type}.json')
        elif args.analysis_type == 'prompt_generation':
            args.save_file += (f'top_k_{args.top_k}_'
                               f'doc_max_length_{args.doc_max_length}_'
                               f'prompt_type_{args.prompt_type}.json')
        args.save_file = os.path.join(root_path, args.save_file)

    print(json.dumps(vars(args), indent=2))
    return args
