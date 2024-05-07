import json
import shlex
import argparse
import platform
import sys, os
import random
from tqdm import tqdm
from prompt.hotpotqa_prompt import original_prompt
from copy import deepcopy
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from prompt import conala_prompt, tldr_prompt, hotpotqa_prompt
from dataset_utils.dataset_configs import HotpotQALoader, WikiCorpusLoader, PythonDocsLoader
from retriever.sparse_retriever import sparse_retriever_config
from retriever.dense_retriever import dense_retriever_config


random.seed(0)


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


def get_irrelevant_doc(irrelevant_type, doc_length, model_type, n):
    assert irrelevant_type in ['dummy', 'diff']
    assert model_type in ['gpt', 'llama']
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
    parser.add_argument('--dataset', type=str, choices=['tldr', 'conala', 'DS1000', 'pandas-numpy-eval', 'hotpotQA'])
    parser.add_argument('--save_file', type=str, default=None)
    # model parameters
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=1000)
    # analysis type
    parser.add_argument('--analysis_type', type=str, choices=['retrieval_quality', 'prompt_generation'])
    # retrieval quality analysis, default: retriever with best performance
    parser.add_argument('--retriever', type=str, default='best', choices=['best', 'bm25', 'contriever', 'miniLM', 'openai-embedding'])
    parser.add_argument('--ret_acc', type=float, default=1)
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
                          f'model_{args.model}_temperature_{args.temperature}_n_{args.n}_'
                          f'analysis_type_{args.analysis_type}_')
        if args.analysis_type == 'retrieval_quality':
            args.save_file += (f'retriever_{args.retriever}_'
                               f'retrieval_acc_{args.ret_acc}_'
                               f'ret_info_type_{args.ret_info_type}.json')
        elif args.analysis_type == 'prompt_generation':
            args.save_file += (f'top_k_{args.top_k}_'
                               f'doc_max_length_{args.doc_max_length}_'
                               f'prompt_type_{args.prompt_type}.json')
        args.save_file = os.path.join(root_path, args.save_file)

    print(json.dumps(vars(args), indent=2))
    return args


def control_ret_acc(ret_acc, oracle_list, dataset_type):
    """
    generate retrieval doc key of each sample based on ret_acc,
    perturb the doc key until it reaches the new ret_acc value
    :param ret_acc:
    :param oracle_list: a list of list that store oracle doc key for each sample
    :param dataset_type:
    :return:
    """
    ret_acc_per_sample = [1] * len(oracle_list)
    cur_ret_acc = sum(ret_acc_per_sample) / len(ret_acc_per_sample)
    assert dataset_type in ['nlp', 'se']
    if dataset_type == 'nlp':
        corpus_doc_keys = WikiCorpusLoader().load_wiki_id()
    perturb_oracle_list = deepcopy(oracle_list)
    while cur_ret_acc > ret_acc:
        random_idx = random.randint(0, len(perturb_oracle_list) - 1)
        oracle_doc_length = len(perturb_oracle_list[random_idx])
        random_idx_inner = random.randint(0, oracle_doc_length-1)
        if perturb_oracle_list[random_idx][random_idx_inner] == oracle_list[random_idx][random_idx_inner]:  # replace oracle doc with random key
            perturb_oracle_list[random_idx][random_idx_inner] = corpus_doc_keys[random.randint(0, len(corpus_doc_keys) - 1)]
            ret_acc_per_sample[random_idx] = (ret_acc_per_sample[random_idx] * oracle_doc_length - 1) / oracle_doc_length
            cur_ret_acc = sum(ret_acc_per_sample) / len(ret_acc_per_sample)

    docs = []
    if dataset_type == 'nlp':
        wiki_loader = WikiCorpusLoader()
        for doc_key in tqdm(perturb_oracle_list):
            docs.append(wiki_loader.get_docs(doc_key))
    elif dataset_type == 'se':
        ...

    return perturb_oracle_list, docs


def perturb_ret_doc_type(perturb_doc_type, ret_doc_key_list, oracle_doc_key_list, dataset_type, model_type):
    """
    generate retrieval doc key of each sample based on ret_doc_type, return a list of the docs for each sample
    :param ret_doc_type:
    :return:
    """
    assert perturb_doc_type in ['oracle', 'retrieved', 'distracting', 'random', 'irrelevant_dummy', 'irrelevant_diff', 'none']
    assert dataset_type in ['nlp', 'se']

    if perturb_doc_type in ['irrelevant_diff', 'irrelevant_dummy']:
        if dataset_type == 'nlp':
            doc_length = 100
        elif dataset_type == 'se':
            doc_length = 1000
        docs = []
        for oracle_doc_key in oracle_doc_key_list:
            irrelevant_docs = get_irrelevant_doc(irrelevant_type=perturb_doc_type.split('_')[1], doc_length=doc_length, model_type=model_type, n=len(oracle_doc_key))
            docs.append(irrelevant_docs)
        return []*len(docs), docs
    else:
        if perturb_doc_type == 'oracle':
            doc_key_list = oracle_doc_key_list
        elif perturb_doc_type == 'retrieved':
            doc_key_list = []
            for ret_doc_key, oracle_doc_key in zip(ret_doc_key_list, oracle_doc_key_list):
                doc_key_list.append(ret_doc_key[:len(oracle_doc_key)])
        elif perturb_doc_type == 'distracting':
            doc_key_list = []
            for ret_doc_key, oracle_doc_key in zip(ret_doc_key_list, oracle_doc_key_list):
                sample_doc_key = []
                for key in ret_doc_key:
                    if key not in oracle_doc_key:
                        sample_doc_key.append(key)
                    if len(sample_doc_key) == len(oracle_doc_key):
                        break
                doc_key_list.append(sample_doc_key)
        elif perturb_doc_type == 'random':
            doc_key_list = []
            if dataset_type == 'nlp':
                wiki_loader = WikiCorpusLoader()
                corpus_id_list = wiki_loader.load_wiki_id()
            elif dataset_type == 'se':
                ...
            for oracle_doc_key in oracle_doc_key_list:
                doc_key_list.append(random.sample(corpus_id_list, k=len(oracle_doc_key)))
        elif perturb_doc_type == 'none':
            doc_key_list = []*len(oracle_doc_key_list)

        docs = []
        if dataset_type == 'nlp':
            wiki_loader = WikiCorpusLoader()
            for doc_key in doc_key_list:
                docs.append(wiki_loader.get_docs(doc_key))
        elif dataset_type == 'se':
            ...

        return doc_key_list, docs




def process_retrieval_doc():
    ...


def apply_prompt_method(questions, ret_docs, prompt_type, dataset):
    assert dataset in ['hotpotQA', 'conala', 'DS1000', 'pandas-numpy-eval']
    prompts = []
    if dataset == 'hotpotQA':
        if prompt_type == 'original':
            for question, ret_doc in zip(questions, ret_docs):
                assert len(ret_doc) == 2
                prompt = hotpotqa_prompt.original_prompt.replace('<QUESTION>', question).replace('<POTENTIAL DOCUMENTS 1>', f'1: {ret_doc[0]}').replace('<POTENTIAL DOCUMENTS 2>', f'2: {ret_doc[1]}')
                prompts.append(prompt)
    return prompts





if __name__ == "__main__":
    # test for control_ret_acc
    hotpotqa_loader = HotpotQALoader()
    oracle_list = hotpotqa_loader.load_oracle_list()
    oracle_list = [oracle['oracle_docs'] for oracle in oracle_list]
    wiki_loader = WikiCorpusLoader()
    wiki_id_list = wiki_loader.load_wiki_id()
    perturb_oracle_list = control_ret_acc(0.8, oracle_list, wiki_id_list)
    ret_acc = 0
    for perturb_oracle, oracle in zip(perturb_oracle_list, oracle_list):
        count = sum(1 for x, y in zip(perturb_oracle, oracle) if x == y)
        ret_acc = ret_acc + count/len(perturb_oracle)
    ret_acc = ret_acc/len(oracle_list)
    print(ret_acc)

