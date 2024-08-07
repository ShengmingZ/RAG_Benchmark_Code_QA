import json
import shlex
import argparse
import platform
import sys, os
import random
import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from functools import partial
from copy import deepcopy
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from prompt import conala_prompt, DS1000_prompt, pandas_numpy_eval_prompt, NQ_TriviaQA_prompt, hotpotQA_prompt
from retriever.retriever_utils import retriever_config, get_ret_results
from dataset_utils.conala_utils import ConalaLoader
from dataset_utils.DS1000_utils import DS1000Loader
from dataset_utils.hotpotQA_utils import HotpotQAUtils
from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
from dataset_utils.corpus_utils import PythonDocsLoader, WikiCorpusLoader

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(0)

AVG_PROMPT_LENGTH_HOTPOT = ...
AVG_PROMPT_LENGTH_NQ = ...
AVG_PROMPT_LENGTH_CONALA = ...
AVG_PROMPT_LENGTH_DS1000 = ...
AVG_PROMPT_LENGTH_PANDASEVAL = ...


def save_results_to_files(save_file, gene_results, overwrite=False):
    if overwrite is False and os.path.exists(save_file):
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


def approximate_token(prompts, model):
    pl_list = []
    if model.startswith('gpt'):
        import tiktoken
        max_tokens, avg_tokens = 0, 0
        encoding = tiktoken.encoding_for_model(model)
        for prompt in prompts:
            total_prompt = prompt[0] + prompt[1]
            tokens = len(encoding.encode(total_prompt))
            pl_list.append(tokens)
            avg_tokens += tokens
            if tokens > max_tokens: max_tokens = tokens
        avg_tokens = avg_tokens / len(prompts)
        print(f"Average tokens: {avg_tokens:.3f}, Max tokens: {max_tokens}")
    elif model.startswith('llama') or model.startswith('codellama'):
        if model == 'llama2-13b-chat':
            model = 'meta-llama/Llama-2-13b-chat-hf'
        elif model == 'codellama-13b-instruct':
            model = 'codellama/CodeLlama-13b-Instruct-hf'
        elif model == 'llama3-8b':
            model = 'meta-llama/Meta-Llama-3-8B'
        access_token = "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"
        tokenizer = AutoTokenizer.from_pretrained(model, torch_dtype=torch.float16, token=access_token)
        avg_tokens = 0
        max_tokens = 0
        for prompt in prompts:
            tokens = len(tokenizer(prompt, return_tensors='pt')['input_ids'][0])
            pl_list.append(tokens)
            if tokens > max_tokens: max_tokens = tokens
            avg_tokens += tokens
        avg_tokens = avg_tokens / len(prompts)
        print(f"Average tokens: {avg_tokens:.3f}, Max tokens: {max_tokens}")
    else:
        raise Exception("Invalid model name")
    return pl_list


def truncate_docs(docs, model, max_length):
    if model.startswith('gpt'):
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        truncated_docs = []
        for doc in docs:
            encoded_doc = encoding.encode(doc)
            if len(encoded_doc) > max_length:
                encoded_doc = encoded_doc[:max_length]
                doc = encoding.decode(encoded_doc)
            truncated_docs.append(doc)
    elif model.startswith('llama') or model.startswith('codellama'):
        if model == 'llama2-13b-chat':
            model = 'meta-llama/Llama-2-13b-chat-hf'
        elif model == 'codellama-13b-instruct':
            model = 'codellama/CodeLlama-13b-Instruct-hf'
        elif model == 'llama3-8b':
            model = 'meta-llama/Meta-Llama-3-8B'
        access_token = "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"
        tokenizer = AutoTokenizer.from_pretrained(model, torch_dtype=torch.float16, token=access_token)
        truncated_docs = []
        for doc in docs:
            try:
                tokens = tokenizer.encode(doc, max_length=max_length, truncation=True, add_special_tokens=False)
            except:
                print([doc])
                raise Exception('find duplicated error')
            doc = tokenizer.decode(tokens)
            truncated_docs.append(doc)
    else:
        raise ValueError(f"Unknown model: {model}")

    return truncated_docs


def get_docs_tokens(docs, model):
    docs_tokens = []
    if model.startswith('gpt'):
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        for doc in docs:
            encoded_doc = encoding.encode(doc)
            docs_tokens.append(len(encoded_doc))
    elif model.startswith('llama') or model.startswith('codellama'):
        if model == 'llama2-13b-chat':
            model = 'meta-llama/Llama-2-13b-chat-hf'
        elif model == 'codellama-13b-instruct':
            model = 'codellama/CodeLlama-13b-Instruct-hf'
        elif model == 'llama3-8b':
            model = 'meta-llama/Meta-Llama-3-8B'
        access_token = "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"
        tokenizer = AutoTokenizer.from_pretrained(model, torch_dtype=torch.float16, token=access_token)
        for doc in docs:
            tokens = tokenizer.encode(doc, add_special_tokens=False)
            docs_tokens.append(len(tokens))
    else:
        raise ValueError(f'Unknown model: {model} in get_docs_tokens')
    # if max_length is not None:
    #     for i in range(len(docs_tokens)):
    #         docs_tokens[i] = docs_tokens[i] if docs_tokens[i] < max_length else max_length

    return docs_tokens


def get_irrelevant_docs(irrelevant_type, oracle_docs, model, dataset):
    """
    get irrelevant docs of 2 types, keep doc length == oracle doc length
    :param irrelevant_type:
    :param oracle_doc_keys:
    :param model:
    :param dataset:
    :return:
    """
    assert irrelevant_type in ['dummy', 'diff', 'ellipsis']
    doc_lengths = get_docs_tokens(oracle_docs, model)   # get doc lengths

    if irrelevant_type == 'dummy' or irrelevant_type == 'ellipsis':
        if irrelevant_type == 'dummy':
            dummy_string = 'The wiggly fluff went plop while the jibber-jabber bumbled and tumbled. Fizzle-flop danced around the wibbly-wobbly doodle, and snicker-snack bounced happily. Doodle-doo twirled and swirled in the zigzag zoom, and snuggle-bug snuggled close. Wobble-wobble wandered through the dilly-dally, giggling and jiggling all the while. Squiggle-squabble and waddle-waddle wobbled along, playing in the silly-sally world of random wozzle. The snickety-snack skipped and hopped, while the flibber-jabber giggled and squiggled. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Bloop bloop bloop, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Nulla facilisi. Snick snick snick, jibble jibble jibble.'
        else:
            dummy_string = '................................................................'*100 + '\n'
        dummy_length = get_docs_tokens(docs=[dummy_string], model=model)[0]
        perturbed_docs = []
        for doc_length in doc_lengths:
            perturbed_doc = dummy_string * (int(doc_length/dummy_length)+1) * 2
            perturbed_doc = truncate_docs(docs=[perturbed_doc], model=model, max_length=doc_length)[0]
            perturbed_docs.append(perturbed_doc)
    else:
        if dataset in ['conala', 'DS1000', 'pandas_numpy_eval']: loader = WikiCorpusLoader(prepare_random_docs=True)
        else: loader = PythonDocsLoader(prepare_random_docs=True)
        perturbed_docs = []
        for doc_length in doc_lengths:
            perturbed_doc = ''
            perturbed_doc_length = 0
            while perturbed_doc_length < doc_length:
                random_doc = ' '.join(loader.get_random_docs(10))
                random_doc_length = get_docs_tokens(docs=[random_doc], model=model)[0]
                perturbed_doc += random_doc
                perturbed_doc_length += random_doc_length
            perturbed_doc = truncate_docs(docs=[perturbed_doc], model=model, max_length=doc_length)[0]
            perturbed_docs.append(perturbed_doc)

    return perturbed_docs


def generate_config(in_program_call=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['tldr', 'conala', 'DS1000', 'pandas_numpy_eval', 'hotpotQA', 'NQ', 'TriviaQA'])
    parser.add_argument('--result_save_file', type=str, default=None)
    parser.add_argument('--prompt_save_file', type=str, default=None)
    parser.add_argument('--action', type=str, choices=['gene_prompts', 'gene_responses', 'eval_pred'])

    # model parameters
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106', choices=['llama3-8b', 'llama2-13b-chat', 'codellama-13b-instruct', 'gpt-3.5-turbo-0125', 'gpt-4o'])
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--n', type=int)
    parser.add_argument('--max_tokens', type=int, default=500)
    parser.add_argument('--batch', action='store_true')

    parser.add_argument('--retriever', type=str, default='best', choices=['best', 'BM25', 'contriever', 'miniLM', 'openai-embedding', 'codeT5'])

    parser.add_argument('--analysis_type', type=str, choices=['retrieval_recall', 'retrieval_doc_type', 'retrieval_doc_selection', 'prompt_length', 'prompt_method'])
    # each of the following parameter corresponds to one analysis, when choose one, the default value of the other parameters are the default value of RAG
    parser.add_argument('--ret_acc', type=float, default=1)     # top_k:len(oracle_docs), prompt_type:3shots, ret_doc_type:oracle/distracting
    parser.add_argument('--ret_doc_type', type=str, default='retrieved', choices=['oracle', 'retrieved', 'distracting', 'random', 'irrelevant_dummy', 'irrelevant_diff', 'none', 'ellipsis', 'retrieved_top'])
    parser.add_argument('--doc_selection_type', type=str, default=None)
    parser.add_argument('--doc_max_length', type=int, default=1000)
    parser.add_argument('--prompt_type', type=str, default='0shot', choices=['3shot', '0shot', 'cot', 'least_to_most', 'plan_and_solve', 'self-consistency', 'ir-cot', 'flare'])
    parser.add_argument('--pl_analysis', type=str, default=None)

    args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))


    # if args.n == 1: args.temperature = 0
    # else:
    #     if 'llama' in args.model: args.temperature = 0.6
    #     else: args.temperature = 0.7
    # construct save file
    if args.result_save_file is None:
        if args.retriever != 'openai-embedding': args.result_save_file = f'data/{args.dataset}/results/model_{args.model}_retriever_{args.retriever}.json'
        else:
            args.result_save_file = f'data/{args.dataset}/results/model_{args.model}_n_{args.n}_{args.analysis_type}_'
            if args.analysis_type == 'retrieval_recall':
                args.result_save_file += f'{args.ret_acc}.json'
            elif args.analysis_type == 'retrieval_doc_type':
                args.result_save_file += f'{args.ret_doc_type}.json'
            elif args.analysis_type == 'retrieval_doc_selection':
                args.result_save_file += f'{args.doc_selection_type}.json'
            elif args.analysis_type == 'prompt_length':
                args.result_save_file += f'{args.pl_analysis}.json'
            elif args.analysis_type == 'prompt_method':
                args.result_save_file += f'{args.prompt_type}.json'
            else:
                raise ValueError(f'Unknown analysis type: {args.analysis_type}')
        args.result_save_file = os.path.join(root_path, args.result_save_file)
        args.prompt_save_file = args.result_save_file.replace('.json', '_prompts.jsonl')
        if args.n != 1: args.prompt_save_file = args.prompt_save_file.replace(f'n_{args.n}', 'n_1') # prompt does not change when run multi times

    # print(json.dumps(vars(args), indent=2))
    return args


def get_distracting_docs(ret_result, dataset, k, oracle_docs, dups=None):
    """
    get high similarity but not oracle docs for a sample
    :param oracle_docs: oracle docs of the sample
    :param ret_result: ret result of the sample
    :param dataset:
    :param k: number of distracting docs
    :param dups: doc keys that also need to avoid
    :return:
    """
    distracting_docs = []
    # ret_doc_keys = [item['doc_key'] for item in ret_result]
    for item in ret_result:
        doc_key = item['doc_key']
        if len(distracting_docs) == k: break
        if dups is not None and doc_key in dups: continue
        if dataset == 'NQ' or dataset == 'TriviaQA':
            # doc = WikiCorpusLoader().get_docs(doc_keys_list=[[doc_key]], dataset=dataset, num_procs=1)[0][0]
            # if not NQTriviaQAUtils(dataset).if_has_answer(doc=doc, qs_id=qs_id):
            #     distracting_docs.append(doc_key)
            if not item['has_answer']:
                distracting_docs.append(doc_key)
        else:
            if doc_key not in oracle_docs:
                distracting_docs.append(doc_key)
    return distracting_docs


def control_ret_acc(ret_acc, oracle_list, ret_results, dataset):
    """
    generate retrieval doc key of each sample based on ret_acc, perturb the doc key until it reaches the new ret_acc value
    :param ret_acc:
    :param oracle_list: a list of list that store oracle doc key for each sample
    :param dataset:
    :return:
    """
    # perturb oracle_docs_list with high score related docs until it reaches the ret_acc
    if dataset in ['conala', 'DS1000', 'pandas_numpy_eval', 'hotpotQA']:
        oracle_docs_list = deepcopy([oracle['oracle_docs'] for oracle in oracle_list])
    else:
        oracle_docs_list = deepcopy([[oracle['oracle_doc']] for oracle in oracle_list])

    ret_accs = [1] * len(oracle_docs_list)  # record acc of each sample
    cur_ret_acc = sum(ret_accs) / len(ret_accs) # total acc
    perturb_placeholder = list()    # this placeholder is to store doc keys that are oracle
    for i, oracle in enumerate(oracle_docs_list):
        for j in range(len(oracle)):
            perturb_placeholder.append([i, j])

    while cur_ret_acc > ret_acc:
        perturb_idx = random.sample(perturb_placeholder, 1)[0] # pick an oracle key and perturb
        perturb_placeholder.remove(perturb_idx)
        qs_id = oracle_list[perturb_idx[0]]['qs_id']
        oracle_docs = oracle_list[perturb_idx[0]]['oracle_docs'] if dataset in ['conala', 'DS1000', 'pandas_numpy_eval', 'hotpotQA'] else [oracle_list[perturb_idx[0]]['oracle_doc']]
        oracle_docs_list[perturb_idx[0]][perturb_idx[1]] = get_distracting_docs(ret_result=ret_results[qs_id],
                                                                                oracle_docs=oracle_docs,
                                                                                dataset=dataset,
                                                                                k=1,
                                                                                dups=oracle_docs_list[perturb_idx[0]])[0]
        ret_accs[perturb_idx[0]] = (ret_accs[perturb_idx[0]] * len(oracle_docs_list[perturb_idx[0]]) - 1) / len(oracle_docs_list[perturb_idx[0]])
        cur_ret_acc = sum(ret_accs) / len(ret_accs)

    # else:
    #     # NQ and TriviaQA on
    #     oracle_doc_list = deepcopy([oracle['oracle_doc'] for oracle in oracle_list])
    #     perturb_placeholder = [i for i in range(len(oracle_doc_list))]
    #     cur_ret_acc = len(perturb_placeholder) / len(oracle_doc_list)
    #     while cur_ret_acc > ret_acc:
    #         perturb_idx = random.sample(perturb_placeholder, 1)[0]  # pick an oracle key and perturb
    #         perturb_placeholder.remove(perturb_idx)
    #         oracle_docs_list[]

    if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
        docs = WikiCorpusLoader().get_docs(oracle_docs_list, dataset, num_procs=8)
    else:
        docs = [PythonDocsLoader().get_docs(oracle_docs) for oracle_docs in oracle_docs_list]

    return oracle_docs_list, docs


# def get_top_k_docs(oracle_list, ret_results, top_k, dataset):
#     ret_doc_keys_list = []
#     for oracle in oracle_list:
#         ret_doc_keys = [item['doc_key'] for item in ret_results[oracle['qs_id']]]
#         ret_doc_keys_list.append(ret_doc_keys[:top_k])
#     if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
#         docs_list = WikiCorpusLoader().get_docs(ret_doc_keys_list, dataset, num_procs=8)
#     else:
#         docs_list = [PythonDocsLoader().get_docs(doc_keys) for doc_keys in ret_doc_keys_list]
#     return ret_doc_keys_list, docs_list



def perturb_ret_doc_type(perturb_doc_type, oracle_list, ret_results, model, dataset):
    """
    generate retrieval doc key of each sample based on ret_doc_type, return a list of the docs for each sample
    :param ret_doc_type:
    :return:
    """
    if dataset == 'NQ' or dataset == 'TriviaQA':
        for idx, oracle in enumerate(oracle_list):
            oracle_list[idx]['oracle_docs'] = [oracle_list[idx]['oracle_doc']]

    if perturb_doc_type in ['irrelevant_diff', 'irrelevant_dummy', 'ellipsis']:
        perturb_doc_type = perturb_doc_type if perturb_doc_type == 'ellipsis' else perturb_doc_type.split('_')[1]
        oracle_doc_keys_list = [oracle['oracle_docs'] for oracle in oracle_list]    # get oracle docs list
        if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
            oracle_docs_list = WikiCorpusLoader().get_docs(oracle_doc_keys_list, dataset, num_procs=8)
        else:
            oracle_docs_list = [PythonDocsLoader().get_docs(oracle_docs) for oracle_docs in oracle_doc_keys_list]
        docs_list, doc_keys_list = [], []   # required docs and doc keys
        for oracle_docs in oracle_docs_list:
            docs_list.append(get_irrelevant_docs(irrelevant_type=perturb_doc_type, oracle_docs=oracle_docs, model=model, dataset=dataset))
    else:
        if perturb_doc_type == 'oracle':
            doc_keys_list = [oracle['oracle_docs'] for oracle in oracle_list]
        elif perturb_doc_type.startswith('retrieved'):
            doc_keys_list = []
            for oracle in oracle_list:
                ret_doc_keys = [item['doc_key'] for item in ret_results[oracle['qs_id']]]
                if 'top' in perturb_doc_type: k = 10 if dataset in ['NQ', 'TriviaQA', 'hotpotQA'] else 5
                else: k = len(oracle['oracle_docs'])
                doc_keys_list.append(ret_doc_keys[:k])
        elif perturb_doc_type == 'distracting':
            doc_keys_list = []
            for oracle in oracle_list:
                doc_keys_list.append(get_distracting_docs(ret_result=ret_results[oracle['qs_id']], oracle_docs=oracle['oracle_docs'], dataset=dataset, k=len(oracle['oracle_docs'])))
        elif perturb_doc_type == 'random':
            doc_keys_list = []
            if dataset in ['NQ', 'TriviaQA', 'hotpotQA']: corpus_id_list = WikiCorpusLoader().load_wiki_id(dataset)
            else: corpus_id_list = PythonDocsLoader().load_api_signs()
            for oracle in oracle_list:
                doc_keys_list.append(random.sample(corpus_id_list, k=len(oracle['oracle_docs'])))
        elif perturb_doc_type == 'none':
            doc_keys_list = []
        else:
            raise ValueError('not supported perturb_doc_type {}'.format(perturb_doc_type))

        if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
            docs_list = WikiCorpusLoader().get_docs(doc_keys_list, dataset, num_procs=8)
        else:
            docs_list = [PythonDocsLoader().get_docs(doc_keys) for doc_keys in doc_keys_list]

    return doc_keys_list, docs_list


def _truncate_prompt_to_control_prompt_length(dataset, cur_pl, target_pl, cur_docs, doc_keys, model):
    if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:  # for qa task, mean doc length is 100, keep prompt length in [-50, 50] -> if doc above 50 remove, else append
        if cur_pl - target_pl > 50:
            docs = cur_docs[:-1]
        else:
            docs = cur_docs
    else:  # for code task, mean doc length is 200, but doc length vary, to keep [-100, 100]: if keep -> prompt above less than 100, then keep, if remove prompt lower less 100, then remove, else truncate
        if cur_pl - target_pl < 100:
            docs = cur_docs
        else:
            last_doc_length = get_docs_tokens([cur_docs[-1]], model)[0]
            if cur_pl - last_doc_length < target_pl - 100:
                docs = cur_docs[:-1]
                truncated_last_doc = truncate_docs([cur_docs[-1]], model, max_length=int(target_pl - (cur_pl - last_doc_length)))[0]
                docs.append(truncated_last_doc)
            else:
                docs = cur_docs[:-1]
    return doc_keys[:len(docs)], docs


def get_prompt_of_target_pl(dataset, target_pl, docs, doc_keys, model, question, generate_func):
    for doc_idx in range(len(docs)):
        cur_docs = docs[:doc_idx]
        cur_prompt = generate_func(cur_docs, question, model)
        prompt = cur_prompt[0] + cur_prompt[1] if 'gpt' in model else cur_prompt
        cur_pl = get_docs_tokens([prompt], model)[0]
        if cur_pl > target_pl: break
    doc_keys, docs = _truncate_prompt_to_control_prompt_length(dataset=dataset, cur_pl=cur_pl, target_pl=target_pl, cur_docs=cur_docs, doc_keys=doc_keys, model=model)
    prompt = generate_func(docs, question, model)
    return doc_keys, docs, prompt


def gene_prompts_for_pl_analysis(pl_analysis, oracle_list, qs_list, ret_results, model, dataset, doc_max_length):
    try:
        target_pl = int(pl_analysis.rsplit('_', 1)[1])
    except:
        raise ValueError('not supported pl_analysis type {}'.format(pl_analysis))

    def get_oracle_docs(dataset, oracle_list):
        # get oracle docs
        if dataset == 'NQ' or dataset == 'TriviaQA':
            for idx, oracle in enumerate(oracle_list):
                oracle_list[idx]['oracle_docs'] = [oracle_list[idx]['oracle_doc']]
        oracle_doc_keys_list = [oracle['oracle_docs'] for oracle in oracle_list]
        if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
            oracle_docs_list = WikiCorpusLoader().get_docs(oracle_doc_keys_list, dataset, num_procs=8)
        else:
            oracle_docs_list = [PythonDocsLoader().get_docs(oracle_docs) for oracle_docs in oracle_doc_keys_list]
        return oracle_doc_keys_list, oracle_docs_list

    target_doc_keys_list, prompts = [], []
    if pl_analysis.startswith('oracle'):
        oracle_doc_keys_list, oracle_docs_list = get_oracle_docs(dataset=dataset, oracle_list=oracle_list)
        start_time = time.time()
        # repeat oracle docs until the target prompt length
        generate_func = _get_generate_func(dataset=dataset, no_ret_flag=False, prompt_type='0shot')
        for qs, oracle_docs, oracle_doc_keys in zip(qs_list, oracle_docs_list, oracle_doc_keys_list):
            if dataset in ['conala', 'DS1000', 'pandas_numpy_eval']: oracle_docs = truncate_docs(oracle_docs, model=model, max_length=doc_max_length)
            repeated_oracle_docs, repeated_oracle_doc_keys = oracle_docs * 100, oracle_doc_keys * 100
            doc_keys, docs, prompt = get_prompt_of_target_pl(dataset=dataset, target_pl=target_pl, docs=repeated_oracle_docs,
                                                       doc_keys=repeated_oracle_doc_keys, model=model, question=qs['question'], generate_func=generate_func)
            if pl_analysis.startswith('oracle_pad'):    # oracle_pad_ellipsis_2000
                irrelevant_type = pl_analysis.split('_')[-2]
                padded_docs = docs[len(oracle_docs):]
                irrelevant_docs = get_irrelevant_docs(irrelevant_type=irrelevant_type, oracle_docs=padded_docs, model=model, dataset=dataset)
                if 'reverse' not in pl_analysis: prompt = generate_func(irrelevant_docs+oracle_docs, qs['question'], model)  # add oracle in the last, near the question
                else: prompt = generate_func(oracle_docs+irrelevant_docs, qs['question'], model)    # oracle_pad_reverse_ellipsis_2000
            target_doc_keys_list.append(doc_keys)
            prompts.append(prompt)
        print('time cost: ', time.time() - start_time)

    elif pl_analysis.startswith('distracting'):
        if dataset == 'NQ' or dataset == 'TriviaQA':
            for idx, oracle in enumerate(oracle_list):
                oracle_list[idx]['oracle_docs'] = [oracle_list[idx]['oracle_doc']]
        distracting_doc_keys_list = []
        for oracle in oracle_list:
            distracting_doc_keys_list.append(get_distracting_docs(ret_result=ret_results[oracle['qs_id']], dataset=dataset, k=len(oracle['oracle_docs']), oracle_docs=oracle['oracle_docs']))
        if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
            distracting_docs_list = WikiCorpusLoader().get_docs(distracting_doc_keys_list, dataset, num_procs=8)
        else:
            distracting_docs_list = [PythonDocsLoader().get_docs(doc_keys) for doc_keys in distracting_doc_keys_list]
        generate_func = _get_generate_func(dataset=dataset, no_ret_flag=False, prompt_type='0shot')
        for qs, distracting_docs, distracting_doc_keys in zip(qs_list, distracting_docs_list, distracting_doc_keys_list):
            if dataset in ['conala', 'DS1000', 'pandas_numpy_eval']: distracting_docs = truncate_docs(distracting_docs, model=model, max_length=doc_max_length)
            repeated_distracting_docs, repeated_distracting_doc_keys = distracting_docs * 100, distracting_doc_keys * 100
            doc_keys, docs, prompt = get_prompt_of_target_pl(dataset=dataset, target_pl=target_pl, docs=repeated_distracting_docs,
                                                             doc_keys=repeated_distracting_doc_keys, model=model, question=qs['question'], generate_func=generate_func)
            if pl_analysis.startswith('distracting_pad'):    # oracle_pad_ellipsis_2000
                irrelevant_type = pl_analysis.split('_')[-2]
                padded_docs = docs[len(distracting_docs):]
                irrelevant_docs = get_irrelevant_docs(irrelevant_type=irrelevant_type, oracle_docs=padded_docs, model=model, dataset=dataset)
                if 'reverse' not in pl_analysis: prompt = generate_func(irrelevant_docs+distracting_docs, qs['question'], model)  # add oracle in the last, near the question
                else: prompt = generate_func(distracting_docs+irrelevant_docs, qs['question'], model)  # distracting_pad_reverse_ellipsis_2000
            target_doc_keys_list.append(doc_keys)
            prompts.append(prompt)

    elif pl_analysis.startswith('retrieved'):
        if dataset == 'NQ' or dataset == 'TriviaQA':
            for idx, oracle in enumerate(oracle_list):
                oracle_list[idx]['oracle_docs'] = [oracle_list[idx]['oracle_doc']]
        retrieved_docs_list = []
        retrieved_doc_keys_list = []
        ret_results_docs = get_docs_for_ret_results(ret_results, dataset)
        for oracle in oracle_list:
            if 'top' in pl_analysis: k = 10 if dataset in ['NQ', 'TriviaQA', 'hotpotQA'] else 5
            else: k = len(oracle['oracle_docs'])
            retrieved_docs_list.append([item['doc'] for item in ret_results_docs[oracle['qs_id']][:k]])
            retrieved_doc_keys_list.append([item['doc_key'] for item in ret_results_docs[oracle['qs_id']][:k]])
        generate_func = _get_generate_func(dataset=dataset, no_ret_flag=False, prompt_type='0shot')
        for qs, retrieved_docs, retrieved_doc_keys in zip(qs_list, retrieved_docs_list, retrieved_doc_keys_list):
            if dataset in ['conala', 'DS1000', 'pandas_numpy_eval']: retrieved_docs = truncate_docs(retrieved_docs, model=model, max_length=doc_max_length)
            repeated_retrieved_docs, repeated_retrieved_doc_keys = retrieved_docs * 100, retrieved_doc_keys * 100
            doc_keys, docs, prompt = get_prompt_of_target_pl(dataset=dataset, target_pl=target_pl, docs=repeated_retrieved_docs,
                                                             doc_keys=repeated_retrieved_doc_keys, model=model, question=qs['question'], generate_func=generate_func)
            if 'pad' in pl_analysis:    # retrieved_top_pad_ellipsis_2000
                irrelevant_type = pl_analysis.split('_')[-2]
                padded_docs = docs[len(retrieved_docs):]
                irrelevant_docs = get_irrelevant_docs(irrelevant_type=irrelevant_type, oracle_docs=padded_docs, model=model, dataset=dataset)
                if 'reverse' not in pl_analysis: prompt = generate_func(irrelevant_docs+retrieved_docs, qs['question'], model)  # add oracle in the last, near the question
                else: prompt = generate_func(retrieved_docs+irrelevant_docs, qs['question'], model)     # retrieved_top_pad_reverse_ellipsis_2000
            target_doc_keys_list.append(doc_keys)
            prompts.append(prompt)

    elif pl_analysis.startswith('random'):
        if dataset == 'NQ' or dataset == 'TriviaQA':
            for idx, oracle in enumerate(oracle_list):
                oracle_list[idx]['oracle_docs'] = [oracle_list[idx]['oracle_doc']]
        # get random docs
        if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
            corpus_id_list = WikiCorpusLoader().load_wiki_id(dataset)
        else:
            corpus_id_list = PythonDocsLoader().load_api_signs()
        random_doc_keys_list = []
        for oracle in oracle_list:
            if 'repeat' in pl_analysis: random_doc_keys_list.append(random.sample(corpus_id_list, k=len(oracle['oracle_docs']))*100)
            else: random_doc_keys_list.append(random.sample(corpus_id_list, 100))
        if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
            random_docs_list = WikiCorpusLoader().get_docs(random_doc_keys_list, dataset, num_procs=8)
        else:
            random_docs_list = [PythonDocsLoader().get_docs(doc_keys) for doc_keys in random_doc_keys_list]
        start_time = time.time()
        generate_func = _get_generate_func(dataset=dataset, no_ret_flag=False, prompt_type='0shot')
        for qs, random_docs, random_doc_keys in zip(qs_list, random_docs_list, random_doc_keys_list):
            if dataset in ['conala', 'DS1000', 'pandas_numpy_eval']: random_docs = truncate_docs(random_docs, model=model, max_length=doc_max_length)
            doc_keys, docs, prompt = get_prompt_of_target_pl(dataset=dataset, target_pl=target_pl, docs=random_docs,
                                                       doc_keys=random_doc_keys, model=model, question=qs['question'], generate_func=generate_func)
            target_doc_keys_list.append(doc_keys)
            prompts.append(prompt)
        print('time cost: ', time.time() - start_time)

    elif pl_analysis.startswith('irrelevant') or pl_analysis.startswith('ellipsis'):
        # get oracle docs, and use get_irrelevant docs to control the same prompt length
        oracle_doc_keys_list, oracle_docs_list = get_oracle_docs(dataset=dataset, oracle_list=oracle_list)
        # if 'pretend' in pl_analysis: generate_func = _get_generate_func(dataset=dataset, no_ret_flag=False, prompt_type='pretend')
        # else: generate_func = _get_generate_func(dataset=dataset, no_ret_flag=False, prompt_type='0shot')
        generate_func = _get_generate_func(dataset=dataset, no_ret_flag=False, prompt_type='0shot')
        if 'ellipsis' in pl_analysis: irrelevant_type = 'ellipsis'
        else: irrelevant_type = pl_analysis.split('_')[1]
        for qs, oracle_docs, oracle_doc_keys in zip(qs_list, oracle_docs_list, oracle_doc_keys_list):
            if dataset in ['conala', 'DS1000', 'pandas_numpy_eval']: oracle_docs = truncate_docs(oracle_docs, model=model, max_length=doc_max_length)
            if 'repeat' in pl_analysis:
                irrelevant_docs = get_irrelevant_docs(irrelevant_type=irrelevant_type, oracle_docs=oracle_docs, model=model, dataset=dataset)
                repeated_irrelevant_docs, doc_keys = irrelevant_docs*100, oracle_doc_keys*100
                doc_keys, irrelevant_docs, prompt = get_prompt_of_target_pl(dataset=dataset, target_pl=target_pl, docs=repeated_irrelevant_docs, doc_keys=doc_keys, model=model,
                                                                 question=qs['question'], generate_func=generate_func)
            else:
                repeated_oracle_docs, repeated_oracle_doc_keys = oracle_docs * 100, oracle_doc_keys * 100
                doc_keys, docs, prompt = get_prompt_of_target_pl(dataset=dataset, target_pl=target_pl, docs=repeated_oracle_docs,
                                                                 doc_keys=repeated_oracle_doc_keys, model=model, question=qs['question'], generate_func=generate_func)
                irrelevant_docs = get_irrelevant_docs(irrelevant_type=irrelevant_type, oracle_docs=docs, model=model, dataset=dataset)
                prompt = generate_func(irrelevant_docs, qs['question'], model)
            prompts.append(prompt)

    elif pl_analysis.startswith('none'):
        generate_func = _get_generate_func(dataset=dataset, no_ret_flag=True, prompt_type='0shot')
        irrelevant_type = pl_analysis.split('_')[2]
        # ellipses = '................................................................'*100 + '\n'
        # ellipses_length = get_docs_tokens([ellipses], model)[0]
        for qs in qs_list:
            original_prompt = generate_func(qs['question'], model)
            original_pl = get_docs_tokens([original_prompt] if 'llama' in model else [original_prompt[0]+original_prompt[1]], model)[0]
            # target_ellipses = ellipses * (int(target_pl / ellipses_length)) * 2     # pad ellipses
            # target_ellipses = truncate_docs(docs=[target_ellipses], model=model, max_length=target_pl-original_pl)[0]
            target_pl_new = target_pl - original_pl
            if irrelevant_type == 'dummy' or irrelevant_type == 'ellipsis':
                if irrelevant_type == 'dummy':
                    dummy_string = 'The wiggly fluff went plop while the jibber-jabber bumbled and tumbled. Fizzle-flop danced around the wibbly-wobbly doodle, and snicker-snack bounced happily. Doodle-doo twirled and swirled in the zigzag zoom, and snuggle-bug snuggled close. Wobble-wobble wandered through the dilly-dally, giggling and jiggling all the while. Squiggle-squabble and waddle-waddle wobbled along, playing in the silly-sally world of random wozzle. The snickety-snack skipped and hopped, while the flibber-jabber giggled and squiggled. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Bloop bloop bloop, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Nulla facilisi. Snick snick snick, jibble jibble jibble.'
                else:
                    dummy_string = '................................................................' * 100 + '\n'
                dummy_length = get_docs_tokens(docs=[dummy_string], model=model)[0]
                padded_doc = dummy_string * (int(target_pl_new / dummy_length) + 1) * 2
                padded_doc = truncate_docs(docs=[padded_doc], model=model, max_length=target_pl_new)[0]
            else:
                if dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
                    loader = WikiCorpusLoader(prepare_random_docs=True)
                else:
                    loader = PythonDocsLoader(prepare_random_docs=True)
                padded_doc = ''
                padded_doc_length = 0
                while padded_doc_length < target_pl_new:
                    diff_doc = ' '.join(loader.get_random_docs(10))
                    random_doc_length = get_docs_tokens(docs=[diff_doc], model=model)[0]
                    padded_doc += diff_doc
                    padded_doc_length += random_doc_length
                padded_doc = truncate_docs(docs=[padded_doc], model=model, max_length=target_pl_new)[0]
            prompt = generate_func(qs['question'], model, pads=padded_doc)
            prompts.append(prompt)

    elif pl_analysis.startswith('self_gene'):
        generate_func = _get_generate_func(dataset=dataset, no_ret_flag=True, prompt_type=pl_analysis)
        for qs in qs_list:
            prompt = generate_func(qs['question'], model)
            prompts.append(prompt)

    else:
        raise ValueError('not supported prompt length analysis {}'.format(pl_analysis))
    pl_list = approximate_token(prompts, model)
    return target_doc_keys_list, prompts, pl_list


def select_by_simi_score(ret_results, doc_selection_type, oracle_list, dataset):
    try:
        top_p = float(doc_selection_type.replace('simi_score_', ''))
    except:
        raise ValueError('invalid selection type format {}'.format(doc_selection_type))
    # type 1: use global max min
    # simi_score_min, simi_score_max = 0, 0
    # for qs_id in ret_results.keys():
    #     for item in ret_results[qs_id]:
    #         simi_score_max = item['score'] if simi_score_max < item['score'] else simi_score_max
    #         simi_score_min = item['score'] if simi_score_min > item['score'] else simi_score_min
    # threshold = (simi_score_max - simi_score_min) * (1 - top_p) + simi_score_min
    # count_ret_docs_above = []
    # for qs_id in ret_results:
    #     count_ret_docs_above.append(len([item for item in ret_results[qs_id] if item['score'] > threshold]))
    # print(count_ret_docs_above)
    # avg_ret_docs_above = sum(count_ret_docs_above) / len(ret_results)
    # print(avg_ret_docs_above)
    # type 2: use local max min
    if dataset in ['NQ', 'TriviaQA', 'hotpointQA']: max_doc_num = 10
    else: max_doc_num = 5
    count_ret_docs_above = []
    ret_doc_keys_list = []
    for qs_id in ret_results.keys():    # todo: need to use oracle list to iterate
        scores_list = [item['score'] for item in ret_results[qs_id]]
        simi_score_min, simi_score_max = min(scores_list), max(scores_list)
        threshold = (simi_score_max - simi_score_min) * (1 - top_p) + simi_score_min
        count_ret_docs_above.append(len([item for item in ret_results[qs_id] if item['score'] > threshold][:max_doc_num]))
        ret_doc_keys_list.append([item['doc_key'] for item in ret_results[qs_id] if item['score'] > threshold][:max_doc_num])
    # print(count_ret_docs_above)
    # avg_ret_docs_above = sum(count_ret_docs_above) / len(ret_results)
    # print('{:.3f}'.format(avg_ret_docs_above))
    return ret_doc_keys_list

def select_by_rerank(ret_results, doc_selection_type):
    try:
        rerank_method = float(doc_selection_type.split('_')[1])
        assert rerank_method in ['cohere', 'gpt']
    except:
        raise ValueError('invalid selection type format {}'.format(doc_selection_type))


def get_docs_for_ret_results(ret_results, dataset):
    ret_docs_path = os.path.join(root_path, 'data', dataset, 'ret_results_docs_openai-embedding.json')
    if os.path.exists(ret_docs_path):
        ret_results_docs = json.load(open(ret_docs_path, 'r'))
    else:
        # get docs
        ret_doc_keys_list = []
        for qs_id in ret_results.keys():
            ret_result = ret_results[qs_id]
            ret_doc_keys_list.append([item['doc_key'] for item in ret_result[:100]])
        if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
            docs_list = WikiCorpusLoader().get_docs(ret_doc_keys_list, dataset, num_procs=8)
        else:
            docs_list = [PythonDocsLoader().get_docs(doc_keys) for doc_keys in ret_doc_keys_list]
        # store
        ret_results_docs = dict()
        for doc_keys, docs, qs_id in zip(ret_doc_keys_list, docs_list, ret_results.keys()):
            ret_results_docs[qs_id] = [dict(doc_key=doc_key, doc=doc) for doc_key, doc in zip(doc_keys, docs)]
        with open(ret_docs_path, 'w+') as f:
            json.dump(ret_results_docs, f, indent=2)
    return ret_results_docs


def gene_prompts_by_prompt_length(ret_results, doc_selection_type, qs_list, dataset, model, doc_max_length):
    """
    this function is for doc selection: control prompt length, directly return the target prompt
    :param ret_results:
    :param doc_selection_type:
    :param oracle_list:
    :param dataset:
    :param model:
    :return:
    """
    try:
        target_pl = float(doc_selection_type.replace('pl_', ''))
    except:
        raise ValueError('invalid prompt length selection format {}'.format(doc_selection_type))
    generate_func = _get_generate_func(dataset=dataset, no_ret_flag=False, prompt_type='0shot')
    ret_results_docs = get_docs_for_ret_results(ret_results, dataset)
    prompts = []
    ret_doc_keys_list = []
    for qs in qs_list:
        ret_result_docs = ret_results_docs[qs['qs_id']]
        ret_docs = [item['doc'] for item in ret_result_docs]
        ret_doc_keys = [item['doc_key'] for item in ret_result_docs]
        if dataset in ['conala', 'DS1000', 'pandas_numpy_eval']: ret_docs = truncate_docs(ret_docs, model=model, max_length=doc_max_length)
        doc_keys, docs, prompt = get_prompt_of_target_pl(dataset=dataset, target_pl=target_pl, docs=ret_docs,
                                                         doc_keys=ret_doc_keys, model=model, question=qs['question'], generate_func=generate_func)
        ret_doc_keys_list.append(doc_keys)
        prompts.append(prompt)
        # for doc_idx in range(len(ret_result_docs)):
        #     cur_ret_docs = [item['doc'] for item in ret_result_docs[:doc_idx]]
        #     if dataset in ['conala', 'DS1000', 'pandas_numpy_eval']: cur_ret_docs = truncate_docs(cur_ret_docs, model=model, max_length=doc_max_length)
        #     cur_prompt = generate_func(cur_ret_docs, qs['question'], model)
        #     prompt = cur_prompt[0] + cur_prompt[1] if 'gpt' in model else cur_prompt
        #     cur_pl = get_docs_tokens([prompt], model)[0]
        #     if cur_pl > target_pl: break
        # if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:   # for qa task, mean doc length is 100, keep prompt length in [-50, 50] -> if doc above 50 remove, else append
        #     if cur_pl - target_pl > 50:
        #         ret_docs = cur_ret_docs[:-1]
        #     else: ret_docs = cur_ret_docs
        # else:   # for code task, mean doc length is 200, but doc length vary, to keep [-100, 100]: if keep -> prompt above less than 100, then keep, if remove prompt lower less 100, then remove, else truncate
        #     if cur_pl - target_pl < 100: ret_docs = cur_ret_docs
        #     else:
        #         last_doc_length = get_docs_tokens([cur_ret_docs[-1]], model)[0]
        #         if cur_pl - last_doc_length < target_pl - 100:
        #             ret_docs = cur_ret_docs[:-1]
        #             truncated_last_doc = truncate_docs([cur_ret_docs[-1]], model, max_length=int(target_pl-(cur_pl-last_doc_length)))[0]
        #             ret_docs.append(truncated_last_doc)
        #         else:
        #             ret_docs = cur_ret_docs[:-1]
        # ret_doc_keys = [item['doc_key'] for item in ret_result_docs[:len(ret_docs)]]
        # ret_doc_keys_list.append(ret_doc_keys)
        # prompt = generate_func(ret_docs, qs['question'], model)
        # prompts.append(prompt)
    pl_list = approximate_token(prompts, model)
    return ret_doc_keys_list, prompts, pl_list



def select_retrieval_docs(ret_results, oracle_list, doc_selection_type, dataset):
    if 'top' in doc_selection_type:
        try:
            top_k = int(doc_selection_type.split('_')[1])
        except:
            raise ValueError('invalid selection type format {}'.format(doc_selection_type))
        ret_doc_keys_list = []
        for oracle in oracle_list:
            ret_doc_keys_list.append([item['doc_key'] for item in ret_results[oracle['qs_id']][:top_k]])
    elif 'simi_score' in doc_selection_type:
        ret_doc_keys_list = select_by_simi_score(ret_results=ret_results, doc_selection_type=doc_selection_type, oracle_list=oracle_list, dataset=dataset)
    # elif 'rerank' in doc_selection_type:
    #     ret_doc_keys_list = select_by_rerank(ret_results, doc_selection_type)
    else:
        raise ValueError('not supported doc_selection_type {}'.format(doc_selection_type))

    if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
        docs_list = WikiCorpusLoader().get_docs(ret_doc_keys_list, dataset, num_procs=8)
    else:
        docs_list = [PythonDocsLoader().get_docs(doc_keys) for doc_keys in ret_doc_keys_list]
    return ret_doc_keys_list, docs_list


def _get_generate_func(dataset, no_ret_flag, prompt_type):
    if no_ret_flag is True:     # no retrieval
        if dataset in ['NQ', 'TriviaQA']:
            if prompt_type == '0shot':
                generate_func = NQ_TriviaQA_prompt.prompt_0shot_no_ret
            elif prompt_type == 'self_gene':
                generate_func = NQ_TriviaQA_prompt.prompt_self_gene
            elif prompt_type == 'self_gene_2':
                generate_func = NQ_TriviaQA_prompt.prompt_self_gene_2
            else:
                raise ValueError(f'Invalid prompt type: {prompt_type} for dataset {dataset}')
        elif dataset == 'conala':
            if prompt_type == '0shot':
                generate_func = conala_prompt.prompt_0shot_no_ret
            else:
                raise ValueError(f'Invalid prompt type: {prompt_type} for dataset {dataset}')
        elif dataset == 'DS1000':
            if prompt_type == '0shot':
                generate_func = DS1000_prompt.prompt_0shot_no_ret
            else:
                raise ValueError(f"Invalid prompt type: {prompt_type} for dataset {dataset}")
        elif dataset == 'pandas_numpy_eval':
            if prompt_type == '0shot':
                generate_func = pandas_numpy_eval_prompt.prompt_0shot_no_ret
            else:
                raise ValueError(f"Invalid prompt type: {prompt_type} for dataset {dataset}")
        elif dataset == 'hotpotQA':
            if prompt_type == '0shot':
                generate_func = hotpotQA_prompt.prompt_0shot_no_ret
            else:
                raise ValueError(f"Invalid prompt type: {prompt_type} for dataset {dataset}")
        else:
            raise ValueError(f'invalid dataset {dataset}')
    else:
        if dataset == 'NQ' or dataset == 'TriviaQA':
            if prompt_type == '0shot':
                generate_func = NQ_TriviaQA_prompt.prompt_0shot
            elif prompt_type == '3shot':
                generate_func = NQ_TriviaQA_prompt.prompt_3shot
            # elif prompt_type == 'pretend':
            #     generate_func = NQ_TriviaQA_prompt.prompt_pretend
            # elif prompt_type == 'self_pad':
            #     generate_func = NQ_TriviaQA_prompt.prompt_self_pad
            elif prompt_type == 'least_to_most':
                generate_func = NQ_TriviaQA_prompt.prompt_least_to_most
            elif prompt_type == 'plan_and_solve':
                generate_func = NQ_TriviaQA_prompt.prompt_plan_and_solve
            elif prompt_type == 'cot' or prompt_type == 'self-consistency':
                generate_func = NQ_TriviaQA_prompt.prompt_cot
            else:
                raise ValueError(f"Invalid prompt type: {prompt_type} for dataset {dataset}")
        elif dataset == 'conala':
            if prompt_type == '0shot':
                generate_func = conala_prompt.prompt_0shot
            elif prompt_type == '3shot':
                generate_func = conala_prompt.prompt_3shot
            elif prompt_type == 'least_to_most':
                generate_func = conala_prompt.prompt_least_to_most
            elif prompt_type == 'plan_and_solve':
                generate_func = conala_prompt.prompt_plan_and_solve
            elif prompt_type == 'cot' or prompt_type == 'self-consistency':
                generate_func = conala_prompt.prompt_cot
            else:
                raise ValueError(f"Invalid prompt type: {prompt_type} for dataset {dataset}")
        elif dataset == 'DS1000':
            if prompt_type == '0shot':
                generate_func = DS1000_prompt.prompt_0shot
            elif prompt_type == '3shot':
                generate_func = DS1000_prompt.prompt_3shot
            elif prompt_type == 'least_to_most':
                generate_func = DS1000_prompt.prompt_least_to_most
            elif prompt_type == 'plan_and_solve':
                generate_func = DS1000_prompt.prompt_plan_and_solve
            elif prompt_type == 'cot' or prompt_type == 'self-consistency':
                generate_func = DS1000_prompt.prompt_cot
            else:
                raise ValueError(f"Invalid prompt type: {prompt_type} for dataset {dataset}")
        elif dataset == 'pandas_numpy_eval':
            if prompt_type == '0shot':
                generate_func = pandas_numpy_eval_prompt.prompt_0shot
            elif prompt_type == '3shot':
                generate_func = pandas_numpy_eval_prompt.prompt_3shot
            elif prompt_type == 'least_to_most':
                generate_func = pandas_numpy_eval_prompt.prompt_least_to_most
            elif prompt_type == 'plan_and_solve':
                generate_func = pandas_numpy_eval_prompt.prompt_plan_and_solve
            elif prompt_type == 'cot' or prompt_type == 'self-consistency':
                generate_func = pandas_numpy_eval_prompt.prompt_cot
            else:
                raise ValueError(f"Invalid prompt type: {prompt_type} for dataset {dataset}")
        elif dataset == 'hotpotQA':
            if prompt_type == '0shot':
                generate_func = hotpotQA_prompt.prompt_0shot
            elif prompt_type == '3shot':
                generate_func = hotpotQA_prompt.prompt_3shot
            elif prompt_type == 'least_to_most':
                generate_func = hotpotQA_prompt.prompt_least_to_most
            elif prompt_type == 'plan_and_solve':
                generate_func = hotpotQA_prompt.prompt_plan_and_solve
            elif prompt_type == 'cot' or prompt_type == 'self-consistency':
                generate_func = hotpotQA_prompt.prompt_cot
            else:
                raise ValueError(f"Invalid prompt type: {prompt_type} for dataset {dataset}")
        else:
            raise ValueError(f'invalid dataset {dataset}')
    return generate_func


def generate_prompts(questions, ret_docs_list, prompt_type, dataset, model_name, doc_max_length):
    generate_func = _get_generate_func(dataset=dataset, no_ret_flag=True if len(ret_docs_list) == 0 else False, prompt_type=prompt_type)   # get prompt gene func
    if len(ret_docs_list) == 0:     # no retrieval
        prompts = []
        for question in questions:
            prompts.append(generate_func(question, model_name))
    else:
        _ret_docs_list = list()
        for docs in ret_docs_list:
            _ret_docs_list.append(truncate_docs(docs, model_name, doc_max_length))
        ret_docs_list = _ret_docs_list
        prompts = []
        for ret_docs, question in zip(ret_docs_list, questions):
            prompts.append(generate_func(ret_docs, question, model_name))

    pl_list = approximate_token(prompts, model_name)
    # if exceed prompt limit, truncate more docs
    token_limit = 8000 if 'gpt' in model_name else 8000
    for idx, pl in enumerate(pl_list):
        new_doc_max_length = doc_max_length
        while pl > token_limit:
            new_doc_max_length -= 100
            ret_docs_list[idx] = truncate_docs(ret_docs_list[idx], model_name, new_doc_max_length)
            prompt = generate_func(ret_docs_list[idx], questions[idx], model_name)
            pl = get_docs_tokens(docs=[prompt[0]+prompt[1]] if 'gpt' in model_name else [prompt], model=model_name)[0]
            prompts[idx] = prompt
    pl_list = approximate_token(prompts, model_name)
    return prompts, pl_list





if __name__ == "__main__":
    """test control prompt length"""
    in_program_call = None
    # in_program_call = '--model llama2-13b-chat --temperature 0 --n 1 --dataset conala --retriever openai-embedding --analysis_type retrieval_doc_selection --doc_selection_type pl_1000'
    in_program_call = '--model gpt-3.5-turbo-0125 --temperature 0 --n 1 --dataset conala --retriever openai-embedding --analysis_type prompt_length --pl_analysis distracting_4000'  # random
    args = generate_config(in_program_call)
    # loader = NQTriviaQAUtils(dataset='NQ')
    loader = ConalaLoader()
    # loader = HotpotQAUtils()
    qs_list = loader.load_qs_list()[7:8]
    oracle_list = loader.load_oracle_list()[7:8]
    ret_results = get_ret_results(dataset=args.dataset, retriever='openai-embedding')
    # ret_doc_keys_list, prompts, pl_list = gene_prompts_by_prompt_length(ret_results=ret_results, doc_selection_type=args.doc_selection_type, qs_list=qs_list, dataset=args.dataset, model=args.model, doc_max_length=args.doc_max_length)
    doc_keys_list, prompts, pl_list = gene_prompts_for_pl_analysis(pl_analysis=args.pl_analysis, oracle_list=oracle_list, qs_list=qs_list, ret_results=ret_results,
                                                                   model=args.model, dataset=args.dataset, doc_max_length=args.doc_max_length)
    print(prompts[0][0])
    print(prompts[0][1])
    print(get_docs_tokens([prompts[0][0] + prompts[0][1]], model=args.model))
    # print(prompts[1][1])
    # print(doc_keys_list)
    # # true_pl_list = get_docs_tokens([prompt[0]+prompt[1] for prompt in prompts], model=args.model)
    # true_pl_list = get_docs_tokens(prompts, args.model)
    # for pl, true_pl in zip(pl_list, true_pl_list):
    #     print(pl, true_pl)



    """a temp run to get docs for ret results"""
    # in_program_call = None
    # # in_program_call = '--model gpt-3.5-turbo-0125 --temperature 0 --n 1 --dataset NQ --retriever openai-embedding --analysis_type retrieval_doc_selection'
    # args = generate_config(in_program_call)
    # ret_results = get_ret_results(dataset=args.dataset, retriever=args.retriever)
    # get_docs_for_ret_results(ret_results=ret_results, dataset=args.dataset)


    """test control ret type"""
    # loader = DS1000Loader()
    # oracle_list = loader.load_oracle_list()
    # qs_list = loader.load_qs_list()
    # ret_results = get_ret_results(dataset='DS1000', retriever='openai-embedding')
    # # print([oracle['oracle_docs'] for oracle in oracle_list])
    # oracle_list = oracle_list[:1]
    # perturb_oracle_keys, docs = perturb_ret_doc_type(perturb_doc_type='irrelevant_dummy', oracle_list=oracle_list, ret_results=ret_results,
    #                                                  model='codellama-13b-instruct', dataset='DS1000')
    # print(docs[0])
    # print(get_docs_tokens(docs[0], model='codellama-13b-instruct'))
    # oracle_docs = PythonDocsLoader().get_docs(oracle_list[0]['oracle_docs'])
    # print(get_docs_tokens(oracle_docs, model='codellama-13b-instruct'))


    """
    test control ret_acc
    """
    # dataset = 'TriviaQA'
    # loader = NQTriviaQAUtils(dataset)
    # # loader = DS1000Loader()
    # # loader = HotpotQAUtils()
    # oracle_list = loader.load_oracle_list()
    # qs_list = loader.load_qs_list()
    # ret_results = get_ret_results(dataset=dataset, retriever='openai-embedding')
    # perturb_oracle_keys, docs = control_ret_acc(ret_acc=0.2, oracle_list=oracle_list[:200], ret_results=ret_results, dataset=dataset)
    # # loader.eval_sp(preds=perturb_oracle_keys, golds=[oracle['oracle_docs'] for oracle in oracle_list[:200]], top_k=[2])
    # # golds = [oracle['oracle_docs'] for oracle in oracle_list]
    # # preds = perturb_oracle_keys
    # # recall_n = 0
    # # for gold, pred in zip(golds, preds):
    # #     cur_hit = sum([x in pred for x in gold])
    # #     recall_n += cur_hit / len(gold)
    # # recall_n /= len(preds)
    # # print(recall_n)
    # wrong_count = 0
    # for i, doc_keys in enumerate(perturb_oracle_keys):
    #     doc = docs[i][0]
    #     if not loader.if_has_answer(doc=doc, qs_id=oracle_list[i]['qs_id']):
    #         wrong_count += 1
    #         # print(f'wrong oracle for doc {qs_list[i]}')
    # acc = 1 - (wrong_count / len(perturb_oracle_keys))
    # print(acc)

    """
    try select by simi socre
    """
    # dataset = 'hotpotQA'
    # ret_results = get_ret_results(dataset=dataset, retriever='openai-embedding')
    # # select_by_simi_score(ret_results=ret_results, doc_selection_type='simi_score_0.05')
    # # select_by_simi_score(ret_results=ret_results, doc_selection_type='simi_score_0.1')
    # # select_by_simi_score(ret_results=ret_results, doc_selection_type='simi_score_0.2')
    # # select_by_simi_score(ret_results=ret_results, doc_selection_type='simi_score_0.3')
    # ret_doc_keys_list = select_by_simi_score(ret_results=ret_results, doc_selection_type='simi_score_0.3')
    # loader = HotpotQAUtils()
    # oracle_list = loader.load_oracle_list()
    # scores = loader.eval_sp(preds=ret_doc_keys_list, golds=[oracle['oracle_docs'] for oracle in oracle_list], top_k=[5])
    """         0.05    0.1     0.2     0.3     0.5
    conala:     1.5     2.1     3.9     6.6     17.9
    DS1000:     1.4     2       3.5     6.2     17.4
    pde:        1.4     1.9     3.7     6.8     19.1
    NQ:         1.4     1.9     3.4     6       16.3
    Tri:        1.3     1.8     3.1     5.3     14.7
    hotpot:     1.2     1.4     2.2     3.4     8.6
    """

    """
    count python avg doc length
    """
    # python_docs = [item['doc'] for item in PythonDocsLoader().load_api_docs()]
    # doc_lengths = get_docs_tokens(python_docs, model='gpt-3.5-turbo')
    # avg_length = 0
    # for length in doc_lengths:
    #     # avg_length += length if length <= 1000 else 1000
    #     avg_length += length
    # avg_length = avg_length / len(doc_lengths)
    # print(avg_length)
