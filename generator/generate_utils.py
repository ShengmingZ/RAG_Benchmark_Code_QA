import json
import shlex
import argparse
import platform
import sys, os
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from functools import partial

from prompt.hotpotqa_prompt import original_prompt
from copy import deepcopy
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from prompt import conala_prompt
from retriever.retriever_utils import retriever_config, get_ret_results
from dataset_utils.conala_utils import ConalaLoader
from dataset_utils.DS1000_utils import DS1000Loader
from dataset_utils.hotpotQA_utils import HotpotQAUtils
from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
from dataset_utils.corpus_utils import PythonDocsLoader, WikiCorpusLoader

random.seed(0)

AVG_PROMPT_LENGTH_HOTPOT = ...
AVG_PROMPT_LENGTH_NQ = ...
AVG_PROMPT_LENGTH_CONALA = ...
AVG_PROMPT_LENGTH_DS1000 = ...
AVG_PROMPT_LENGTH_PANDASEVAL = ...


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


def approximate_token(prompts, model):
    if model.startswith('gpt'):
        import tiktoken
        max_tokens, avg_tokens = 0, 0
        encoding = tiktoken.encoding_for_model(model)
        for prompt in prompts:
            tokens = len(encoding.encode(prompt))
            avg_tokens += tokens
            if tokens > max_tokens: max_tokens = tokens
        avg_tokens = avg_tokens / len(prompts)
        print(f"Average tokens: {avg_tokens:.3f}")
        return avg_tokens
    elif model.startswith('llama'):
        if model == 'llama2-13b-chat':
            model = 'meta-llama/Llama-2-13b-chat-hf'
        elif model == 'codellama-13b-instruct':
            model = 'codellama/CodeLlama-13b-Instruct-hf'
        elif model == 'llama3-8b':
            model = 'meta-llama/Meta-Llama-3-8B'
        access_token = "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"
        tokenizer = AutoTokenizer.from_pretrained(model, torch_dtype=torch.float16, token=access_token)
        avg_tokens = 0
        for prompt in prompts:
            tokens = len(tokenizer(prompt, return_tensors='pt')['input_ids'][0])
            avg_tokens += tokens
        avg_tokens = avg_tokens / len(prompts)
        print(f"Average tokens: {avg_tokens:.3f}")
        return avg_tokens



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
    elif model.startswith('llama'):
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
            tokens = tokenizer.encode(doc, max_length=max_length, truncation=True, add_special_tokens=False)
            doc = tokenizer.decode(tokens[:max_length])
            truncated_docs.append(doc)

    return truncated_docs



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
    parser.add_argument('--max_tokens', type=int, default=100)

    parser.add_argument('--retriever', type=str, default='best', choices=['best', 'BM25', 'contriever', 'miniLM', 'openai-embedding'])

    parser.add_argument('--analysis_type', type=str, choices=['retrieval_recall', 'retrieval_doc_type'])
    # each of the following parameter corresponds to one analysis, when choose one, the default value of the other parameters are the default value of RAG
    parser.add_argument('--ret_acc', type=float, default=1)     # top_k:len(oracle_docs), prompt_type:3shots, ret_doc_type:oracle/distracting
    parser.add_argument('--ret_doc_type', type=str, default='retrieved', choices=['oracle', 'retrieved', 'distracting', 'random', 'irrelevant', 'none'])
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--doc_max_length', type=int, default=1000)
    parser.add_argument('--prompt_type', type=str, default='3shots', choices=['3shots', '0shot', 'instruct', 'CoT'])

    args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))

    # construct save file
    if args.save_file is None:
        args.save_file = f'data/{args.dataset}/results/model_{args.model}_temperature_{args.temperature}_n_{args.n}_{args.analysis_type}_'
        if args.analysis_type == 'retrieval_recall':
            args.save_file += f'{args.ret_acc}.json'
        elif args.analysis_type == 'retrieval_doc_type':
            args.save_file += f'{args.ret_doc_type}.json'
        args.save_file = os.path.join(root_path, args.save_file)

    print(json.dumps(vars(args), indent=2))
    return args


def get_distracting_doc(qs_id, oracle_docs, ret_results, dataset):
    ret_result = ret_results[qs_id]
    for item in ret_result:
        doc_key = item['doc_key']
        if dataset == 'NQ' or dataset == 'TriviaQA':
            doc = WikiCorpusLoader().get_docs(doc_keys_list=[[doc_key]], dataset=dataset, num_procs=1)[0][0]
            if not NQTriviaQAUtils(dataset).if_has_answer(doc=doc, qs_id=qs_id):
                return doc_key
        else:
            if doc_key not in oracle_docs:
                return doc_key


def control_ret_acc(ret_acc, oracle_list, ret_results, dataset):
    """
    generate retrieval doc key of each sample based on ret_acc, perturb the doc key until it reaches the new ret_acc value
    :param ret_acc:
    :param oracle_list: a list of list that store oracle doc key for each sample
    :param dataset:
    :return:
    """
    # perturb oracle_docs_list with high score related docs until it reaches the ret_acc
    oracle_docs_list = deepcopy([oracle['oracle_docs'] for oracle in oracle_list])
    ret_accs = [1] * len(oracle_docs_list)  # record acc of each sample
    cur_ret_acc = sum(ret_accs) / len(ret_accs) # total acc
    perturb_placeholder = list()    # this placeholder is to store doc keys that are oracle
    for i, oracle in enumerate(oracle_docs_list):
        for j in range(len(oracle)):
            perturb_placeholder.append([i, j])

    while cur_ret_acc > ret_acc:
        perturb_idx = random.sample(perturb_placeholder, 1)[0] # pick an oracle key and perturb
        perturb_placeholder.remove(perturb_idx)
        oracle_docs_list[perturb_idx[0]][perturb_idx[1]] = get_distracting_doc(oracle_list[perturb_idx[0]]['qs_id'], oracle_docs_list[perturb_idx[0]], ret_results, dataset)
        ret_accs[perturb_idx[0]] = (ret_accs[perturb_idx[0]] * len(oracle_docs_list[perturb_idx[0]]) - 1) / len(oracle_docs_list[perturb_idx[0]])
        cur_ret_acc = sum(ret_accs) / len(ret_accs)

    if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
        docs = WikiCorpusLoader().get_docs(oracle_docs_list, dataset)
    else:
        docs = [PythonDocsLoader().get_docs(oracle_docs) for oracle_docs in oracle_docs_list]

    return oracle_docs_list, docs


def perturb_ret_doc_type(perturb_doc_type, ret_doc_key_list, oracle_doc_key_list):
    """
    generate retrieval doc key of each sample based on ret_doc_type, return a list of the docs for each sample
    :param ret_doc_type:
    :return:
    """
    assert perturb_doc_type in ['oracle', 'retrieved', 'distracting', 'random', 'irrelevant_dummy', 'irrelevant_diff', 'none']

    if perturb_doc_type in ['irrelevant_diff', 'irrelevant_dummy']:
        docs = []
        for oracle_doc_key in oracle_doc_key_list:
            irrelevant_docs = get_irrelevant_doc(irrelevant_type=perturb_doc_type.split('_')[1], doc_length=doc_length, n=len(oracle_doc_key))
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
            # wiki_loader = WikiCorpusLoader()
            # for doc_key in doc_key_list:
            #     docs.append(wiki_loader.get_docs(doc_key))
            docs = WikiCorpusLoader().get_docs(doc_key_list)
        elif dataset_type == 'se':
            ...

        return doc_key_list, docs




def process_retrieval_doc():
    ...


def generate_prompts(questions, ret_docs_list, prompt_type, dataset, model_name, doc_max_length):
    _ret_docs_list = list()
    for docs in ret_docs_list:
        _ret_docs_list.append(truncate_docs(docs, model_name, doc_max_length))
    ret_docs_list = _ret_docs_list

    if dataset == 'NQ':
        if prompt_type == '3shots':
            generate_func = ...

    elif dataset == 'conala':
        if prompt_type == '3shots':
            if model_name.startswith('llama'):
                generate_func = partial(conala_prompt.llama_3shot_prompt, model=model_name)
            elif model_name.startswith('gpt'):
                generate_func = conala_prompt.gpt_3shots_prompt

    prompts = []
    for ret_docs, question in zip(ret_docs_list, questions):
        prompts.append(generate_func(ret_docs, question))
    print(prompts[0])
    approximate_token(prompts, model_name)

    return prompts





if __name__ == "__main__":
    # test for control_ret_acc
    # hotpotqa_loader = HotpotQALoader()
    # oracle_list = hotpotqa_loader.load_oracle_list()
    # oracle_list = [oracle['oracle_docs'] for oracle in oracle_list]
    # wiki_loader = WikiCorpusLoader()
    # wiki_id_list = wiki_loader.load_wiki_id()
    # perturb_oracle_list = control_ret_acc(0.8, oracle_list, wiki_id_list)
    # ret_acc = 0
    # for perturb_oracle, oracle in zip(perturb_oracle_list, oracle_list):
    #     count = sum(1 for x, y in zip(perturb_oracle, oracle) if x == y)
    #     ret_acc = ret_acc + count/len(perturb_oracle)
    # ret_acc = ret_acc/len(oracle_list)
    # print(ret_acc)

    # test control ret_acc
    loader = ConalaLoader()
    oracle_list = loader.load_oracle_list()
    ret_results = get_ret_results(dataset='conala', retriever='BM25')
    # print([oracle['oracle_docs'] for oracle in oracle_list])
    perturb_oracle_keys = control_ret_acc(ret_acc=0.8, oracle_list=oracle_list, ret_results=ret_results, dataset='conala')

    golds = [oracle['oracle_docs'] for oracle in oracle_list]
    preds = perturb_oracle_keys
    print(golds)
    print(preds)
    recall_n = 0
    for gold, pred in zip(golds, preds):
        cur_hit = sum([x in pred for x in gold])
        recall_n += cur_hit / len(gold)
    recall_n /= len(preds)
    print(recall_n)

    # doc = """
    # concat(objs: 'Iterable[NDFrame] | Mapping[Hashable, NDFrame]', axis=0, join='outer', ignore_index: 'bool' = False, keys=None, levels=None, names=None, verify_integrity: 'bool' = False, sort: 'bool' = False, copy: 'bool' = True)     Concatenate pandas objects along a particular axis with optional set logic     along the other axes.          Can also add a layer of hierarchical indexing on the concatenation axis,     which may be useful if the labels are the same (or overlapping) on     the passed axis number.          Parameters     ----------     objs : a sequence or mapping of Series or DataFrame objects         If a mapping is passed, the sorted keys will be used as the `keys`         argument, unless it is passed, in which case the values will be         selected (see below). Any None objects will be dropped silently unless         they are all None in which case a ValueError will be raised.     axis : {0/'index', 1/'columns'}, default 0         The axis to concatenate along.     join : {'inner', 'outer'}, default 'outer'         How to handle indexes on other axis (or axes).     ignore_index : bool, default False         If True, do not use the index values along the concatenation axis. The         resulting axis will be labeled 0, ..., n - 1. This is useful if you are         concatenating objects where the concatenation axis does not have         meaningful indexing information. Note the index values on the other         axes are still respected in the join.     keys : sequence, default None         If multiple levels passed, should contain tuples. Construct         hierarchical index using the passed keys as the outermost level.     levels : list of sequences, default None         Specific levels (unique values) to use for constructing a         MultiIndex. Otherwise they will be inferred from the keys.     names : list, default None         Names for the levels in the resulting hierarchical index.     verify_integrity : bool, default False         Check whether the new concatenated axis contains duplicates. This can         be very expensive relative to the actual data concatenation.     sort : bool, default False         Sort non-concatenation axis if it is not already aligned when `join`         is 'outer'.         This has no effect when ``join='inner'``, which already preserves         the order of the non-concatenation axis.              .. versionchanged:: 1.0.0                 Changed to not sort by default.          copy : bool, default True         If False, do not copy data unnecessarily.          Returns     -------     object, type of objs         When concatenating all ``Series`` along the index (axis=0), a         ``Series`` is returned. When ``objs`` contains at least one         ``DataFrame``, a ``DataFrame`` is returned. When concatenating along         the columns (axis=1), a ``DataFrame`` is returned.          See Also     --------     Series.append : Concatenate Series.     DataFrame.append : Concatenate DataFrames.     DataFrame.join : Join DataFrames using indexes.     DataFrame.merge : Merge DataFrames by indexes or columns.          Notes     -----     The keys, levels, and names arguments are all optional.          A walkthrough of how this method fits in with other tools for combining     pandas objects can be found `here     <https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html>`__.          Examples     --------     Combine two ``Series``.          >>> s1 = pd.Series(['a', 'b'])     >>> s2 = pd.Series(['c', 'd'])     >>> pd.concat([s1, s2])     0    a     1    b     0    c     1    d     dtype: object          Clear the existing index and reset it in the result     by setting the ``ignore_index`` option to ``True``.          >>> pd.concat([s1, s2], ignore_index=True)     0    a     1    b     2    c     3    d     dtype: object          Add a hierarchical index at the outermost level of     the data with the ``keys`` option.          >>> pd.concat([s1, s2], keys=['s1', 's2'])     s1  0    a         1    b     s2  0    c         1    d     dtype: object          Label the index keys you create with the ``names`` option.          >>> pd.concat([s1, s2], keys=['s1', 's2'],     ...           names=['Series name', 'Row ID'])     Series name  Row ID     s1           0         a                  1         b     s2           0         c                  1         d     dtype: object          Combine two ``DataFrame`` objects with identical columns.          >>> df1 = pd.DataFrame([['a', 1], ['b', 2]],     ...                    columns=['letter', 'number'])     >>> df1       letter  number     0      a       1     1      b       2     >>> df2 = pd.DataFrame([['c', 3], ['d', 4]],     ...                    columns=['letter', 'number'])     >>> df2       letter  number     0      c       3     1      d       4     >>> pd.concat([df1, df2])       letter  number     0      a       1     1      b       2     0      c       3     1      d       4          Combine ``DataFrame`` objects with overlapping columns     and return everything. Columns outside the intersection will     be filled with ``NaN`` values.          >>> df3 = pd.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']],     ...                    columns=['letter', 'number', 'animal'])     >>> df3       letter  number animal     0      c       3    cat     1      d       4    dog     >>> pd.concat([df1, df3], sort=False)       letter  number animal     0      a       1    NaN     1      b       2    NaN     0      c       3    cat     1      d       4    dog          Combine ``DataFrame`` objects with overlapping columns     and return only those that are shared by passing ``inner`` to     the ``join`` keyword argument.          >>> pd.concat([df1, df3], join="inner")       letter  number     0      a       1     1      b       2     0      c       3     1      d       4          Combine ``DataFrame`` objects horizontally along the x axis by     passing in ``axis=1``.          >>> df4 = pd.DataFrame([['bird', 'polly'], ['monkey', 'george']],     ...                    columns=['animal', 'name'])     >>> pd.concat([df1, df4], axis=1)       letter  number  animal    name     0      a       1    bird   polly     1      b       2  monkey  george          Prevent the result from including duplicate index values with the     ``verify_integrity`` option.          >>> df5 = pd.DataFrame([1], index=['a'])     >>> df5        0     a  1     >>> df6 = pd.DataFrame([2], index=['a'])     >>> df6        0     a  2     >>> pd.concat([df5, df6], verify_integrity=True)     Traceback (most recent call last):         ...     ValueError: Indexes have overlapping values: ['a']
    # """
    # print(doc)
    # doc = truncate_docs([doc], model='llama3-8b')[0]
    # print(doc)

