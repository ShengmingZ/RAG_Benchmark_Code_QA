import json
import argparse
import shlex
import platform
import sys, os
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_utils.conala_utils import ConalaLoader
from dataset_utils.DS1000_utils import DS1000Loader
from dataset_utils.hotpotQA_utils import HotpotQAUtils
from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
from dataset_utils.corpus_utils import PythonDocsLoader, WikiCorpusLoader


BEST_RETRIEVER = {
    "hotpotQA": "",
    "NQ": "",
    "TriviaQA": "",
    "DS1000": "",
    "conala": "",
    "pandas_numpy_eval": ""
}


model_name_dict = {'codeT5': 'neulab/docprompting-codet5-python-doc-retriever',
                   'codet5_ots': 'Salesforce/codet5-base',
                   'miniLM': 'sentence-transformers/all-MiniLM-L6-v2',
                   'openai-embedding': 'text-embedding-3-small',
                   'contriever': 'facebook/contriever'}


def retriever_config(in_program_call=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['conala', 'DS1000', 'pandas_numpy_eval', 'hotpotQA', 'NQ', 'TriviaQA'])
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--retriever', type=str, choices=['BM25', 'openai-embedding', 'miniLM', 'contriever', 'codeT5'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sim_func', default='cls_distance.cosine', choices=('cls_distance.cosine', 'cls_distance.l2', 'bertscore'))
    parser.add_argument('--normalize_embed', action='store_true')
    args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))
    args.ret_result = os.path.join(root_path, f'data/{args.dataset}/ret_result_{args.retriever}.json')
    args.qs_embed_file = os.path.join(root_path, f'data/{args.dataset}/qs_embed_{args.retriever}')
    if args.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
        args.corpus = 'python_docs'
        args.corpus_embed_file = os.path.join(root_path, f'data/python_docs/embed_{args.retriever}')
    elif args.dataset in ['hotpotQA']:
        args.corpus = 'wiki_hotpot'
        args.corpus_embed_file = f'data/zhaoshengming/wikipedia/embed_{args.retriever}_hotpot'
    elif args.dataset in ['TriviaQA', 'NQ']:
        args.corpus = 'wiki_nq'
        args.corpus_embed_file = f'/data/zhaoshengming/wikipedia/embed_{args.retriever}_NQ'
    if args.retriever != 'BM25':
        args.model_name = model_name_dict[args.retriever]
    else:
        if args.dataset in ['NQ', 'TriviaQA']:
            args.es_idx = 'wiki_nq'
        elif args.dataset == 'hotpotQA':
            args.es_idx = 'wiki_hotpot'
        else:
            args.es_idx = 'python_docs'

    print(json.dumps(vars(args), indent=2))
    return args


def get_ret_results(dataset, retriever, normalize=False):
    if retriever == "best":
        retriever = BEST_RETRIEVER[dataset]
    args = retriever_config(f'--dataset {dataset} --retriever {retriever}')
    ret_result_file = args.ret_result
    if normalize:
        ret_result_file = args.result_file.replace(".json", "_normalized.json")

    return json.load(open(ret_result_file, 'r'))



def ret_eval(args):
    dataset = args.dataset
    if dataset == 'hotpotQA':
        loader = HotpotQAUtils()
    elif dataset == 'NQ' or dataset == 'TriviaQA':
        loader = NQTriviaQAUtils(dataset)
    elif dataset == 'conala':
        loader = ConalaLoader()
    elif dataset == 'DS1000':
        loader = DS1000Loader()
    elif dataset == 'pandas_numpy_eval':
        loader = PandasNumpyEvalLoader()
    ret_results = json.load(open(args.ret_result, 'r'))

    top_k = [1, 3, 5, 10, 20, 50, 100]
    if dataset == 'hotpotQA':
        oracle_list = loader.load_oracle_list()
        golds, preds = list(), list()
        for item in oracle_list:
            golds.append(item['oracle_docs'])
            preds.append([tmp['doc_key'] for tmp in ret_results[item['qs_id']]])
        metrics = loader.eval_sp(preds, golds, top_k=top_k)
    elif dataset == 'NQ' or dataset == 'TriviaQA':
        oracle_list = loader.load_oracle_list()
        ret_doc_keys_list, answers_list = [], []
        for oracle in oracle_list:
            answers_list.append(oracle['answers'])
            ret_doc_keys_list.append([tmp['doc_key'] for tmp in ret_results[oracle['qs_id']]][:top_k[-1]])
        ret_docs_list = WikiCorpusLoader().get_docs(ret_doc_keys_list, dataset)
        print('load docs done')
        hits_rate = loader.retrieval_eval(docs_list=ret_docs_list, answers_list=answers_list, top_k=top_k)
    elif dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
        oracle_list = loader.load_oracle_list()
        golds, preds = list(), list()
        for oracle in oracle_list:
            golds.append(oracle['oracle_docs'])
            preds.append([tmp['doc_key'] for tmp in ret_results[oracle['qs_id']]])
        recall_n = loader.calc_recall(src=golds, pred=preds, top_k=top_k)

