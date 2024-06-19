import json
import argparse
import shlex
import platform
import sys, os
import unicodedata
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
        args.corpus_embed_file = f'/data/zhaoshengming/wikipedia/embed_{args.retriever}_hotpot'
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


def verify_ret_docs(args):
    from tqdm import tqdm
    assert args.dataset in ['NQ', 'TriviaQA', 'hotpotQA']
    ret_results = json.load(open(args.ret_result, 'r'))
    loader = NQTriviaQAUtils(args.dataset)
    corpus_loader = WikiCorpusLoader()
    for qs_id in tqdm(ret_results.keys()):
        ret_doc_keys = [item['doc_key'] for item in ret_results[qs_id]]
        ret_docs = corpus_loader.get_docs([ret_doc_keys], args.dataset, num_procs=1)[0]
        for idx, ret_doc in enumerate(ret_docs):
            if loader.if_has_answer(ret_doc, qs_id): ret_results[qs_id][idx]['has_answer'] = True
            else: ret_results[qs_id][idx]['has_answer'] = False

    with open(args.ret_result, 'w+') as f:
        json.dump(ret_results, f, indent=2)



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

    # top_k = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
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



if __name__ == '__main__':
    in_program_call = '--dataset NQ --retriever openai-embedding'
    args = retriever_config(in_program_call)
    verify_ret_docs(args)

    # dataset = args.dataset
    # loader = HotpotQAUtils()
    # oracle_list = loader.load_oracle_list()
    # ret_results = json.load(open(args.ret_result, 'r'))
    # gold = oracle_list[1]['oracle_docs']
    # pred = [tmp['doc_key'] for tmp in ret_results[oracle_list[1]['qs_id']]]
    # # print(gold)
    # # print(pred)
    # # loader.eval_sp(golds=[gold], preds=[pred], top_k=[1,3,5,10,20,50,100])
    #
    # docs = WikiCorpusLoader().get_docs([['Ugni myricoides']], dataset='hotpotQA', num_procs=4)
    # print(docs)

    # test get_docs
    # in_program_call = '--dataset NQ --retriever BM25'
    # args = retriever_config(in_program_call)
    # dataset = args.dataset
    # loader = NQTriviaQAUtils(dataset)
    # top_k = [1, 3, 5, 10, 20, 50, 100]
    # oracle_list = loader.load_oracle_list()
    # ret_results = json.load(open(args.ret_result, 'r'))
    # ret_doc_keys_list, answers_list = [], []
    # for oracle in oracle_list:
    #     answers_list.append(oracle['answers'])
    #     ret_doc_keys_list.append([tmp['doc_key'] for tmp in ret_results[oracle['qs_id']]][:top_k[-1]])
    # print(ret_doc_keys_list[0])

    # import time
    # start_time = time.time()
    # ret_docs_list = WikiCorpusLoader().get_docs(ret_doc_keys_list[:1], 'NQ', 1)
    # # print('duration: ', time.time() - start_time)
    # print(ret_docs_list)


    # test_doc_keys = [[0, 232], [1, 12], [2, 891]]
    # test_doc_keys.sort(key=lambda x:x[1], reverse=False)
    # import csv
    # wiki_corpus_file_NQ = os.path.join(root_path, 'data/wikipedia/psgs_w100.tsv')
    # with open(wiki_corpus_file_NQ, 'r', newline='') as tsvfile:
    #     reader = csv.reader(tsvfile, delimiter='\t')
    #     current_place = 0
    #     for item in test_doc_keys:
    #         skip = item[1] - current_place
    #         for _ in range(skip):
    #             next(reader)
    #         for row in reader:
    #             print(row)
    #             break
    #         current_place = current_place + skip + 1

    """
    test offset indexing
    """
    # import time
    # wiki_corpus_file_NQ = os.path.join(root_path, 'data/wikipedia/psgs_w100.tsv')
    #
    # start_time = time.time()
    # if os.path.exists('temp.txt'):
    #     index_offsets = []
    #     with open('temp.txt', 'r') as f:
    #         for line in f:
    #             index_offsets.append(int(line.strip()))
    # else:
    #     index_offsets = []
    #     current_offset = 0
    #     with open(wiki_corpus_file_NQ, 'r', encoding='iso-8859-1') as index_file:
    #         for line in index_file:
    #             index_offsets.append(current_offset)
    #             current_offset = current_offset + len(line) + 1
    #     with open('temp.txt', 'w+') as f:
    #         for item in index_offsets:
    #             f.write(f"{item}\n")
    # print('build offsets duration: ', time.time() - start_time)
    #
    #
    # doc_keys = ret_doc_keys_list[0]
    # doc_keys_placeholder = [[idx, int(key)] for idx, key in enumerate(doc_keys)]
    # doc_keys_placeholder.sort(key=lambda x: x[1], reverse=False)
    # docs = [None] * len(doc_keys)
    # with open(wiki_corpus_file_NQ, 'r', encoding='iso-8859-1') as f:
    #     for item in doc_keys_placeholder:
    #         f.seek(index_offsets[item[1]])
    #         row = f.read(index_offsets[item[1]+1] - index_offsets[item[1]])
    #         try:
    #             docs[item[0]] = row.split('\t')[1][1:-1]
    #         except:
    #             print(row)
    #             print(item)
    # print('byte seek duration: ', time.time() - start_time)
    # print(docs)


    # import csv
    # count = 0
    # with open(wiki_corpus_file_NQ, 'r', newline='') as tsvfile:
    #     reader = csv.reader(tsvfile, delimiter='\t')
    #     for row in reader:
    #         for idx in target_idx:
    #             if row[0] == str(idx):
    #                 print([row[1]])
    #                 break

