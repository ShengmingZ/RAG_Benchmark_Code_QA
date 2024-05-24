import platform
import subprocess
import sys, os
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
import json
import time
import argparse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk
import shlex
import csv
from retriever_utils import retriever_config, ret_eval
from dataset_utils.conala_utils import ConalaLoader
from dataset_utils.DS1000_utils import DS1000Loader
from dataset_utils.hotpotQA_utils import HotpotQAUtils
from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
from dataset_utils.corpus_utils import PythonDocsLoader, WikiCorpusLoader
from tqdm import tqdm


# class tldr_BM25:
#     def __init__(self, bm25_args):
#         self.ret_result_path_whole = bm25_args.tldr_ret_result_whole
#         self.ret_result_path_line = bm25_args.tldr_ret_result_line
#         self.top_k = bm25_args.tldr_top_k_cmd
#         self.top_k_cmd = bm25_args.tldr_top_k_line_cmd
#         self.top_k_sent = bm25_args.tldr_top_k_line_sent
#         self.es_idx_whole = bm25_args.tldr_idx_whole
#         self.es_idx_line = bm25_args.tldr_idx_line
#         self.tldr_loader = TldrLoader()
#         self.dataset_type = bm25_args.dataset_type
#         """
#             run
#             docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.7.0
#             to launch elasticsearch server
#
#         """
#         self.es = Elasticsearch("http://localhost:9200")
#         print(self.es.info().body)
#
#     def create_index(self):
#
#         def process_tldr_file(docs, index):
#             processed_docs = list()
#             for doc_key, doc in docs.items():
#                 cmd_name = '_'.join(doc_key.split("_")[:-1]) if doc_key[-1].isdigit() else doc_key
#                 processed_docs.append(dict(_index=index, doc=doc, doc_key=doc_key, cmd_name=cmd_name))
#             return processed_docs
#
#         # whole level
#         processed_docs = process_tldr_file(docs=self.tldr_loader.load_doc_list_whole(), index=self.es_idx_whole)
#         self.es.indices.delete(index=self.es_idx_whole, ignore=[400, 404])
#         self.es.indices.create(index=self.es_idx_whole)
#         bulk(self.es, processed_docs, index=self.es_idx_whole)
#
#         # sentence level
#         processed_docs = process_tldr_file(docs=self.tldr_loader.load_doc_list_line(), index=self.es_idx_line)
#         self.es.indices.delete(index=self.es_idx_line, ignore=[400, 404])
#         self.es.indices.create(index=self.es_idx_line)
#         bulk(self.es, processed_docs, index=self.es_idx_line)
#
#     def bm25_retrieve(self, query, index):
#         res = self.es.search(index=index, body=query)['hits']['hits']
#         _res = list()
#         for item in res:
#             _res.append({'doc_key': item['_source']['doc_key'], 'score': item['_score']})
#         return _res
#
#     """
#     first retrieve whole manual
#     """
#     def retrieve_whole_level(self):
#         qs_list = self.tldr_loader.load_qs_list(self.dataset_type)
#         res_list = dict()
#         for qs in qs_list:
#             query = {'query':
#                          {'match':
#                               {'doc': qs['nl']}},
#                      'size': self.top_k}
#             res_list[qs['qs_id']] = self.bm25_retrieve(query=query, index=self.es_idx_whole)
#
#         with open(self.ret_result_path_whole, 'w+') as f:
#             json.dump(res_list, f, indent=2)
#
#         # calc hit rate
#         auth_cmd_name_list = [oracle['doc_keys'][0] for oracle in self.tldr_loader.load_oracle_list(self.dataset_type)]
#         retrieved_result = list()
#         for res_key in res_list:
#             retrieved_result.append([item['doc_key'] for item in res_list[res_key]])
#         tldr_eval(auth_cmd_name_list, retrieved_result, top_k=[1, 3, 5, 10, 15, 20, 30])
#
#     """
#     further retrieve details from manual
#     """
#     def retrieve_line_level(self):
#         qs_list = self.tldr_loader.load_qs_list(self.dataset_type)
#         ret_result_whole = json.load(open(self.ret_result_path_whole, 'r'))
#         assert len(qs_list) == len(ret_result_whole)
#
#         res_list = dict()
#         for qs in qs_list:
#             whole_ret = ret_result_whole[qs['qs_id']]
#             retrieved_cmd_list = [item['doc_key'] for item in whole_ret][:self.top_k_cmd]
#             res_list_for_cmd = list()
#             for cmd in retrieved_cmd_list:
#                 query = {'query':
#                         {'bool':
#                             {'must':
#                                 [{'term': {'cmd_name.keyword': cmd}},
#                                  {'match': {'doc': qs['nl']}}]
#                             }
#                         },
#                     'size': self.top_k_sent}
#                 res_list_for_cmd.append(self.bm25_retrieve(query=query, index=self.es_idx_line))
#             res_list[qs['qs_id']] = res_list_for_cmd
#
#         with open(self.ret_result_path_line, 'w+') as f:
#             json.dump(res_list, f, indent=2)


"""
    run 
    docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.7.0
    to launch elasticsearch server
"""


class conala():
    def __init__(self, bm25_args):
        self.ret_result_path = bm25_args.conala_ret_result
        self.top_k = bm25_args.conala_top_k
        self.es_idx = bm25_args.conala_idx
        self.conala_loader = ConalaLoader()

        self.es = Elasticsearch("http://localhost:9200")
        print(self.es.info().body)

    def create_index(self):
        doc_firstpara = self.conala_dataloader.load_doc_list_firstpara()

        processed_docs = list()
        for doc_key in doc_firstpara.keys():
            processed_docs.append(dict(_index=self.es_idx, doc=doc_firstpara[doc_key], doc_key=doc_key))

        self.es.indices.delete(index=self.es_idx, ignore=[400, 404])
        self.es.indices.create(index=self.es_idx)
        bulk(self.es, processed_docs, index=self.es_idx)

    def bm25_retrieve(self, query, index):
        res = self.es.search(index=index, body=query)['hits']['hits']
        _res = list()
        for item in res:
            _res.append({'doc_key': item['_source']['doc_key'], 'score': item['_score']})
        return _res

    def retrieve(self):
        qs_list = self.conala_dataloader.load_qs_list(self.dataset_type)
        res_list = dict()
        for idx, qs in enumerate(qs_list):
            query = {'query':
                         {'match':
                              {'doc': qs['nl']}},
                     'size': self.top_k}
            res_list[qs['qs_id']] = self.bm25_retrieve(query=query, index=self.es_idx)

        with open(self.ret_result_path, 'w+') as f:
            json.dump(res_list, f, indent=2)

        # calc hit rate
        oracle_list = self.conala_dataloader.load_oracle_list(self.dataset_type)
        gold, pred = list(), list()
        for item in oracle_list:
            gold.append(item['doc_keys'])
            pred.append([tmp['doc_key'] for tmp in res_list[item['qs_id']]])
        conala_eval(gold=gold, pred=pred)


class hotpotQA_BM25:
    def __init__(self, bm25_args):
        self.ret_result_path = bm25_args.hotpotqa_ret_result
        self.top_k = bm25_args.top_k
        self.es_idx = bm25_args.hotpotqa_idx
        hotpotqa_loader = HotpotQALoader()
        self.qs_list = hotpotqa_loader.load_qs_list()
        self.oracle_list = hotpotqa_loader.load_oracle_list()
        self.es = Elasticsearch("http://localhost:9200")
        print(self.es.info().body)

    def create_index(self):
        def steam_wiki_data():
            wiki_corpus_file = os.path.join(root_path, 'data/wikipedia/psgs_w100.tsv')
            with open(wiki_corpus_file, 'r', newline='') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                for row in reader:
                    data = dict(_index=self.es_idx, doc_key=row[2] + '_' + row[0], doc=row[1])
                    yield data

        if not self.es.indices.exists(index=self.es_idx):
            self.es.indices.create(index=self.es_idx)
            stream = steam_wiki_data()
            for ok, res in streaming_bulk(self.es, actions=stream):
                if not ok:
                    print(res)

    def bm25_retrieve(self, query, index):
        res = self.es.search(index=index, body=query)['hits']['hits']
        _res = list()
        for item in res:
            _res.append({'doc_key': item['_source']['doc_key'], 'score': item['_score']})
        return _res

    def retrieve(self):
        res_dict = dict()
        for idx, qs in enumerate(self.qs_list):
            query = {'query':
                         {'match':
                              {'doc': qs['question']}},
                     'size': self.top_k}
            res_dict[qs['qs_id']] = self.bm25_retrieve(query=query, index=self.es_idx)

        with open(self.ret_result_path, 'w+') as f:
            json.dump(res_dict, f, indent=2)

        # calc hit rate
        gold, pred = list(), list()
        for item in self.oracle_list:
            gold.append(item['oracle_docs'])
            pred.append([tmp['doc_key'] for tmp in res_dict[item['qs_id']]])
        metrics = eval_sp(golds=gold, preds=pred, top_k=[1,3,5,10,20,50,100])
        print(metrics)


# def sparse_retriever_config(in_program_call=None):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, choices=['conala', 'DS1000', 'pandas-numpy-eval', 'hotpotQA', 'NQ', 'TriviaQA'])
#     # parser.add_argument('--dataset_type', type=str, default='test', choices=['test', 'train', 'dev'])
#     parser.add_argument('--top_k', type=int, default=200)
#     args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))
#     args.ret_result = f'{root_path}/data/{args.dataset}/ret_result_BM25.json'
#     args.bm25_idx = f'{args.dataset}'
#
#     return args



def create_idx_for_corpus(args):
    dataset = args.dataset
    es_idx = args.es_idx
    # if dataset in ['hotpotQA', 'NQ', 'TriviaQA']:
    #     es_image_file = os.path.join(root_path, 'data/python_docs/es_image_python_docs.tar')
    # elif dataset in ['NQ', 'TriviaQA']:
    #     es_image_file = os.path.join(root_path, 'data/wikipedia/es_image_wiki_hotpot.tar')
    # elif dataset in ['hotpotQA']:
    #     es_image_file = os.path.join(root_path, 'data/wikipedia/es_image_wiki_NQ.tar')
    # if os.path.exists(es_image_file):
    #     subprocess.check_output(f'docker load --input {es_image_file}', shell=True)
    # else:
    #     subprocess.check_output(f'docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.7.0', shell=True)
    es = Elasticsearch("http://localhost:9200")
    print(es.info().body)

    def steam_corpus_data(dataset):
        if dataset in ['hotpotQA', 'NQ', 'TriviaQA']:
            loader = WikiCorpusLoader()
            for data in loader.load_wiki_corpus_iter(dataset):
                yield dict(_index=es_idx, doc_key=data['id'], doc=data['text'])
        elif dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
            loader = PythonDocsLoader()
            python_docs = loader.load_api_docs()
            for doc in python_docs:
                # print(doc['doc'].split('\n')[1])
                yield dict(_index=es_idx, doc_key=doc['api_sign'][0], doc=doc['doc'])

    # es.indices.delete(index=es_idx, ignore=[400, 404])
    if not es.indices.exists(index=es_idx):
        es.indices.create(index=es_idx)
        stream = steam_corpus_data(dataset)
        for ok, res in streaming_bulk(es, actions=stream):
            if not ok:
                print(res)


def retrieve(args):
    if not os.path.exists(args.ret_result):
        # load
        dataset = args.dataset
        es_idx = args.es_idx
        es = Elasticsearch("http://localhost:9200")
        print(es.info().body)
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
        qs_list = loader.load_qs_list()

        # retrieve
        def bm25_retrieve(query, index):
            res = es.search(index=index, body=query)['hits']['hits']
            _res = list()
            for item in res:
                _res.append({'doc_key': item['_source']['doc_key'], 'score': item['_score']})
            return _res

        ret_results = dict()
        for idx, qs in tqdm(enumerate(qs_list), total=len(qs_list)):
            query = {'query':
                         {'match':
                              {'doc': qs['question']}},
                     'size': args.top_k}
            ret_results[qs['qs_id']] = bm25_retrieve(query=query, index=es_idx)

        with open(args.ret_result, 'w+') as f:
            json.dump(ret_results, f, indent=2)
    else:
        print(f'ret result exists for {args.ret_result}')





if __name__ == '__main__':
    # in_program_call = '--dataset NQ --retriever BM25'
    in_program_call = None
    args = retriever_config(in_program_call)
    # create_idx_for_corpus(args)
    # retrieve(args)
    ret_eval(args)
