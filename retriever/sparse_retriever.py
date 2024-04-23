import platform
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
from dataset_utils.dataset_configs import TldrLoader, ConalaLoader, HotpotQALoader, load_wiki_corpus_iter
from retriever.retriaval_evaluate import tldr_eval, conala_eval
import shlex
import csv


class tldr_BM25:
    """
    still a
    """
    def __init__(self, bm25_args):
        self.ret_result_path_whole = bm25_args.tldr_ret_result_whole
        self.ret_result_path_line = bm25_args.tldr_ret_result_line
        self.top_k = bm25_args.tldr_top_k_cmd
        self.top_k_cmd = bm25_args.tldr_top_k_line_cmd
        self.top_k_sent = bm25_args.tldr_top_k_line_sent
        self.es_idx_whole = bm25_args.tldr_idx_whole
        self.es_idx_line = bm25_args.tldr_idx_line
        self.tldr_loader = TldrLoader()
        self.dataset_type = bm25_args.dataset_type
        """
            run 
            docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.7.0
            to launch elasticsearch server

        """
        self.es = Elasticsearch("http://localhost:9200")
        print(self.es.info().body)

    def create_index(self):

        def process_tldr_file(docs, index):
            processed_docs = list()
            for doc_key, doc in docs.items():
                cmd_name = '_'.join(doc_key.split("_")[:-1]) if doc_key[-1].isdigit() else doc_key
                processed_docs.append(dict(_index=index, doc=doc, doc_key=doc_key, cmd_name=cmd_name))
            return processed_docs

        # whole level
        processed_docs = process_tldr_file(docs=self.tldr_loader.load_doc_list_whole(), index=self.es_idx_whole)
        self.es.indices.delete(index=self.es_idx_whole, ignore=[400, 404])
        self.es.indices.create(index=self.es_idx_whole)
        bulk(self.es, processed_docs, index=self.es_idx_whole)

        # sentence level
        processed_docs = process_tldr_file(docs=self.tldr_loader.load_doc_list_line(), index=self.es_idx_line)
        self.es.indices.delete(index=self.es_idx_line, ignore=[400, 404])
        self.es.indices.create(index=self.es_idx_line)
        bulk(self.es, processed_docs, index=self.es_idx_line)

    def bm25_retrieve(self, query, index):
        res = self.es.search(index=index, body=query)['hits']['hits']
        _res = list()
        for item in res:
            _res.append({'doc_key': item['_source']['doc_key'], 'score': item['_score']})
        return _res

    """
    first retrieve whole manual
    """
    def retrieve_whole_level(self):
        qs_list = self.tldr_loader.load_qs_list(self.dataset_type)
        res_list = dict()
        for qs in qs_list:
            query = {'query':
                         {'match':
                              {'doc': qs['nl']}},
                     'size': self.top_k}
            res_list[qs['qs_id']] = self.bm25_retrieve(query=query, index=self.es_idx_whole)

        with open(self.ret_result_path_whole, 'w+') as f:
            json.dump(res_list, f, indent=2)

        # calc hit rate
        auth_cmd_name_list = [oracle['doc_keys'][0] for oracle in self.tldr_loader.load_oracle_list(self.dataset_type)]
        retrieved_result = list()
        for res_key in res_list:
            retrieved_result.append([item['doc_key'] for item in res_list[res_key]])
        tldr_eval(auth_cmd_name_list, retrieved_result, top_k=[1, 3, 5, 10, 15, 20, 30])

    """
    further retrieve details from manual 
    """
    def retrieve_line_level(self):
        qs_list = self.tldr_loader.load_qs_list(self.dataset_type)
        ret_result_whole = json.load(open(self.ret_result_path_whole, 'r'))
        assert len(qs_list) == len(ret_result_whole)

        res_list = dict()
        for qs in qs_list:
            whole_ret = ret_result_whole[qs['qs_id']]
            retrieved_cmd_list = [item['doc_key'] for item in whole_ret][:self.top_k_cmd]
            res_list_for_cmd = list()
            for cmd in retrieved_cmd_list:
                query = {'query':
                        {'bool':
                            {'must':
                                [{'term': {'cmd_name.keyword': cmd}},
                                 {'match': {'doc': qs['nl']}}]
                            }
                        },
                    'size': self.top_k_sent}
                res_list_for_cmd.append(self.bm25_retrieve(query=query, index=self.es_idx_line))
            res_list[qs['qs_id']] = res_list_for_cmd

        with open(self.ret_result_path_line, 'w+') as f:
            json.dump(res_list, f, indent=2)


class conala_BM25():
    def __init__(self, bm25_args):
        self.ret_result_path = bm25_args.conala_ret_result
        self.top_k = bm25_args.conala_top_k
        self.es_idx = bm25_args.conala_idx
        self.conala_dataloader = ConalaLoader()
        self.dataset_type = bm25_args.dataset_type
        """
            run 
            docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.7.0
            to launch elasticsearch server
        """
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
        self.es_idx = bm25_args.conala_idx
        hotpotqa_loader = HotpotQALoader()
        self.qs_list = hotpotqa_loader.load_qs_list()
        """
            run 
            docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.7.0
            to launch elasticsearch server
        """
        self.es = Elasticsearch("http://localhost:9200")
        print(self.es.info().body)

    def create_index(self):
        def steam_wiki_data():
            wiki_corpus_file = os.path.join(root_path, 'data/wikipedia/psgs_w100.tsv')
            with open(wiki_corpus_file, 'r', newline='') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                for row in reader:
                    data = dict(_id=self.es_idx, doc_key=[row[2], row[0]], doc=row[1])
                    yield data

        if not self.es.indices.exists(index=self.es_idx):
            self.es.indices.create(index=self.es_idx)
            stream = steam_wiki_data()
            for ok, res in streaming_bulk(self.es, actions=stream):
                if not ok:
                    print(res)



def sparse_retriever_config(in_program_call=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tldr', choices=['tldr', 'conala', 'ds1000'])
    parser.add_argument('--dataset_type', type=str, default='test', choices=['test', 'train', 'dev'])
    parser.add_argument('--top_k', type=int, default=200)

    parser.add_argument('--tldr_ret_result_whole', type=str, default="/Users/zhaoshengming/Code_RAG_Benchmark/docprompting_data/tldr/dev_ret_result_whole_BM25.json")
    parser.add_argument('--tldr_ret_result_line', type=str, default="/Users/zhaoshengming/Code_RAG_Benchmark/docprompting_data/tldr/dev_ret_result_line_BM25.json")
    parser.add_argument('--tldr_idx_whole', type=str, default="tldr_whole")
    parser.add_argument('--tldr_idx_line', type=str, default="tldr_line")
    parser.add_argument('--tldr_top_k_cmd', type=int, default=35)
    parser.add_argument('--tldr_top_k_line_cmd', type=int, default=5)
    parser.add_argument('--tldr_top_k_line_sent', type=int, default=30)

    parser.add_argument('--conala_ret_result', type=str, default=f"{root_path}/data/conala/ret_result_BM25.json")
    parser.add_argument('--conala_idx', type=str, default="conala")
    parser.add_argument('--conala_top_k', type=int, default=200)

    parser.add_argument('--ds1000_ret_result', type=str, default=f"{root_path}/DS-1000/ret_result_BM25.json")
    parser.add_argument('--ds1000_idx', type=str, default="ds1000")
    parser.add_argument('--ds1000_top_k', type=int, default=100)

    parser.add_argument('--hotpotqa_ret_result', type=str, default=f"{root_path}/hotpotqa/ret_result_BM25.json")
    parser.add_argument('hotpotqa_idx', type=str, default="hotpotqa")

    args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))
    return args


if __name__ == '__main__':
    in_program_call = '--dataset conala --dataset_type test'
    bm25_args = sparse_retriever_config(in_program_call)

    if bm25_args.dataset == 'tldr':
        tldr_retriever = tldr_BM25(bm25_args)
        tldr_retriever.create_index()
        time.sleep(1)
        # tldr_retriever.retrieve_whole_level()
        tldr_retriever.retrieve_line_level()

    elif bm25_args.dataset == 'conala':
        conala_retriever = conala_BM25(bm25_args)
        conala_retriever.create_index()
        time.sleep(1)
        conala_retriever.retrieve()
