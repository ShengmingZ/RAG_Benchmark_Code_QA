# bm25_args = sparse_retriever_config('--dataset tldr')
# tldr_qs = json.load(open(bm25_args.tldr_qs_file, 'r'))
# oracle_results = json.load(open(bm25_args.tldr_oracle, 'r'))
# tldr_doc_whole = json.load(open(bm25_args.tldr_doc_whole, 'r'))
# tldr_doc_line = json.load(open(bm25_args.tldr_doc_line, 'r'))
#
# gene_results = list()
# prompts = list()
# for qs_idx, item in enumerate(tldr_qs):
#     qs = item['nl']
#     if item['cmd_name'] not in tldr_doc_whole.keys():
#         print(item['cmd_name'])

# tldr_args = tldr_config()
# qs_file = json.load(open(tldr_args.qs_file, 'r'))
# oracle = json.load(open(tldr_args.oracle, 'r'))
# tldr_doc_line = json.load(open(tldr_args.doc_line, 'r'))
#
# cmd = None
# cmd_line_count = 999
# for key in tldr_doc_line.keys():
#     if cmd != key.split('_')[0]:
#         if cmd_line_count <= 5: print(cmd)
#         cmd = key.split('_')[0]
#         cmd_line_count = 1
#     else:
#         cmd_line_count += 1


# tldr_args = tldr_config()
# tldr_key_list_whole = list(json.load(open(tldr_args.doc_whole, 'r')).keys())
# tldr_key_list_line = list(json.load(open(tldr_args.doc_line, 'r')).keys())
#
# for idx, key in enumerate(tldr_key_list_line):
#     cmd = key.rsplit('_',1)[0]
#     if cmd == 'gh-config': print(tldr_key_list_line[idx])
# print(tldr_key_list_line[-1])

# from dataset_utils import conala_config
#
# ret_save_file = 'docprompting_data/conala/ret_results_docprompting-codet5-python-doc-retriever.json'
# conala_args = conala_config()
# d = json.load(open(conala_args.oracle, 'r'))
# gold = [item['oracle_man'] for item in d]
# print(gold[0])
# r_d = json.load(open(ret_save_file, 'r'))
# # pred = [r_d[x['question_id']]['retrieved'] for x in d]
# pred = []
# for result in r_d:
#     pred.append([item['lib_key'] for item in result])
# print(pred[0])
# top_k = [1, 3, 5, 8, 10, 12, 15, 20, 30, 50, 100, 200]

# auth_src_embed_file = 'docprompting_data/conala/.tmp/src_embedding.npy'
# src_embed_file = 'docprompting_data/conala/.tmp/src_embedding_docprompting-codet5-python-doc-retriever.npy'
# import numpy as np
# auth_src_embed = np.load(auth_src_embed_file)
# src_embed = np.load(src_embed_file)

# from collections import Counter
# tldr_args = tldr_config()
# questions = json.load(open(tldr_args.qs_file, 'r'))
# qs_id_list = [qs['question_id'] for qs in questions]
#
# element_counts = Counter(qs_id_list)
#
# # Print the result
# for element, count in element_counts.items():
#     if count > 1:
#         print(f"{element}: {count} times")

# from elasticsearch import Elasticsearch
# from sparse_retriever import conala_BM25, sparse_retriever_config
# from dataset_utils import conala_config
#
# question = 'Create list `instancelist` containing 29 objects of type MyClass'
# query = {'query':
#              {'match':
#                   {'doc': question}},
#          'size': 200}
# # query = {'query': {'match': {'doc': 'Create list `instancelist` containing 29 objects of type MyClass'}}, 'size': 200}
# es = Elasticsearch("http://localhost:9200")
# res = es.search(index='conala', body=query)['hits']['hits']
#
# sparse_retriever_args = sparse_retriever_config('--dataset conala')
# conala_args = conala_config()
# conala_retriever = conala_BM25(sparse_retriever_args, conala_args)
# conala_retriever.bm25_retrieve(query, 'conala')

# from elasticsearch import Elasticsearch
#
# es = Elasticsearch("http://localhost:9200")
# print(es.info().body)
# query = {'query': {'match': {'doc': 'Create list `instancelist` containing 29 objects of type MyClass'}}, 'size': 200}
# res = es.search(index='conala', body=query)['hits']['hits']
# print(len(res))

from dataset.dataset_configs import TldrLoader
from collections import Counter

tldr_loader = TldrLoader()
oracle_list = tldr_loader.load_oracle_list('test')
qs_list = tldr_loader.load_qs_list('test')
counter = Counter(d['qs_id'] for d in qs_list)
print(counter)


print(len(oracle_list), len(qs_list))
for idx, oracle in enumerate(oracle_list):
    if oracle['qs_id'] != qs_list[idx]['qs_id']:
        print(oracle)
        print(qs_list[idx])
        break