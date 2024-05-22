import platform
import sys, os
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
import argparse
import shlex
import json
from collections import OrderedDict

import numpy as np
import faiss
from dataset_utils.dataset_configs import TldrLoader, ConalaLoader, WikiCorpusLoader, HotpotQALoader, NQLoader
from retriever.retriaval_evaluate import conala_eval, tldr_eval
from retriever.dense_encoder import DenseRetrievalEncoder
from dataset_utils.hotpot_evaluate_v1 import eval_sp


model_name_dict = {'codet5_conala': 'neulab/docprompting-codet5-python-doc-retriever',
                   'codet5_ots': 'Salesforce/codet5-base',
                   'roberta_conala': '',
                   'roberta_ots': 'roberta-large-mnli',
                   'miniLM': 'sentence-transformers/all-MiniLM-L6-v2',
                   'openai-embedding': 'text-embedding-3-small',
                   'contriever': 'facebook/contriever'}


def retrieve(qs_embed, doc_embed, qs_id_list, doc_key_list, save_file, top_k):
    """
    compute similarity using FAISS
    :param qs_embed: a numpy array consists of qs embedding
    :param doc_embed: a numpy array consists of doc embedding
    :param qs_id_list: a list of qs id, in accordance with qs_embed
    :param doc_key_list: a list of doc id, in accordance with doc_embed
    :param save_file: save retrieval result
    :param top_k:
    :return:
    """
    # retrieve
    assert qs_embed.shape[0] == len(qs_id_list)
    assert doc_embed.shape[0] == len(doc_key_list)
    indexer = faiss.IndexFlatIP(doc_embed.shape[1])
    indexer.add(doc_embed)
    D, I = indexer.search(qs_embed, top_k)

    # process and save results
    ret_results = dict()
    for qs_idx, (dist, retrieved_index) in enumerate(zip(D, I)):
        retrieved_doc_key = [doc_key_list[x] for x in retrieved_index]
        results = list()
        for key, distance in zip(retrieved_doc_key, dist):
            results.append(dict(doc_key=key, score=float(distance)))
        ret_results[qs_id_list[qs_idx]] = results
    # print(save_file)
    # print(ret_results[qs_id_list[0]])
    with open(save_file, 'w+') as f:
        json.dump(ret_results, f, indent=2)

    return ret_results


def conala_retrieve(args):
    conala_loader = ConalaLoader()
    qs_list = conala_loader.load_qs_list(args.dataset_type)
    qs_id_list = [qs['qs_id'] for qs in qs_list]
    doc_list_firstpara = conala_loader.load_doc_list_firstpara()
    doc_key_list = list(doc_list_firstpara.keys())

    source_embed = np.load(args.conala_qs_embed_save_file + '.npy')
    target_embed = np.load(args.conala_doc_firstpara_embed_save_file + '.npy')
    # source_embed = np.load('docprompting_data/conala/.tmp/src_embedding.npy')
    # target_embed = np.load('docprompting_data/conala/.tmp/tgt_embedding.npy')

    ret_results = retrieve(source_embed, target_embed, qs_id_list, doc_key_list, args.save_file, args.top_k)

    oracle_list = conala_loader.load_oracle_list(args.dataset_type)
    gold, pred = list(), list()
    for item in oracle_list:
        gold.append(item['doc_keys'])
        pred.append([tmp['doc_key'] for tmp in ret_results[item['qs_id']]])
    conala_eval(gold=gold, pred=pred)


# def tldr_whole_retrieve(args):
#     tldr_loader = TldrLoader()
#     doc_list_whole = tldr_loader.load_doc_list_whole()
#     key_list_whole = list(doc_list_whole.keys())
#     qs_list = tldr_loader.load_qs_list(args.dataset_type)
#     qs_id_list = [qs['qs_id'] for qs in qs_list]
#
#     source_embed = np.load(args.tldr_qs_embed_save_file + '.npy')
#     doc_whole_embed = np.load(args.tldr_doc_whole_embed_save_file + '.npy')
#
#     # truncate doc and retrieve
#     save_file_whole = args.save_file.replace('.json', '_whole.json')
#     results_whole = retrieve(source_embed, doc_whole_embed, qs_id_list, key_list_whole, save_file_whole, args.top_k)
#
#     # evaluate
#     gold = [oracle['doc_keys'][0] for oracle in tldr_loader.load_oracle_list(args.dataset_type)]
#     pred = list()
#     for res_key in results_whole.keys():
#         pred.append([item['doc_key'] for item in results_whole[res_key]])
#     tldr_eval(src=gold, pred=pred, top_k=[1, 3, 5, 10, 15, 20, 30])
#
#
# def tldr_line_retrieve(args):
#     tldr_loader = TldrLoader()
#     doc_list_whole = tldr_loader.load_doc_list_whole()
#     doc_list_line = tldr_loader.load_doc_list_line()
#     key_list_whole = list(doc_list_whole.keys())
#     key_list_line = list(doc_list_line.keys())
#     # load whole results
#     save_file_whole = args.save_file.replace('.json', '_whole.json')
#     save_file_line = args.save_file.replace('.json', '_line.json')
#     doc_line_embed = np.load(args.tldr_doc_line_embed_save_file + '.npy')


# def old_tldr_line_retrieve(args):
#     tldr_loader = TldrLoader()
#     doc_list_whole = tldr_loader.load_doc_list_whole()
#     key_list_whole = list(doc_list_whole.keys())
#     doc_list_line = tldr_loader.load_doc_list_line()
#     key_list_line = list(doc_list_line.keys())
#     qs_list = tldr_loader.load_qs_list(args.dataset_type)
#     qs_id_list = [qs['qs_id'] for qs in qs_list]
#
#     qs_embed = np.load(args.tldr_qs_embed_save_file + '.npy')
#     doc_whole_embed = np.load(args.tldr_doc_whole_embed_save_file + '.npy')
#     doc_line_embed = np.load(args.tldr_doc_line_embed_save_file + '.npy')
#
#     # way 1: retrieve sentence level
#     save_file_line = args.save_file.replace('.json', '_line.json')
#     results_line = retrieve(qs_embed, doc_line_embed, qs_id_list, key_list_line, save_file_line, args.top_k)
#
#     # evaluate
#     pred, gold = [], []
#     gold = [oracle['doc_keys'][0] for oracle in tldr_loader.load_oracle_list(args.dataset_type)]
#     for item in qs_list:
#         pred_result = results_line[item['qs_id']]
#         pred_key_list = [item['doc_key'].rsplit('_', 1)[0] for item in pred_result]
#         pred.append(list(OrderedDict.fromkeys(pred_key_list)))
#     tldr_eval(gold, pred, top_k=[1, 3, 5, 10, 15, 20, 30])
#
#     # way 2: calc mean of each sentence and then retrieve doc level
#     assert len(key_list_line) == doc_line_embed.shape[0]
#     doc_whole_mean_embed = np.zeros(doc_whole_embed.shape)
#     cmd, cmd_idx, cmd_length = key_list_line[0].rsplit('_', 1)[0], 0, 0
#     for idx, key in enumerate(key_list_line):
#         embed = doc_line_embed[idx]
#         if key.rsplit('_', 1)[0] != cmd:
#             doc_whole_mean_embed[cmd_idx] = doc_whole_mean_embed[cmd_idx] / cmd_length
#             cmd_idx += 1
#             cmd = key.rsplit('_', 1)[0]
#             cmd_length = 0
#         cmd_length += 1
#         doc_whole_mean_embed[cmd_idx] += embed
#
#     save_file_line = args.save_file.replace('.json', '_line.json')
#     results_line_sum = retrieve(qs_embed, doc_whole_mean_embed.astype(np.float32), qs_id_list, key_list_whole, save_file_line, args.top_k)
#
#     # evaluate
#     gold = [oracle['doc_keys'][0] for oracle in tldr_loader.load_oracle_list(args.dataset_type)]
#     pred = [results_line_sum[qs_id] for qs_id in qs_id_list]
#     tldr_eval(gold, pred, top_k=[1, 3, 5, 10, 15, 20, 30])


def embed_corpus(args):
    encoder = DenseRetrievalEncoder(args)
    if args.corpus == 'wiki':
        wiki_loader = WikiCorpusLoader()
        wiki_data = wiki_loader.load_wiki_corpus()
        wiki_data = [item['text'] for item in wiki_data]
        encoder.encode(dataset=wiki_data, save_file=args.wikipedia_docs_embed_save_file)
    elif args.corpus == 'python_docs':
        ...


def normalize_embed(embed_file):
    def normalize(vectors):
        # normalize an array of vectors
        nor_vectors = list()
        for vector in vectors:
            # nor_vector = (vector - np.mean(vector)) / np.std(vector)
            nor_vector = vector / np.linalg.norm(vector)
            nor_vectors.append(nor_vector)
        nor_vectors = np.stack(nor_vectors)
        assert nor_vectors.shape == vectors.shape
        return nor_vectors

    embed = np.load(embed_file + '.npy')
    nor_embed = normalize(embed)
    save_file = embed_file + '_normalized'
    np.save(save_file, nor_embed)


def nlp_retrieve(args):
    # encode
    if args.dataset == 'hotpotQA':
        hotpotqa_loader = HotpotQALoader()
        qs_list = hotpotqa_loader.load_qs_list()
        embed_save_file = args.hotpotqa_qs_embed_save_file
    elif args.dataset == 'NQ':
        nq_loader = NQLoader()
        qs_list = nq_loader.load_qs_list()
        embed_save_file = args.NQ_qs_embed_save_file
    if not os.path.exists(embed_save_file):
        encoder = DenseRetrievalEncoder(args)
        encoder.encode(dataset=[qs['question'] for qs in qs_list], save_file=embed_save_file)
    if args.normalize_embed is True:
        normalize_embed(embed_save_file)
        qs_embed = np.load(args.hotpotQA_qs_embed_save_file + '_normalized' + '.npy')
        args.result_file = args.result_file.replace('.json', '_normalized.json')
    else:
        qs_embed = np.load(args.hotpotQA_embed_save_file + '.npy')

    # load vectors
    doc_embed = np.load(args.wikipedia_docs_embed_save_file + '.npy')
    qs_id_list = [qs['qs_id'] for qs in qs_list]
    wiki_loader = WikiCorpusLoader()
    wiki_id_list = wiki_loader.load_wiki_id()

    # retrieve
    retrieve(qs_embed, doc_embed, qs_id_list, wiki_id_list, args.result_file, args.top_k)


def hotpotqa_eval(args):
    hotpotqa_loader = HotpotQALoader()
    oracle_list = hotpotqa_loader.load_oracle_list()
    ret_results = json.load(open(args.result_file, 'r'))
    gold, pred = list(), list()
    for item in oracle_list:
        gold.append(item['oracle_docs'])
        pred.append([tmp['doc_key'] for tmp in ret_results[item['qs_id']]])
    metrics = eval_sp(preds=pred, golds=gold, top_k=[1,3,5,10,20,50,100])
    print(metrics)


# def dense_retriever_config(in_program_call=None):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, choices=['conala', 'hotpotQA', 'NQ', 'TriviaQA'])
#     parser.add_argument('--corpus', type=str, choices=['wiki', 'python_docs'])
#     # parser.add_argument('--dataset_type', type=str, default='test', choices=['train', 'test', 'dev'])
#     parser.add_argument('--model_name', type=str, choices=['miniLM', 'openai-embedding', 'contriever'])
#     parser.add_argument('--batch_size', type=int, default=128)
#     parser.add_argument('--top_k', type=int, default=200)
#     parser.add_argument('--sim_func', default='cls_distance.cosine', choices=('cls_distance.cosine', 'cls_distance.l2', 'bertscore'))
#     parser.add_argument('--normalize_embed', action='store_true')
#     parser.add_argument('--result_file', default=None)
#
#     args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))
#     if args.result_file is None:
#         args.result_file = os.path.join(root_path, f'data/{args.dataset}/ret_results_{args.model_name}.json')
#     # record qs and corpus embeddings
#     args.conala_qs_embed_save_file = os.path.join(root_path, f'data/conala/embedding/qs_{args.model_name}')
#     args.conala_doc_firstpara_embed_save_file = os.path.join(root_path, f'data/conala/embedding/doc_firstpara_{args.model_name}')
#     # elif args.dataset == 'tldr':
#     #     args.tldr_qs_embed_save_file = os.path.join(root_path, f'docprompting_data/tldr/embedding/qs_{model_name_splitted}')
#     #     args.tldr_doc_whole_embed_save_file = os.path.join(root_path, 'docprompting_data/tldr/embedding/doc_whole_{model_name_splitted}')
#     #     args.tldr_doc_line_embed_save_file = os.path.join(root_path, f'docprompting_data/tldr/embedding/doc_line_{model_name_splitted}')
#     args.hotpotQA_qs_embed_save_file = os.path.join(root_path, f'data/hotpotQA/qs_embed_{args.model_name}')
#     args.NQ_qs_embed_save_file = os.path.join(root_path, f'data/NQ/qs_embed_{args.model_name}')
#     args.python_docs_embed_save_file = os.path.join(root_path, f'data/python_docs/embed_{args.model_name}')
#     args.wikipedia_docs_embed_save_file = f'/data/zhaoshengming/wikipedia/embed_{args.model_name}'
#     args.model_name = model_name_dict[args.model_name]
#
#     print(json.dumps(vars(args), indent=2))
#     return args



if __name__ == '__main__':
    # normalize or not
    in_program_call = f"--dataset hotpotQA \
                        --corpus wiki \
                        --model_name contriever \
                        --sim_func cls_distance.cosine"
    ret_args = dense_retriever_config(in_program_call)

    # encode and normalize corpus
    if not os.path.exists(ret_args.wikipedia_docs_embed_save_file + '.npy'):
        embed_corpus(ret_args)
    if ret_args.normalize_embed is True:
        if not os.path.exists(ret_args.wikipedia_docs_embed_save_file + '_normalized' + '.npy'):
            normalize_embed(ret_args.wikipedia_docs_embed_save_file)
        ret_args.wikipedia_docs_embed_save_file += '_normalized'

    # encode qs and retrieve
    nlp_retrieve(ret_args)
    if ret_args.dataset == 'hotpotQA':
        hotpotqa_eval(ret_args)


    # if ret_args.dataset == 'tldr':
    #     tldr_loader = TldrLoader()
    #     # encode qs
    #     # qs_list = tldr_loader.load_qs_list(ret_args.dataset_type)
    #     # qs_list = [item['nl'] for item in qs_list]
    #     # encoder.encode(dataset=qs_list, save_file=ret_args.tldr_qs_embed_save_file)
    #     # encode doc whole-level
    #     # doc_list_whole = list(tldr_loader.load_doc_list_whole().values())
    #     # encoder.encode(dataset=doc_list_whole, save_file=ret_args.tldr_doc_whole_embed_save_file)
    #     # encode doc line-level
    #     doc_list_line = list(tldr_loader.load_doc_list_line().values())
    #     encoder.encode(dataset=doc_list_line, save_file=ret_args.tldr_doc_line_embed_save_file)
    #
    #     # tldr_whole_retrieve(ret_args)
    #     # tldr_line_retrieve(ret_args)
    #
    #
    # elif ret_args.dataset == 'conala':
    #     conala_loader = ConalaLoader()
    #     # encode qs
    #     qs_list = conala_loader.load_qs_list(ret_args.dataset_type)
    #     qs_list = [qs['nl'] for qs in qs_list]
    #     encoder.encode(dataset=qs_list, save_file=ret_args.conala_qs_embed_save_file)
    #     # encode doc firstpara
    #     doc_list_firstpara = list(conala_loader.load_doc_list_firstpara().values())
    #     encoder.encode(dataset=doc_list_firstpara, save_file=ret_args.conala_doc_firstpara_embed_save_file)
    #
    #     conala_retrieve(ret_args)
