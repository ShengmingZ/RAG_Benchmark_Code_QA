import argparse
import shlex
import json, os
from collections import OrderedDict

import numpy as np
import faiss
from dataset.dataset_configs import conala_config, tldr_config
from retriaval_evaluate import conala_eval, tldr_eval
from dense_encoder import DenseRetrievalEncoder


model_name_dict = {'codet5_conala': 'neulab/docprompting-codet5-python-doc-retriever',
                   'codet5_ots': 'Salesforce/codet5-base',
                   'roberta_conala': '',
                   'roberta_ots': 'roberta-large-mnli'}


def retrieve(source_embed, target_embed, source_id_list, target_key_list, save_file, top_k):
    # retrieve
    assert source_embed.shape[0] == len(source_id_list)
    assert target_embed.shape[0] == len(target_key_list)
    indexer = faiss.IndexFlatIP(target_embed.shape[1])
    indexer.add(target_embed)
    D, I = indexer.search(source_embed, top_k)

    # process and save results
    ret_results = dict()
    for source_idx, (dist, retrieved_index) in enumerate(zip(D, I)):
        retrieved_target_key = [target_key_list[x] for x in retrieved_index]
        results = list()
        for key, distance in zip(retrieved_target_key, dist):
            results.append(dict(lib_key=key, score=float(distance)))
        ret_results[source_id_list[source_idx]] = results
    with open(save_file, 'w+') as f:
        json.dump(ret_results, f, indent=2)

    return ret_results


def conala_retrieve(args):
    conala_args = conala_config()
    conala_key_list = list()
    with open(conala_args.doc_firstpara_idx_file) as f:
        for line in f:
            conala_key_list.append(line.strip())
    conala_qs_id_list = list()
    with open(conala_args.qs_idx_file) as f:
        for line in f:
            conala_qs_id_list.append(line.strip())

    source_embed = np.load(args.conala_qs_embed_save_file + '.npy')
    target_embed = np.load(args.conala_doc_firstpara_embed_save_file + '.npy')
    # source_embed = np.load('docprompting_data/conala/.tmp/src_embedding.npy')
    # target_embed = np.load('docprompting_data/conala/.tmp/tgt_embedding.npy')

    retrieve(source_embed, target_embed, conala_qs_id_list, conala_key_list, args.save_file, args.top_k)

    conala_eval(args.save_file)


def tldr_whole_retrieve(args):
    tldr_args = tldr_config()
    key_list_whole = list(json.load(open(tldr_args.doc_whole, 'r')).keys())
    source_embed = np.load(args.tldr_qs_embed_save_file + '.npy')
    result_whole_save_file = args.save_file.replace('.json', '_whole.json')
    doc_whole_embed = np.load(args.tldr_doc_whole_embed_save_file + '.npy')
    qs_list = json.load(open(tldr_args.qs_file, 'r'))
    qs_id_list = [item['question_id'] for item in qs_list]

    # truncate doc and retrieve
    results_whole = retrieve(source_embed, doc_whole_embed, qs_id_list, key_list_whole, result_whole_save_file, args.top_k)

    # evaluate
    gold, pred = [], []
    for item in qs_list:
        gold.append(item['cmd_name'])
        pred.append(results_whole[item['question_id']])
    tldr_eval(gold, pred, top_k=[1, 3, 5, 10, 15, 20, 30])


def tldr_line_retrieve(args):
    tldr_args = tldr_config()
    key_list_whole = list(json.load(open(tldr_args.doc_whole, 'r')).keys())
    key_list_line = list(json.load(open(tldr_args.doc_line, 'r')).keys())
    source_embed = np.load(args.tldr_qs_embed_save_file + '.npy')
    result_line_save_file = args.save_file.replace('.json', '_line.json')
    doc_whole_embed = np.load(args.tldr_doc_whole_embed_save_file + '.npy')
    doc_line_embed = np.load(args.tldr_doc_line_embed_save_file + '.npy')
    qs_list = json.load(open(tldr_args.qs_file, 'r'))
    qs_id_list = [item['question_id'] for item in qs_list]

    # way 1: retrieve sentence level
    results_line = retrieve(source_embed, doc_line_embed, qs_id_list, key_list_line, result_line_save_file, args.top_k)

    # evaluate
    pred, gold = [], []
    for item in qs_list:
        gold.append(item['cmd_name'])
        pred_result = results_line[item['question_id']]
        pred_key_list = [item['lib_key'].rsplit('_', 1)[0] for item in pred_result]
        pred.append(list(OrderedDict.fromkeys(pred_key_list)))
    tldr_eval(gold, pred, top_k=[1, 3, 5, 10, 15, 20, 30])

    # way 2: get mean of each sentence and retrieve doc level
    assert len(key_list_line) == doc_line_embed.shape[0]
    doc_whole_mean_embed = np.zeros(doc_whole_embed.shape)
    cmd, cmd_idx, cmd_length = key_list_line[0].rsplit('_', 1)[0], 0, 0
    for idx, key in enumerate(key_list_line):
        embed = doc_line_embed[idx]
        if key.rsplit('_', 1)[0] != cmd:
            doc_whole_mean_embed[cmd_idx] = doc_whole_mean_embed[cmd_idx] / cmd_length
            cmd_idx += 1
            cmd = key.rsplit('_', 1)[0]
            cmd_length = 0
        cmd_length += 1
        doc_whole_mean_embed[cmd_idx] += embed
    results_line_sum = retrieve(source_embed, doc_whole_mean_embed.astype(np.float32), qs_id_list, key_list_whole, result_line_save_file, args.top_k)

    # evaluate
    gold, pred = [], []
    for item in qs_list:
        gold.append(item['cmd_name'])
        pred.append(results_line_sum[item['question_id']])
    tldr_eval(gold, pred, top_k=[1, 3, 5, 10, 15, 20, 30])


def dense_retriever_config(in_program_call=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='conala', choices=['conala', 'tldr'])
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=200)
    # parser.add_argument('--cpu', action='store_true')
    # parser.add_argument('--pooler', choices=('cls', 'cls_before_pooler'), default='cls')
    # parser.add_argument('--log_level', default='verbose')
    parser.add_argument('--sim_func', default='cls_distance.cosine', choices=('cls_distance.cosine', 'cls_distance.l2', 'bertscore'))
    # parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--normalize_embed', action='store_true')
    parser.add_argument('--save_file', default=None)

    args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))
    model_name_splitted = args.model_name.rsplit('/',1)[-1]
    if args.save_file is None:
        args.save_file = f'docprompting_data/{args.dataset}/ret_results_{model_name_splitted}.json'

    if args.dataset == 'conala':
        args.conala_qs_embed_save_file = f'docprompting_data/conala/.tmp/src_embedding_{model_name_splitted}'
        args.conala_doc_firstpara_embed_save_file = f'docprompting_data/conala/.tmp/tgt_embedding_{model_name_splitted}'
    elif args.dataset == 'tldr':
        args.tldr_qs_embed_save_file = f'docprompting_data/tldr/.tmp/src_embedding_{model_name_splitted}'
        args.tldr_doc_whole_embed_save_file = f'docprompting_data/tldr/.tmp/whole_embedding_{model_name_splitted}'
        args.tldr_doc_line_embed_save_file = f'docprompting_data/tldr/.tmp/line_embedding_{model_name_splitted}'

    print(json.dumps(vars(args), indent=2))
    return args


if __name__ == '__main__':
    model_name = model_name_dict['codet5_ots']
    in_program_call = f"--dataset conala \
                        --model_name {model_name} \
                        --sim_func cls_distance.cosine"
    ret_args = dense_retriever_config(in_program_call)
    encoder = DenseRetrievalEncoder(ret_args)

    if ret_args.dataset == 'tldr':
        tldr_args = tldr_config()

        tldr_qs_list = json.load(open(tldr_args.qs_file, 'r'))
        tldr_qs_list = [item['nl'] for item in tldr_qs_list]
        encoder.encode(dataset=tldr_qs_list, save_file=ret_args.tldr_qs_embed_save_file)

        tldr_doc_whole = json.load(open(tldr_args.doc_whole, 'r'))
        _tldr_doc_whole = list()    # skip first line in docs
        for key, doc in tldr_doc_whole.items():
            if key == 'wp': _tldr_doc_whole.append(doc)
            else: _tldr_doc_whole.append(doc.split('\n',1)[1])
        encoder.encode(dataset=_tldr_doc_whole, save_file=ret_args.tldr_doc_whole_embed_save_file)

        tldr_doc_line = list(json.load(open(tldr_args.doc_line, 'r')).values())
        # encoder.encode(dataset=tldr_doc_line, save_file=ret_args.tldr_doc_line_embed_save_file)

        tldr_whole_retrieve(ret_args)
        # tldr_line_retrieve(ret_args)


    elif ret_args.dataset == 'conala':
        conala_args = conala_config()

        with open(conala_args.qs_file, "r") as f:
            dataset = []
            for line in f:
                dataset.append(line.strip())
        # encoder.encode(dataset=dataset, save_file=ret_args.conala_qs_embed_save_file)

        with open(conala_args.doc_firstpara_file, "r") as f:
            dataset = []
            for line in f:
                dataset.append(line.strip())
        # encoder.encode(dataset=dataset, save_file=ret_args.conala_doc_firstpara_embed_save_file)

        conala_retrieve(ret_args)
