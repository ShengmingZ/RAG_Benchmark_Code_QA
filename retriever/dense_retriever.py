import platform
import sys, os
import h5py
import backoff
import openai
import json
import numpy as np
import faiss
from npy_append_array import NpyAppendArray

system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)

from retriever.dense_encoder import DenseRetrievalEncoder
from retriever_utils import retriever_config, ret_eval
from dataset_utils.conala_utils import ConalaLoader
from dataset_utils.DS1000_utils import DS1000Loader
from dataset_utils.hotpotQA_utils import HotpotQAUtils
from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
from dataset_utils.corpus_utils import PythonDocsLoader, WikiCorpusLoader

openai.api_key = os.environ['OPENAI_API_KEY']

def retrieve_by_faiss(qs_embed, doc_embed, qs_id_list, doc_key_list, save_file, top_k):
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
    if doc_embed is None:   # for WIKI openai embedding, OOM if load all to the memory
        doc_embed_file = '/data/zhaoshengming/wikipedia/embed_openai-embedding_NQ.h5'

        def yield_batches_from_hdf5(file_path, dataset_name='wiki_embedding', batch_size=1024):
            with h5py.File(file_path, 'r') as f:
                dataset = f[dataset_name]
                total_size = dataset.shape[0]
                for start in range(0, total_size, batch_size):
                    end = min(start + batch_size, total_size)
                    yield dataset[start:end]

        indexer = faiss.IndexFlatIP(qs_embed.shape[1])
        # res = faiss.StandardGpuResources()
        # gpu_indexer = faiss.index_cpu_to_gpu(res, 0, indexer)
        for batch in yield_batches_from_hdf5(doc_embed_file):
            if batch.dtype != np.float32: batch = batch.astype(np.float32)
            indexer.add(batch)
        if qs_embed.dtype != np.float32: qs_embed = qs_embed.astype(np.float32)
        D, I = indexer.search(qs_embed, top_k)
    else:
        assert qs_embed.shape[0] == len(qs_id_list)
        assert doc_embed.shape[0] == len(doc_key_list)
        if qs_embed.dtype != np.float32: qs_embed = qs_embed.astype(np.float32)
        if doc_embed.dtype != np.float32: doc_embed = doc_embed.astype(np.float32)
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


# def conala_retrieve(args):
#     conala_loader = ConalaLoader()
#     qs_list = conala_loader.load_qs_list(args.dataset_type)
#     qs_id_list = [qs['qs_id'] for qs in qs_list]
#     doc_list_firstpara = conala_loader.load_doc_list_firstpara()
#     doc_key_list = list(doc_list_firstpara.keys())
#
#     source_embed = np.load(args.conala_qs_embed_save_file + '.npy')
#     target_embed = np.load(args.conala_doc_firstpara_embed_save_file + '.npy')
#     # source_embed = np.load('docprompting_data/conala/.tmp/src_embedding.npy')
#     # target_embed = np.load('docprompting_data/conala/.tmp/tgt_embedding.npy')
#
#     ret_results = retrieve(source_embed, target_embed, qs_id_list, doc_key_list, args.save_file, args.top_k)
#
#     oracle_list = conala_loader.load_oracle_list(args.dataset_type)
#     gold, pred = list(), list()
#     for item in oracle_list:
#         gold.append(item['doc_keys'])
#         pred.append([tmp['doc_key'] for tmp in ret_results[item['qs_id']]])
#     conala_eval(gold=gold, pred=pred)


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
    # separately consider openai embedding
    if args.corpus == 'wiki_nq' and args.retriever == 'openai-embedding':
        wiki_loader = WikiCorpusLoader()
        if os.path.exists(args.corpus_embed_file + '.npy'):
            all_embeddings = np.load(args.corpus_embed_file + '.npy')
        else:
            all_embeddings = None
        resume = all_embeddings.shape[0] if all_embeddings is not None else 0

        @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
        def openai_encode(model_name, texts):
            return openai.Embedding.create(model=model_name, input=texts)

        def embed_and_append(model_name, batch):
            response = openai_encode(model_name=model_name, texts=batch)
            embeds = np.array([data["embedding"] for data in response['data']])
            with NpyAppendArray(args.corpus_embed_file + '.npy') as npaa:
                npaa.append(embeds)

        batch_count = 0
        data_count = 0
        batch = []
        all_embeddings_count = all_embeddings.shape[0]
        for data in wiki_loader.load_wiki_corpus_iter('NQ'):
            if data_count < resume:
                data_count += 1
                continue
            else:
                if batch_count == 1024:
                    embed_and_append(model_name='text-embedding-3-small', batch=batch)
                    all_embeddings_count += 1
                    print(f'done embedding {all_embeddings_count}')
                    batch = []
                    batch_count = 0
                batch_count += 1
                batch.append(data['text'])
        embed_and_append(model_name='text-embedding-3-small', batch=batch)  # last batch is not embed in the loop

    else:
        if os.path.exists(args.corpus_embed_file + '.npy'):
            print(f'embedding file exists for {args.corpus_embed_file}')
            return
        encoder = DenseRetrievalEncoder(args)
        if args.corpus == 'wiki_nq' or args.corpus == 'wiki_hotpot':
            wiki_loader = WikiCorpusLoader()
            wiki_data = wiki_loader.load_wiki_corpus(args.dataset)
            dataset = [item['text'] for item in wiki_data]
        elif args.corpus == 'python_docs':
            loader = PythonDocsLoader()
            python_docs = loader.load_api_docs()
            dataset = [item['doc'] for item in python_docs]
        encoder.encode(dataset=dataset, save_file=args.corpus_embed_file)
        if args.normalize_embed is True:
            normalize_embed(args.corpus_embed_file)


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

    if os.path.exists(embed_file + '_normalized' + '.npy'):
        print(f'embedding file exists for {embed_file+"_normalized"}')
        return
    embed = np.load(embed_file + '.npy')
    nor_embed = normalize(embed)
    save_file = embed_file + '_normalized'
    np.save(save_file, nor_embed)


def retrieve(args):
    if not os.path.exists(args.ret_result):
        # load
        if args.dataset == 'hotpotQA':
            loader = HotpotQAUtils()
            doc_id_list = WikiCorpusLoader().load_wiki_id(args.dataset)
        elif args.dataset == 'NQ' or args.dataset == 'TriviaQA':
            loader = NQTriviaQAUtils(args.dataset)
            doc_id_list = WikiCorpusLoader().load_wiki_id(args.dataset)
        elif args.dataset == 'conala':
            loader = ConalaLoader()
            # doc_id_list = [item[0] for item in PythonDocsLoader().load_api_signs()]
            doc_id_list = PythonDocsLoader().load_api_signs()
        elif args.dataset == 'DS1000':
            loader = DS1000Loader()
            # doc_id_list = [item[0] for item in PythonDocsLoader().load_api_signs()]
            doc_id_list = PythonDocsLoader().load_api_signs()
        elif args.dataset == 'pandas_numpy_eval':
            loader = PandasNumpyEvalLoader()
            # doc_id_list = [item[0] for item in PythonDocsLoader().load_api_signs()]
            doc_id_list = PythonDocsLoader().load_api_signs()
        qs_list = loader.load_qs_list()
        qs_id_list = [qs['qs_id'] for qs in qs_list]

        # get embed
        if not os.path.exists(args.qs_embed_file+'.npy'):
            encoder = DenseRetrievalEncoder(args)
            encoder.encode(dataset=[qs['question'] for qs in qs_list], save_file=args.qs_embed_file)
        if args.normalize_embed:
            normalize_embed(args.qs_embed_file)
            qs_embed = np.load(args.qs_embed_file + '_normalized' + '.npy')
            args.ret_result = args.result_file.replace('.json', '_normalized.json')
        else:
            qs_embed = np.load(args.qs_embed_file + '.npy')
        if args.corpus == 'wiki_nq' and args.retriever == 'openai-embedding':   # only for WIKI openai-embedding, deal with OOM
            doc_embed = None
        else:
            if args.normalize_embed:
                doc_embed = np.load(args.corpus_embed_file + '_normalized' + '.npy')
            else:
                doc_embed = np.load(args.corpus_embed_file + '.npy')

        # retrieve
        ret_results = retrieve_by_faiss(qs_embed, doc_embed, qs_id_list, doc_id_list, args.ret_result, args.top_k)
        print(f'done retrieval for {args.ret_result}')

    else:
        print(f'ret results exists for {args.ret_result}')



if __name__ == '__main__':
    in_program_call = None
    # in_program_call = f"--dataset hotpotQA --retriever miniLM"
    ret_args = retriever_config(in_program_call)

    embed_corpus(ret_args)
    retrieve(ret_args)
    ret_eval(ret_args)

    # test openai embed
    # in_program_call = f"--dataset NQ --retriever openai-embedding"
    # ret_args = retriever_config(in_program_call)
    # # embed_corpus(ret_args)
    # wiki_loader = WikiCorpusLoader()
    # wiki_data = wiki_loader.load_wiki_corpus(ret_args.dataset)
    # dataset = [item['text'] for item in wiki_data]
    # print(dataset[1])
    #
    # import tiktoken
    # import openai
    # OPENAI_TOKENIZER = "cl100k_base"
    # OPENAI_MAX_TOKENS = 1000
    # encoding = tiktoken.get_encoding(OPENAI_TOKENIZER)
    # data = dataset[1]
    # encoded_doc = encoding.encode(data)[:OPENAI_MAX_TOKENS]
    # data = encoding.decode(encoded_doc)
    # print(data)
    # response = openai.Embedding.create(model='text-embedding-3-small', input=data)
    # embeds = np.array([response['data'][0]['embedding']])
