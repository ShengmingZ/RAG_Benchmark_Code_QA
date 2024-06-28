import cohere
import platform
import sys, os
import backoff
import unicodedata
import json
from tqdm import tqdm
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from retriever.retriever_utils import get_ret_results, retriever_config, ret_eval
from dataset_utils.corpus_utils import WikiCorpusLoader, PythonDocsLoader
from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
from dataset_utils.hotpotQA_utils import HotpotQAUtils
from dataset_utils.DS1000_utils import DS1000Loader
from dataset_utils.conala_utils import ConalaLoader
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
from generator.generate_utils import truncate_docs
from generator.run_model import chatgpt

COHERE_API_KEY = 'fYEeEiZr5jloupFJyBxbZr0FCrHCkpabEIsCNplm'
co = cohere.Client(COHERE_API_KEY)


@backoff.on_exception(backoff.constant, cohere.errors.TooManyRequestsError, interval=30)
def cohere_with_backoff(question, docs):
    response = co.rerank(
        model="rerank-english-v3.0",
        query=question,
        documents=docs,
        top_n=100,
    )
    response = response.json()
    response = json.loads(response)
    return response

def llm_rerank(question, docs, model):
    sys_prompt_reverse = """you are a helpful assistant, given a document and a question,
your task is to identify whether the document can derive the question, 
you should first give some explanation why, and then give the exactly answer yes or no"""

    sys_prompt = """you are a helpful assistant, given a document and a question, '
your task is to identify whether this document is helpful to answer the question, 
you should first give some explanation why, and then give the exactly answer yes or no"""


def rerank(ret_args, rerank_type):
    dataset = ret_args.dataset
    rerank_ret_results_file = ret_args.ret_result.replace('.json', '_rerank_cohere.json')
    # if not os.path.exists(rerank_ret_results_file):
    if True:
        # load ret results and qs_list
        ret_results = json.load(open(ret_args.ret_result, 'r'))
        if dataset == 'NQ' or dataset == 'TriviaQA': qs_list = NQTriviaQAUtils(dataset).load_qs_list()
        elif dataset == 'hotpotQA': qs_list = HotpotQAUtils().load_qs_list()
        elif dataset == 'conala': qs_list = ConalaLoader().load_qs_list()
        elif dataset == 'DS1000': qs_list = DS1000Loader().load_qs_list()
        elif dataset == 'pandas_numpy_eval': qs_list = PandasNumpyEvalLoader().load_qs_list()
        else: raise ValueError(f'Unknown dataset: {dataset}')
        # get top 100 docs for each sample as docs_list
        ret_doc_keys_list = []
        for qs in qs_list:
            ret_doc_keys_list.append([item['doc_key'] for item in ret_results[qs['qs_id']]][:100])
        if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
            docs_list = WikiCorpusLoader().get_docs(ret_doc_keys_list, dataset, num_procs=8)
        else:
            docs_list = []
            for ret_doc_keys in ret_doc_keys_list:  # truncate 1000 tokens to keep consistency with generation
                docs_list.append(truncate_docs(PythonDocsLoader().get_docs(ret_doc_keys), model='gpt-3.5-turbo', max_length=1000))
        # rerank
        rerank_results = dict()
        for qs, docs in tqdm(zip(qs_list, docs_list), total=len(qs_list)):
            if rerank_type == 'cohere':
                response = cohere_with_backoff(qs['question'], docs)
                # proc response
                ret_result = ret_results[qs['qs_id']]
                if dataset == 'NQ' or dataset == 'TriviaQA':
                    doc_indexes = [item['index'] for item in response['results']]
                    has_answer_list = [item['has_answer'] for item in ret_result]
                    doc_key_list = [item['doc_key'] for item in ret_result]
                    rerank_has_answer_list = [has_answer_list[index] for index in doc_indexes]
                    rerank_doc_key_list = [doc_key_list[index] for index in doc_indexes]
                    rerank_rele_score_list = [item['relevance_score'] for item in response['results']]
                    rerank_results[qs['qs_id']] = [dict(doc_key=doc_key, score=score, has_answer=has_answer) for doc_key, score, has_answer
                                             in zip(rerank_doc_key_list, rerank_rele_score_list, rerank_has_answer_list)]
                    rerank_results[qs['qs_id']].append(response)
                else:
                    doc_indexes = [item['index'] for item in response['results']]
                    doc_key_list = [item['doc_key'] for item in ret_result]
                    rerank_doc_key_list = [doc_key_list[index] for index in doc_indexes]
                    rerank_rele_score_list = [item['relevance_score'] for item in response['results']]
                    rerank_results[qs['qs_id']] = [dict(doc_key=doc_key, score=score) for doc_key, score in zip(rerank_doc_key_list, rerank_rele_score_list)]
                    rerank_results[qs['qs_id']].append(response)
            elif rerank_type == 'gpt':
                ...
            else:
                raise NotImplementedError

        # save
        with open(rerank_ret_results_file, 'w+') as f:
            json.dump(rerank_results, f, indent=2)

    # eval
    if dataset not in ['NQ', 'TriviaQA']:
        ret_args.ret_result = rerank_ret_results_file
        ret_eval(ret_args)
    else:
        rerank_results = json.load(open(rerank_ret_results_file, 'r'))
        top_k = [1,3,5,10,20,50,100]
        metrics = dict()
        for k in top_k:
            metrics[k] = 0
            for rerank_result in rerank_results.values():
                has_answer_list = [item['has_answer'] for item in rerank_result[:k]]
                if True in has_answer_list:
                    metrics[k] += 1
            metrics[k] = metrics[k] / len(rerank_results)
        print(metrics)


if __name__ == '__main__':
    in_program_call = None
    # in_program_call = '--dataset pandas_numpy_eval --retriever openai-embedding'
    ret_args = retriever_config(in_program_call)
    rerank(ret_args, rerank_type='cohere')

    # ret_eval(ret_args)


    # qs_list = DS1000Loader().load_qs_list()
    # ret_results = json.load(open(ret_args.ret_result, 'r'))
    # for qs_id, qs in zip(ret_results.keys(), qs_list):
    #     if qs_id != qs['qs_id']:
    #         print(qs_id)
    #         ret_results.pop(qs_id)
    #         break
    # with open(ret_args.ret_result, 'w+') as f:
    #     json.dump(ret_results, f, indent=2)
