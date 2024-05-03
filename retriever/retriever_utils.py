from retriever.sparse_retriever import sparse_retriever_config
from retriever.dense_retriever import dense_retriever_config
import json

BEST_RETRIEVER = {
    "hotpotQA": "",
    "NQ": "",
    "TriviaQA": "",
    "DS1000": "",
    "conala": "",
    "pandas-numpy-eval": ""
}

def get_ret_results(dataset, retriever, normalize=False):
    if retriever == "best":
        retriever = BEST_RETRIEVER[dataset]
    if retriever == "bm25":
        args = sparse_retriever_config(f"--dataset {dataset}")
        ret_result_file = args.ret_result
    elif retriever in ["contriever", 'miniLM', 'openai-embedding']:
        args = dense_retriever_config(f"--dataset {dataset} --model_name {retriever}")
        if normalize:
            ret_result_file = args.result_file.replace(".json", "_normalized.json")

    return json.load(open(ret_result_file, 'r'))