from sparse_retriever import sparse_retriever_config
from dense_retriever import dense_retriever_config
import json

BEST_RETRIEVER = {
    "hotpotQA": "",
    "NQ": "",
    "TriviaQA": "",
    "DS1000": "",
    "conala": "",
    "pandas-numpy-eval": ""
}

def get_ret_result(dataset, retriever, normalize):
    if retriever == "best":
        retriever = BEST_RETRIEVER[dataset]
    if retriever == "BM25":
        args = sparse_retriever_config(f"--dataset {dataset}")
        ret_result_file = args.ret_result
    elif retriever == "Contriever":
        ...
    else:
        args = dense_retriever_config(f"--dataset {dataset} --model_name {retriever}")
        if normalize:
            ret_result_file = args.result_file.replace(".json", "_normalized.json")

    return json.load(open(ret_result_file, 'r'))