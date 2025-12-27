"""
provide an inferface for retrieval, isolating old retrieving codes
"""
import random
import sys
sys.path.append('..')
# import platform
# system = platform.system()
# if system == 'Darwin':
#     root_path = '/'
# elif system == 'Linux':
#     root_path = '/home/zhaoshengming/RAG_Benchmark_Code_QA'
# sys.path.insert(0, root_path)
import json
import openai
from generator.generate_utils import get_docs_tokens
from retriever.retriever_utils import retriever_config, get_ret_results
import re
from collections import defaultdict
from dataset_utils.corpus_utils import PythonDocsLoader, WikiCorpusLoader
from copy import deepcopy


class RetrievalProvider:
    def __init__(self, dataset, retriever):  # dataset in ['conala', 'DS1000', 'pandas_numpy_eval', 'hotpotQA', 'NQ', 'TriviaQA']
        assert dataset in ['conala', 'DS1000', 'pandas_numpy_eval', 'hotpotQA', 'NQ', 'TriviaQA']
        self.dataset = dataset
        self.python_docs_path = '../data/python_docs/redirected_filtered_docs.json'
        self.retriever = retriever  # select from ['BM25', 'openai-embedding', 'miniLM', 'contriever', 'codeT5']
        self.ret_results_path_old = f'../data/{dataset}/ret_result_{self.retriever}.json'
        if dataset in ['NQ', 'hotpotQA', 'TriviaQA']:
            self.ret_results_path = f'../data/{dataset}/ret_result_{self.retriever}.json'
            self.oracle_doc_keys_path = f'../data/{dataset}/sampled_data.json'
        else:
            self.ret_results_path = f'../data/{dataset}/filtered_ret_result_{self.retriever}.json'
            self.oracle_doc_keys_path = f'../data/{dataset}/oracle_docs_matched_processed.json'

        self.oracle_docs_path = f'../data/{dataset}/persist_oracle_docs.json'
        self.ret_docs_path = f'../data/{dataset}/persist_ret_docs_{self.retriever}.json'

        # retrieval recall
        self.recall_range = [1.0, 0.8, 0.6, 0.4, 0.2, 0]

    # def get_ret_results(self):
    #     return json.load(open(self.ret_results_path, 'r'))

    def filter_ret_results(self):
        assert 'filtered' in self.ret_results_path  # make sure don't overwrite original result
        python_docs = json.load(open(self.python_docs_path, 'r'))
        python_apis = []
        for item in python_docs:
            python_apis.extend(item['api_sign'])

        ret_results = json.load(open(self.ret_results_path_old, 'r'))
        filtered_ret_results = dict()
        for data_id in ret_results:
            filtered_ret_results[data_id] = []
            for item in ret_results[data_id]:
                if item['doc_key'] in python_apis:  # only keep item that doc key exists in current python apis
                    filtered_ret_results[data_id].append(item)

        with open(self.ret_results_path, 'w+') as f:
            json.dump(filtered_ret_results, f, indent=2)

    @staticmethod
    def _look_up_doc(api_sign, python_docs):
        for item in python_docs:
            if api_sign in item['api_sign']:
                return item['doc']
        return False

    def get_oracle_docs(self):
        # python_docs = json.load(open(self.python_docs_path, 'r'))
        # oracle_doc_keys_list = json.load(open(self.oracle_doc_keys_path, 'r'))
        # oracle_docs = dict()
        # for oracle_item in oracle_doc_keys_list:
        #     data_id = oracle_item['qs_id']
        #     oracle_docs[data_id] = []
        #     doc_keys = oracle_item['oracle_docs']
        #     for doc_key in doc_keys:
        #         doc_text = self._look_up_doc(api_sign=doc_key, python_docs=python_docs)
        #         if doc_text is False: raise Exception(f'Oracle doc {doc_key} not found')
        #         oracle_docs[data_id].append(doc_text)
        #         # if get_docs_tokens([doc_text], model='gpt-3.5-turbo')[0] <= 20:
        #         #     print(doc_key)
        # return oracle_docs
        oracle_docs = json.load(open(self.oracle_docs_path, 'r'))
        return oracle_docs

    def get_ret_docs(self):
        return json.load(open(self.ret_docs_path, 'r'))

    def persist_oracle_docs(self):
        """
        load and store oracle documents for each dataset
        :return:
        """
        oracle_docs = dict()
        if self.dataset in ['pandas_numpy_eval', 'DS1000', 'conala']:
            oracle_doc_keys = json.load(open(self.oracle_doc_keys_path, 'r'))
            doc_loader = PythonDocsLoader()
            for item in oracle_doc_keys:
                oracle_docs[item['qs_id']] = doc_loader.get_docs(item['oracle_docs'])
        else:
            doc_loader = WikiCorpusLoader()
            qs_list = json.load(open(self.oracle_doc_keys_path, 'r', encoding='utf-8'))
            oracle_doc_keys = dict()
            for idx, qs in enumerate(qs_list):
                if self.dataset == 'hotpotQA':
                    proc_sp = list(set([sp[0] for sp in qs['supporting_facts']]))
                    assert len(proc_sp) == 2    # 2 supporting facts per sample in hotpotQA
                    oracle_doc_keys[qs['_id']] = proc_sp
                else:
                    data_id = str(idx)      # data id for NQ and TriviaQA
                    oracle_doc_keys[data_id] = [qs['oracle_doc']]
            oracle_doc_keys_list = [oracle_doc_keys[key] for key in oracle_doc_keys]     # transform to the format to doc_loader
            oracle_docs_list = doc_loader.get_docs(doc_keys_list=oracle_doc_keys_list, dataset=self.dataset, num_procs=8)
            for docs, data_id in zip(oracle_docs_list, oracle_doc_keys):
                oracle_docs[data_id] = docs

        with open(self.oracle_docs_path, 'w+', encoding='utf-8') as f:
            json.dump(oracle_docs, f, indent=2)


    def persist_ret_docs(self):
        """
        load and store oracle documents for each dataset
        :return:
        """
        ret_docs = dict()
        if self.dataset in ['pandas_numpy_eval', 'DS1000', 'conala']:
            ret_doc_keys = json.load(open(self.ret_results_path, 'r'))
            doc_loader = PythonDocsLoader()
            for qs_id in ret_doc_keys:
                ret_docs[qs_id] = doc_loader.get_docs([item['doc_key'] for item in ret_doc_keys[qs_id]])
        else:
            doc_loader = WikiCorpusLoader()
            ret_doc_keys = json.load(open(self.ret_results_path, 'r', encoding='utf-8'))
            for qs_id in ret_doc_keys:
                ret_doc_keys[qs_id] = [item['doc_key'] for item in ret_doc_keys[qs_id]]
            ret_doc_keys_list = [ret_doc_keys[key] for key in ret_doc_keys]     # transform to the format to doc_loader
            ret_docs_list = doc_loader.get_docs(doc_keys_list=ret_doc_keys_list, dataset=self.dataset, num_procs=8)
            for docs, data_id in zip(ret_docs_list, ret_doc_keys):
                ret_docs[data_id] = docs

        with open(self.ret_docs_path, 'w+', encoding='utf-8') as f:
            json.dump(ret_docs, f, indent=2)


    # def persist_ret_docs(self):
    #     """
    #     load and store retrieval documents for each dataset
    #     :return:
    #     """
    #     ret_docs = dict()
    #     if self.dataset in ['pandas_numpy_eval', 'DS1000', 'conala']:
    #         ret_doc_keys = json.load(open(self.ret_results_path, 'r'))
    #         doc_loader = PythonDocsLoader()
    #         for qs_id in ret_doc_keys:
    #             ret_docs[qs_id] = doc_loader.get_docs([elem['doc_key'] for elem in ret_doc_keys[qs_id]])
    #     elif self.dataset in ['NQ']:
    #         original_ret_doc_path = f'../data/{self.dataset}/ret_results_docs_openai-embedding.json'
    #         original_ret_docs = json.load(open(original_ret_doc_path, 'r'))
    #         original_ret_doc_keys_path = f'../data/{self.dataset}/ret_result_openai-embedding.json'
    #         original_ret_doc_keys = json.load(open(original_ret_doc_keys_path, 'r'))
    #         ret_docs = dict()
    #         for idx, pid in enumerate(original_ret_docs):
    #             ret_doc = []
    #             for doc_idx, doc_item in enumerate(original_ret_docs[pid]):
    #                 assert original_ret_doc_keys[pid][doc_idx]['doc_key'] == original_ret_docs[pid][doc_idx]['doc_key']
    #                 ret_doc.append(dict(doc=original_ret_docs[pid][doc_idx]['doc'], golden=original_ret_doc_keys[pid][doc_idx]['has_answer']))
    #             ret_docs[pid] = ret_doc
    #     elif self.dataset == 'hotpotQA':
    #         original_ret_doc_path = f'../data/{self.dataset}/ret_results_docs_openai-embedding.json'
    #         original_ret_docs = json.load(open(original_ret_doc_path, 'r'))
    #         original_ret_doc_keys_path = f'../data/{self.dataset}/ret_result_openai-embedding.json'
    #         original_ret_doc_keys = json.load(open(original_ret_doc_keys_path, 'r'))
    #         qs_list = json.load(open('../data/hotpotQA/sampled_data.json', 'r'))
    #         ret_docs = dict()
    #         for idx, pid in enumerate(original_ret_docs):
    #             ret_doc = []
    #             assert qs_list[idx]['_id'] == pid
    #             golden_keys = list(set([sp[0] for sp in qs_list[idx]['supporting_facts']]))
    #             for doc_idx, doc_item in enumerate(original_ret_docs[pid]):
    #                 assert original_ret_doc_keys[pid][doc_idx]['doc_key'] == original_ret_docs[pid][doc_idx]['doc_key']
    #                 if original_ret_doc_keys[pid][doc_idx]['doc_key'] in golden_keys: has_answer = True
    #                 else: has_answer = False
    #                 ret_doc.append(dict(doc=original_ret_docs[pid][doc_idx]['doc'], golden=has_answer))
    #             ret_docs[pid] = ret_doc
    #
    #     elif self.dataset == 'TriviaQA':
    #         doc_loader = WikiCorpusLoader()
    #         ret_doc_keys_path = '../data/TriviaQA/ret_result_openai-embedding.json'
    #         ret_doc_keys_with_answer = json.load(open(ret_doc_keys_path, 'r', encoding='utf-8'))
    #         ret_doc_keys = list()
    #         for pid in ret_doc_keys_with_answer:
    #             ret_doc_keys.append([item['doc_key'] for item in ret_doc_keys_with_answer[pid]])
    #         ret_docs_list = doc_loader.get_docs(doc_keys_list=ret_doc_keys, dataset=self.dataset, num_procs=16)
    #         ret_docs = dict()
    #         for docs, pid in zip(ret_docs_list, ret_doc_keys_with_answer):
    #             assert len(docs) == len(ret_doc_keys_with_answer[pid])
    #             pid_doc_list = [dict(doc=doc, golden=item['has_answer']) for doc, item in zip(docs, ret_doc_keys_with_answer[pid])]
    #             ret_docs[pid] = pid_doc_list
    #
    #     with open(self.ret_docs_path, 'w+', encoding='utf-8') as f:
    #         json.dump(ret_docs, f, indent=2)


    def calculate_recall(self, oracle_docs, ret_docs, controlled_docs):
        """Calculate average retrieval recall"""
        total_recall = 0
        count = 0

        for qid in oracle_docs:
            if self.dataset in ['conala', 'DS1000', 'PNE', 'hotpotQA']:
                golden_docs = set(oracle_docs[qid])
                current_docs = set(controlled_docs[qid])

                if len(golden_docs) > 0:
                    retrieved_golden = golden_docs.intersection(current_docs)
                    recall = len(retrieved_golden) / len(golden_docs)
                    total_recall += recall
                    count += 1

            else:
                recall = 0
                for doc in controlled_docs[qid]:    # 对于每一个doc
                    doc_in_ret_docs = False
                    if doc in oracle_docs[qid]: # 如果存在与oracle docs中，则直接返回
                        recall = 1
                        break
                    for item in ret_docs[qid]:  # 检查他是否存在与ret docs中
                        if doc == item['doc']:
                            if item['golden']: recall = 1   # 如果存在且has answer，recall设置为1
                            doc_in_ret_docs = True
                            break
                    assert doc_in_ret_docs
                total_recall += recall
                count += 1



        return total_recall / count if count > 0 else 0

    def creat_controlled_recall_docs(self):
        random.seed = 0

        controlled_recall_docs = dict()
        oracle_docs = self.get_oracle_docs()
        ret_docs = self.get_ret_docs()
        for qid in oracle_docs:
            golden_docs = set(oracle_docs[qid])
            if self.dataset in ['conala', 'DS1000', 'PNE']:
                retrieved_docs = set(ret_docs[qid])
                distractors = retrieved_docs - golden_docs
            else:
                distractors = [item['doc'] for item in ret_docs[qid] if item['golden'] is False]   # for QA, those not golden docs are distractors

            # construct controlled docs
            controlled_recall_docs[qid] = list(golden_docs) + list(distractors)

        for target_recall in self.recall_range[1:]:     # first one is all oracle
            while True:
                current_recall = self.calculate_recall(oracle_docs, ret_docs, controlled_recall_docs)

                if current_recall <= target_recall: break

                removable_golden = []   # collect all existing golden docs in controlled docs
                for qid in oracle_docs:
                    golden_docs = set(oracle_docs[qid])
                    current_docs = controlled_recall_docs[qid]
                    current_golden = golden_docs.intersection(current_docs)

                    for doc in current_golden:
                        removable_golden.append((qid, doc))

                # Remove one random golden doc
                qid_to_modify, doc_to_remove = random.choice(removable_golden)
                controlled_recall_docs[qid_to_modify].remove(doc_to_remove)

            # store the controlled docs
            persist_path = f'../data/{self.dataset}/controlled_docs_recall-{target_recall}.json'
            truncated_docs = {}
            for qid in oracle_docs:
                oracle_length = len(oracle_docs[qid])
                truncated_docs[qid] = controlled_recall_docs[qid][:oracle_length]
            # verify recall
            print(self.calculate_recall(oracle_docs, ret_docs, truncated_docs))
            with open(persist_path, 'w+') as f:
                json.dump(truncated_docs, f, indent=2)


    def creat_controlled_realistic_recall_docs(self):
        random.seed = 0

        used_distractors_dict = json.load(open(f'../data/{self.dataset}/controlled_docs_recall-0.json'))
        distractors_dict = dict()

        oracle_docs = self.get_oracle_docs()
        ret_docs = self.get_ret_docs()
        for qid in oracle_docs:
            golden_docs = set(oracle_docs[qid])
            if self.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
                retrieved_docs = set(ret_docs[qid])
                distractors = retrieved_docs - golden_docs
            else:
                distractors = [item['doc'] for item in ret_docs[qid] if item['golden'] is False]  # for QA, those not golden docs are distractors
            distractors_dict[qid] = [doc for doc in distractors if doc not in set(used_distractors_dict[qid])]

        import numpy as np

        controlled_recall_docs_list = []
        for recall in self.recall_range:
            controlled_recall_docs_list.append(json.load(open(f'../data/{self.dataset}/controlled_docs_recall-{recall}.json')))
        for qid in controlled_recall_docs_list[0]:
            # add more distracting docs
            for i in range(len(controlled_recall_docs_list)):
                controlled_recall_docs_list[i][qid] = controlled_recall_docs_list[i][qid] + distractors_dict[qid][:5]

            # shuffle
            n = len(controlled_recall_docs_list[0][qid])
            perm = np.random.permutation(n)
            for i in range(len(controlled_recall_docs_list)):
                controlled_recall_docs_list[i][qid] = [controlled_recall_docs_list[i][qid][j] for j in perm]

        for i, recall in enumerate(self.recall_range):
            persist_path = f'../data/{self.dataset}/controlled_docs_realistic_recall-{recall}.json'
            with open(persist_path, 'w+') as f:
                json.dump(controlled_recall_docs_list[i], f, indent=2)




    def verify_recall_control(self):
        controlled_recall_docs_path = f'../data/{self.dataset}/controlled_docs_recall-{0.8}.json'
        controlled_recall_docs = json.load(open(controlled_recall_docs_path, 'r'))
        oracle_docs = self.get_oracle_docs()
        total_recall = 0
        count = 0
        for qid in oracle_docs:
            oracle_doc = set(oracle_docs[qid])
            ret_doc = set(controlled_recall_docs[qid])
            retrieved_golden = oracle_doc.intersection(ret_doc)
            recall = len(retrieved_golden) / len(oracle_doc)
            print(recall)
            total_recall += recall
            count += 1
        print(total_recall/count)


    def get_recall_controlled_docs(self, recall):
        assert recall in self.recall_range
        persist_path = f'../data/{self.dataset}/controlled_docs_recall-{recall}.json'
        return json.load(open(persist_path, 'r'))


    def get_realistic_recall_controlled_docs(self, recall):
        assert recall in self.recall_range
        persist_path = f'../data/{self.dataset}/controlled_docs_realistic_recall-{recall}.json'
        return json.load(open(persist_path, 'r'))




if __name__ == '__main__':
    ret_provider = RetrievalProvider(dataset='TriviaQA', retriever='openai-embedding')

    # ret_provider.filter_ret_results()

    # ret_provider.persist_oracle_docs()

    # ret_provider.get_oracle_docs()

    # ret_provider.persist_ret_docs()

    # ret_provider.creat_controlled_recall_docs()

    ret_provider.creat_controlled_realistic_recall_docs()

    # ret_provider.verify_recall_control() # 记得用服务器上的数据覆盖一下！





"""
re-process python documents, remove document that direct to another document (e.g. Alias)
"""

def analyze_python_docs(python_docs_path='../data/python_docs/proc_python_docs.json', model='gpt-3.5-turbo', token_threshold=20):
    with open(python_docs_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    docs = [item.get("doc", "") for item in data]
    api_signs = [item.get("api_sign", "") for item in data]
    token_counts = get_docs_tokens(docs, model)

    max_tokens = max(token_counts)
    min_tokens = min(token_counts)
    max_idx = token_counts.index(max_tokens)
    min_idx = token_counts.index(min_tokens)

    # Find docs with tokens < threshold
    short_docs = []
    for i, (token_count, api_sign, doc) in enumerate(zip(token_counts, api_signs, docs)):
        if token_count < token_threshold:
            short_docs.append({
                "index": i,
                "tokens": token_count,
                "api_sign": api_sign,
                "doc": doc
            })

    # Sort short docs by token count
    short_docs.sort(key=lambda x: x["tokens"])

    return {
        "max_tokens": max_tokens,
        "max_api_sign": data[max_idx].get("api_sign", ""),
        "min_tokens": min_tokens,
        "min_api_sign": data[min_idx].get("api_sign", ""),
        "total_documents": len(data),
        "average_tokens": sum(token_counts) / len(token_counts),
        # New additions
        "short_docs": short_docs,
        "short_docs_count": len(short_docs),
        "short_docs_percentage": (len(short_docs) / len(data)) * 100,
        "threshold": token_threshold
    }





def filter_private_functions(python_docs_path='../data/python_docs/proc_python_docs.json',
                             filtered_docs_path='../data/python_docs/filtered_private_docs.json'):
    """
    Filter out private/internal functions (starting with '_') from API docs.
    """
    with open(python_docs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    public_docs = []
    private_docs = []

    for item in data:
        api_signs = item.get("api_sign", [])

        # func_name = api_signs[0].split('.')[-1]
        # if func_name.startswith('_'):
        #     private_docs.append(item)
        # else:
        #     public_docs.append(item)

        is_private = True

        for api_sign in api_signs:
            func_name = api_sign.split('.')[-1]

            # Filter only very obvious cases
            if (
                    # Obscure dunder methods (keep common ones)
                    (func_name.startswith('__') and func_name.endswith('__') and
                     func_name not in {'__init__', '__call__', '__getitem__', '__setitem__', '__len__', '__str__',
                                       '__repr__', '__iter__', '__next__'}) or

                    # Build/setup modules
                    any(pattern in api_sign for pattern in ['.build.', '.setup.', '._version.', '.conftest.']) or

                    # Internal function patterns
                    any(pattern in func_name for pattern in ['_setup_', '_build_', '_compile_', '_install_'])
            ):
                ...
            else:
                is_private = False
                break
        if is_private:
            private_docs.append(item)
        else:
            public_docs.append(item)

    # Save filtered data
    with open(filtered_docs_path, 'w+', encoding='utf-8') as f:
        json.dump(public_docs, f, indent=2, ensure_ascii=False)

    print(f"Filtered from {len(data)} to {len(public_docs)} public functions")
    print(f"Removed {len(data) - len(public_docs)} private functions")
    print(f"Saved to: {filtered_docs_path}")



def parse_redirect_with_llm(doc_text):
    """
    Use GPT-4o to determine if a short doc is a redirect and extract target.

    Args:
        doc_text (str): The documentation text

    Returns:
        dict: {"is_redirect": bool, "target_function": str or None}
    """
    import openai

    # Set your OpenAI API key here or use environment variable
    openai_api_key = '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'

    prompt = f"""Analyze this Python API documentation snippet and determine if it's a redirect that points to another function.

Documentation text: "{doc_text}"

A documentation is considered a redirect if it:
- Contains phrases like "See :func:", "See torch.", "Alias for", "Use ... instead", "Deprecated"
- Just points to another function without providing actual implementation details

If it's a redirect, extract the exact target full function name (e.g., "torch.isnan", "add", "torch.nn.functional.relu").
If it's not a redirect or you can't determine the target, set target_function to None.

Please strictly follow the response format: <D>True</D> if the doc is a redirect to another function, otherwise output <D>False</D>,
if true, provide the exact target full function in the format <F>function</F>, if False, output <F>None</F> instead"""

    client = openai.OpenAI()  # Uses OPENAI_API_KEY env var

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # Low temperature for consistent parsing
        max_tokens=100  # Short response expected
    )

    result_text = response.choices[0].message.content.strip()
    redirect_flag = result_text.split('<D>')[1].split('</D>')[0]
    if redirect_flag == 'True': redirect_flag = True
    elif redirect_flag == 'False': redirect_flag = False
    else: raise Exception('LLM output wrong format')
    actual_function = result_text.split('<F>')[1].split('</F>')[0]

    return redirect_flag, actual_function if redirect_flag else None


def extract_redirect_mappings(input_path='../data/python_docs/filtered_cpp_private_docs.json',
                              model='gpt-3.5-turbo',
                              token_threshold=100,
                              mappings_output='../data/python_docs/redirect_mappings.jsonl',
                              resume=0):
    """
    Extract redirect mappings from short documentation.

    Args:
        input_path (str): Path to public docs JSON file
        model (str): Model for token counting
        token_threshold (int): Threshold to consider doc "short"
        mappings_output (str): Path to save redirect mappings
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Analyzing {len(data)} documents for redirects...")

    # Calculate tokens for all docs
    docs = [item.get("doc", "") for item in data]
    token_counts = get_docs_tokens(docs, model)

    # Extract short docs
    short_docs = []
    for i, (item, token_count) in enumerate(zip(data, token_counts)):
        if token_count <= token_threshold:
            short_docs.append(item)

    print(f"Found {len(short_docs)} short docs (<= {token_threshold} tokens)")

    # remove docs with indicators
    redirect_keywords = [
        # Direct aliases
        'Alias for',
        'alias for',
        'Alias of',
        'alias of',

        # Cross-references
        'See :func:',
        'See :meth:',
        'See :class:',
        'See :mod:',
        'see :func:',
        'see :meth:',
        'see :class:',

        # Deprecation redirects
        'Deprecated',
        'deprecated',
        'Use :func:',
        'Use :meth:',
        'use :func:',
        'use :meth:',
        'Instead use',
        'instead use',

        # Wrapper indicators
        'Wrapper for',
        'wrapper for',
        'This is a wrapper',
        'Wraps',
        'wraps',

        # Equivalent/shortcut indicators
        'Equivalent to',
        'equivalent to',
        'Same as',
        'same as',
        'Shortcut for',
        'shortcut for',
        'Convenience function',
        'convenience function',

        # Backward compatibility
        'For backward compatibility',
        'for backward compatibility',
        'Legacy',
        'legacy',
        'Compatibility',
        'compatibility',

        # Implementation references
        'Implemented as',
        'implemented as',
        'Calls',
        'calls',
        'Delegates to',
        'delegates to'
    ]
    filtered_short_docs = []
    # if none of the indicators exist in the document
    for short_doc in short_docs:
        if any(indicator in short_doc['doc'] for indicator in redirect_keywords):
            filtered_short_docs.append(short_doc)

    # Analyze short docs with LLM
    redirect_count = 0

    for i, short_doc in enumerate(filtered_short_docs[resume:]):
        doc_text = short_doc.get("doc", "")
        api_signs = short_doc.get("api_sign", [])

        print(f"Analyzing {i + 1 + resume}/{len(filtered_short_docs)}: {api_signs[0]}")

        # Use LLM to analyze
        is_redirect, actual_function = parse_redirect_with_llm(doc_text)

        result_record = {
            "api_sign": api_signs[0],
            "is_redirect": is_redirect,
            "actual_function": actual_function,
            "doc_text": doc_text
        }

        # Append to file immediately
        with open(mappings_output, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_record, ensure_ascii=False) + '\n')

        if is_redirect:
            redirect_count += 1
            print(f"  -> Redirect to: {actual_function}")

    print(f"\nRedirect analysis complete:")
    print(f"Processed {len(filtered_short_docs)} short docs")
    print(f"Found {redirect_count} redirects")
    print(f"Results saved to: {mappings_output}")



def filter_cpp_apis(input_path='../data/python_docs/filtered_private_docs.json',
                    output_path='../data/python_docs/filtered_cpp_private_docs.json',):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cpp_docs = []
    non_cpp_docs = []

    cpp_api_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*\([^()]*\)\n\n$'

    for item in data:
        doc_text = item.get("doc", [])

        is_cpp_api = bool(re.match(cpp_api_pattern, doc_text))

        if is_cpp_api:
            cpp_docs.append(item)
        else:
            non_cpp_docs.append(item)

    # Save filtered data
    with open(output_path, 'w+', encoding='utf-8') as f:
        json.dump(non_cpp_docs, f, indent=2, ensure_ascii=False)

    print(f"Filtered from {len(data)} to {len(non_cpp_docs)} public functions")
    print(f"Removed {len(data) - len(non_cpp_docs)} cpp interface functions")
    print(f"Saved to: {output_path}")



def filter_testing_apis(input_path='../data/python_docs/filtered_cpp_private_docs.json',
                        output_path='../data/python_docs/filtered_test_cpp_private_docs.json'):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    non_test_docs = []

    for item in data:
        api_signs = item.get("api_sign", [])

        # Check if any api_sign contains 'test' in package path
        is_test_function = True
        for api_sign in api_signs:
            api_sign = api_sign.rsplit('.')[0]  # package path
            if 'test' not in api_sign.lower():
                is_test_function = False
                break

        if not is_test_function:
            non_test_docs.append(item)

    # Save filtered data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(non_test_docs, f, indent=2, ensure_ascii=False)

    filtered_count = len(data) - len(non_test_docs)
    print(f"Filtered out {filtered_count} functions containing 'test'")
    print(f"Remaining: {len(non_test_docs)} functions")
    print(f"Saved to: {output_path}")



def apply_redirects(input_path='../data/python_docs/filtered_test_cpp_private_docs.json',
                    mapping_file='../data/python_docs/redirect_mappings.jsonl',
                    output_file='../data/python_docs/redirected_filtered_docs.json'):
    """
    Apply redirects from mapping data to original API documentation
    """


    with open(input_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    mapping_data = []
    with open(mapping_file, 'r') as f:
        for line in f:
            mapping_data.append(json.loads(line.strip()))

    # split mapping data, only keep functions with is_redirect=False
    redirected_indexes = []
    for mapping_item in mapping_data:
        if mapping_item['is_redirect']:
            for index, item in enumerate(datas):
                if mapping_item['api_sign'] in item['api_sign']:
                    redirected_indexes.append(index)
                    mapping_item['api_sign'] = item['api_sign']
                    break
    redirected_data = []
    kept_data = []
    for index in range(len(datas)):
        if index in redirected_indexes:
            redirected_data.append(datas[index])
        else:
            kept_data.append(datas[index])

    for mapping_item in mapping_data:
        if mapping_item['is_redirect'] and mapping_item['actual_function'] != 'None':
            target_function = mapping_item['actual_function']
            for index, data in enumerate(kept_data):
                is_mapped = False
                # matching all api signs
                for api_sign in data['api_sign']:
                    # match pattern of actual function
                    if api_sign == target_function:
                        is_mapped = True
                    elif (target_function in api_sign or api_sign.split('.')[-1] == target_function.split('.')[-1]) and mapping_item['api_sign'][0].split('.')[0] == api_sign.split('.')[0]:
                        is_mapped = True
                    if is_mapped:
                        kept_data[index]['api_sign'].extend(mapping_item['api_sign'])
                        break
                # if matched once, go to next item that needs mapping
                if is_mapped:
                    break

    with open(output_file, 'w+') as f:
        json.dump(kept_data, f, indent=2)

    print('the number of cleaned data: {}'.format(len(kept_data)))


# if __name__ == '__main__':
#     # doc_text = "sigmoid()\n    sigmoid(input, *, out=None) -> Tensor\n    \n    Alias for :func:`torch.special.expit`.\n\n"
#     # flag, exact_name = parse_redirect_with_llm(doc_text)
#     # print(flag, exact_name)
#
#     # filter_private_functions()
#
#     # filter_cpp_apis()
#
#     # filter_testing_apis()
#
#     # extract_redirect_mappings(token_threshold=50)
#
#     apply_redirects()




# def clean_api_docs(python_docs_path='../data/python_docs/proc_python_docs.json', model='gpt-3.5-turbo', token_threshold=20, output_path='../data/python_docs/python_docs_cleaned.json'):
#     """
#     Clean API documentation by:
#     1. Mapping short redirect docs to actual documentation
#     2. Filtering out private/internal functions (starting with '_')
#     3. Combining api_signs for related docs
#
#     Args:
#         python_docs_path (str): Path to the JSON file
#         model (str): Model for token counting
#         token_threshold (int): Threshold to consider a doc "short" (default: 50)
#         output_path (str): Path to save cleaned data (optional)
#
#     Returns:
#         dict: Cleaned documentation data and statistics
#     """
#
#     with open(python_docs_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#
#     print(f"Original dataset: {len(data)} documents")
#
#     # Step 1: Filter out private/internal functions
#     public_docs = []
#     private_docs = []
#
#     for item in data:
#         api_signs = item.get("api_sign", [])
#         # Extract function name from api_sign
#         func_name = api_signs[0].split('.')[-1]
#
#         if func_name.startswith('_'):
#             private_docs.append(item)
#         else:
#             public_docs.append(item)
#
#     print(f"After filtering private functions: {len(public_docs)} public, {len(private_docs)} private")
#
#     # Step 2: Use LLM to process short docs and identify redirects
#     docs = [item.get("doc", "") for item in public_docs]
#     token_counts = get_docs_tokens(docs, model)
#
#     long_docs = []
#     redirect_mappings = defaultdict(set)
#     standalone_short_docs = []
#
#     for i, (item, token_count) in enumerate(zip(public_docs, token_counts)):
#         doc_text = item.get("doc", "")
#         api_signs = item.get("api_sign", [])
#
#         if token_count < token_threshold:
#             # Use LLM to analyze short doc
#             llm_result = parse_redirect_with_llm(doc_text, api_signs)
#
#             if llm_result["is_redirect"] and llm_result["target_function"]:
#                 # This is a redirect - store the mapping
#                 target_func = llm_result["target_function"]
#                 for api_sign in api_signs:
#                     if isinstance(api_sign, str) and api_sign.strip():
#                         redirect_mappings[target_func].add(api_sign)
#                 print(f"  Redirect found: {api_signs} -> {target_func}")
#             else:
#                 # Short but not a redirect - might be legitimate short doc
#                 if token_count >= 10:  # Keep reasonably sized standalone docs
#                     standalone_short_docs.append({"item": item, "tokens": token_count})
#         else:
#             long_docs.append(item)
#
#     print(f"Found {len(short_docs)} short docs and {len(long_docs)} substantial docs")
#
#     # Step 4: Create mapping of function names to long docs
#     long_docs_map = {}
#     for long_doc in long_docs:
#         api_sign = long_doc["item"].get("api_sign", "")
#         func_name = api_sign.split('.')[-1].split('(')[0] if '.' in api_sign else api_sign.split('(')[0]
#
#         if func_name not in long_docs_map:
#             long_docs_map[func_name] = long_doc
#         else:
#             # If duplicate, keep the one with more tokens
#             if long_doc["tokens"] > long_docs_map[func_name]["tokens"]:
#                 long_docs_map[func_name] = long_doc
#
#     # Step 5: Map short docs to long docs and combine api_signs
#     api_sign_mapping = defaultdict(set)
#     unmatched_redirects = []
#
#     for short_doc in short_docs:
#         short_api_sign = short_doc["item"].get("api_sign", "")
#         redirect_target = short_doc["redirect_target"]
#
#         if redirect_target and redirect_target in long_docs_map:
#             # Found a match - add this api_sign to the target doc
#             api_sign_mapping[redirect_target].add(short_api_sign)
#         else:
#             # No clear redirect target found
#             unmatched_redirects.append(short_doc)
#
#     # Step 6: Create final cleaned dataset with new structure
#     cleaned_docs = []
#
#     for func_name, long_doc in long_docs_map.items():
#         original_item = long_doc["item"]
#
#         # Create new structure: change "api_sign" to "api_signs" as a list
#         cleaned_item = {
#             "doc": original_item.get("doc", ""),
#             "api_signs": [original_item.get("api_sign", "")]  # Convert to list
#         }
#
#         # Add redirected api_signs if there are redirects pointing to this doc
#         if func_name in api_sign_mapping:
#             additional_signs = list(api_sign_mapping[func_name])
#             cleaned_item["api_signs"].extend(additional_signs)
#
#         # Remove empty api_signs
#         cleaned_item["api_signs"] = [sign for sign in cleaned_item["api_signs"] if sign.strip()]
#
#         cleaned_docs.append(cleaned_item)
#
#     # Step 7: Add high-quality unmatched short docs (might be legitimate short docs)
#     for short_doc in unmatched_redirects:
#         if short_doc["tokens"] >= 15:  # Keep reasonably sized docs even if short
#             original_item = short_doc["item"]
#             cleaned_item = {
#                 "doc": original_item.get("doc", ""),
#                 "api_signs": [original_item.get("api_sign", "")] if original_item.get("api_sign", "").strip() else []
#             }
#             cleaned_docs.append(cleaned_item)
#
#     print(f"Final cleaned dataset: {len(cleaned_docs)} documents")
#     print(f"Combined {sum(len(api_sign_mapping[k]) for k in api_sign_mapping)} redirect mappings")
#
#     # Step 8: Save cleaned data if output path provided
#     if output_path:
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(cleaned_docs, f, indent=2, ensure_ascii=False)
#         print(f"Cleaned data saved to {output_path}")
#
#     return {
#         "cleaned_docs": cleaned_docs,
#         "original_count": len(data),
#         "private_filtered": len(private_docs),
#         "public_count": len(public_docs),
#         "short_docs_count": len(short_docs),
#         "long_docs_count": len(long_docs),
#         "final_count": len(cleaned_docs),
#         "redirect_mappings": len(api_sign_mapping),
#         "unmatched_redirects": len(unmatched_redirects)
#     }
#
#
#
# if __name__ == '__main__':
#     print(analyze_python_docs())