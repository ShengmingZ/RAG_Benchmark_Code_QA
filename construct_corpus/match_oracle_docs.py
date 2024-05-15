import json
import re
import platform
import sys
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_utils.corpus_utils import PythonDocsLoader



def process_python_docs():
    python_docs = load_api_docs()
    # print(len(python_docs.items()))
    _python_docs = dict()
    for key, value in python_docs.items():
        if value not in python_docs:
            _python_docs[value] = [key]
        else:
            _python_docs[value].append(key)
    # print(len(_python_docs.items()))

    return _python_docs



def process_oracle_docs(oracle_list_file):
    # processed_python_docs = process_python_docs()
    python_docs = load_api_docs()
    python_doc_ids = load_api_signs()
    python_docs_main = dict()
    for key, value in python_docs.items():
        lines = value.split('\n')
        new_value = ''
        for line in lines[3:]:
            new_value += line
        python_docs_main[key] = new_value

    oracle_list = json.load(open(oracle_list_file, 'r'))
    regex_prefix = re.compile('method of (.+) instance')
    regex_method = re.compile('method (.+) in module')
    for idx, oracle in enumerate(oracle_list):
        oracle_doc_ids = []
        for oracle_doc in oracle['oracle_docs']:
            # match_prefix = regex_prefix.search(oracle_doc)
            # match_method = regex_method.search(oracle_doc)
            # if match_prefix and match_method:
            #     api_sign = match_prefix.group(1) + '.' + match_method.group(1)
            #     if api_sign not in python_doc_ids:
            #         # print(oracle['qs_id'], api_sign)
            #         ...
            # else:
            matched_doc_ids = []
            lines = oracle_doc.split('\n')
            oracle_doc_main = ''
            for line in lines[3:]:
                oracle_doc_main += line
            for doc_id, doc_main in python_docs_main.items():
                # lines = doc.split('\n')
                # doc_main = ''
                # for line in lines[3:]:
                #     doc_main += line
                if doc_main == oracle_doc_main:
                    matched_doc_ids.append(doc_id)
            if len(matched_doc_ids) == 0:
                print(oracle['qs_id'])
            else:
                oracle_doc_ids.append(matched_doc_ids[0])

        oracle_list[idx]['oracle_docs'] = oracle_doc_ids

    with open(oracle_list_file.replace('new', 'processed'), 'w+') as f:
        json.dump(oracle_list, f, indent=2)


if __name__ == '__main__':
    ds1000_oracle_file = '../data/DS1000/oracle_docs_matched_new.json'
    pandas_numpy_eval_oracle_file = '../data/pandas-numpy-eval/data/oracle_docs_matched_new.json'
    conala_oracle_file = '../data/conala/oracle_docs_matched_new.json'

    # process_oracle_docs(ds1000_oracle_file)
    # process_oracle_docs(pandas_numpy_eval_oracle_file)
    process_oracle_docs(conala_oracle_file)

    # process_python_docs()

