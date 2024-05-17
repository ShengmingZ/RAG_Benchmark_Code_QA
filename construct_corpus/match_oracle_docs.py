import json
import re
import os
import platform
import sys
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_utils.corpus_utils import PythonDocsLoader


def process_oracle_docs(oracle_list_file, dataset):
    # process oracle list
    oracle_list = json.load(open(oracle_list_file, 'r'))
    for idx, oracle in enumerate(oracle_list):
        new_oracle_docs = []
        for doc in oracle['oracle_docs']:
            lines = doc.split('\n')
            # clear dup docs
            prefix = lines[0]
            is_dup = False
            for repeat_idx, line in enumerate(lines[1:]):
                if line == prefix:
                    is_dup = True
                    break
            if is_dup: repeat_idx += 1
            else: repeat_idx += 2
            # process function head
            function_head = re.sub(r'self[^,]*?:[^,]+?, ', '', lines[2])
            function_head = (function_head.replace('self, ', '').replace('self', '').replace('...', '')
                             .replace(', /', '').replace('/, ', ''))
            try:
                function_head = function_head[:re.search(r'\(.*\)', function_head).end()]
            except:
                ...
            main_content = function_head + '\n' + '\n'.join(lines[3:repeat_idx])
            if is_dup: main_content += '\n'
            new_oracle_docs.append(main_content)
        oracle_list[idx]['oracle_docs'] = new_oracle_docs

    return oracle_list


def main(dataset):
    # process oracle list
    assert dataset in ['pandas-numpy-eval', 'DS1000', 'conala']
    if dataset == 'pandas-numpy-eval':
        oracle_list_file = os.path.join(root_path, 'data/pandas-numpy-eval/data/oracle_docs_matched.json')
    elif dataset == 'DS1000':
        oracle_list_file = os.path.join(root_path, 'data/DS1000/oracle_docs_matched.json')
    else:
        oracle_list_file = os.path.join(root_path, 'data/conala/oracle_docs_matched.json')
    oracle_list = process_oracle_docs(oracle_list_file, dataset)
    python_docs = PythonDocsLoader().load_api_docs()

    # match oracle list with documentations
    for i, oracle in enumerate(oracle_list):
        oracle_list[i]['oracle_ids'] = [None]*len(oracle_list[i]['oracle_docs'])
    for item in python_docs:
        if item['doc'] == "<lambda> lambda x\n\n": continue
        for i, oracle in enumerate(oracle_list):
            for j, oracle_doc in enumerate(oracle['oracle_docs']):
                if item['doc'] == oracle_doc:
                    oracle_list[i]['oracle_ids'][j] = item['api_sign'][0]

    # check if every doc is matched
    for i, oracle in enumerate(oracle_list):
        for j, oracle_id in enumerate(oracle['oracle_ids']):
            if oracle_id is None: print(f'doc match failed for doc {[oracle["oracle_docs"][j][:10000]]}')

    # write back
    _oracle_list = list()
    for i, oracle in enumerate(oracle_list):
        oracle_docs = [item for item in oracle_list[i]['oracle_ids'] if item is not None]
        if oracle_docs != []: _oracle_list.append(dict(qs_id=oracle['qs_id'], oracle_docs=list(set(oracle_docs)), output=oracle['output'], full_name_list=oracle['full_name_list']))
    proc_oracle_list_file = oracle_list_file.replace('.json', '_processed.json')
    with open(proc_oracle_list_file, 'w+') as f:
        json.dump(_oracle_list, f, indent=2)


if __name__ == '__main__':
    # ds1000_oracle_file = 'data/DS1000/oracle_docs_matched_new.json'
    # pandas_numpy_eval_oracle_file = 'data/pandas-numpy-eval/data/oracle_docs_matched_new.json'
    # conala_oracle_file = 'data/conala/oracle_docs_matched_new.json'

    # main('pandas-numpy-eval')
    # main('DS1000')
    main('conala')

    # process_python_docs()

